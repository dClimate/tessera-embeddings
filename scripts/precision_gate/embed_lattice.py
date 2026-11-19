"""Embed ADR 012's lattice on this card for the precision gate, one fresh production model per variant.

Each variant is built through ``build_inference_model`` and ``_prepare_gpu``, so it runs the card's
production forward: on an L40S or an L4, chunks replaying captured CUDA graphs
(``GRAPH_CHUNK_PIXELS``); on any other card, the whole batch. A fresh model per variant matters,
because graphs are cached on the model per bucket shape, and a reused model would replay the first
variant's graphs for every later one.

Variants:
  bf16               the bf16 forward on this card (a model built for a bf16 store)
  bf16_whole         the bf16 forward with chunking off (the whole batch), as before PR #273
  fp8                the shipped FP8 forward: a model built for an FP8 store, which on an L40S runs QKV (after the
                     first layer) and the first FFN projection in E4M3 (``student_v2.build_fp8_layers``), and on any
                     other card the bf16 forward
  fp8_all4           all four transformer linears in E4M3 through ``torch._scaled_mm``, row-wise activation
                     and per-channel weight scales, quantised from the bf16 activations (ADR 043's record)
  fp8_stage1         QKV and FFN up only, quantised from the fp32 LayerNorm output as the planned fused kernel
                     emits it; out-proj and FFN down stay bf16 (the record's Stage 1)
  fp8_stage1_whole   fp8_stage1 with chunking off, to reproduce the record bit for bit

``fp8_all4`` and ``fp8_stage1`` are the record's stand-ins, swapped in for ``student_v2._encoder_layer_forward``
on a bf16 model before its first forward; they need FP8 tensor cores (compute capability 8.9 or later).
Embeddings are quantised with ``quantize_rows``, as the record's were.

usage: embed_lattice.py <card> <lattice_dir> <out_dir> <tile> [<tile> ...] --variants bf16,fp8_stage1,...
       -> <out_dir>/lattice_<card>_<tile>.npz, one file per tile with every variant's _q and _s arrays
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias, as in student_v2
from torch import Tensor, nn

from tessera_embeddings.config.inference import InferenceConfig, Precision, batch_size_for_gpu
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.inference import _prepare_gpu
from tessera_embeddings.inference.models import student_v2
from tessera_embeddings.inference.models.builder import build_inference_model
from tessera_embeddings.inference.quantization import quantize_rows

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
PRODUCTION_LAYER = student_v2._encoder_layer_forward
ROW_BLOCK = 65536


def quant_rows_e4m3(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise E4M3 in blocks of rows: each row's scale depends only on that row, so blocking is exact."""
    q = torch.empty(x2.shape, dtype=E4M3, device=x2.device)
    s = torch.empty((x2.shape[0], 1), dtype=torch.float32, device=x2.device)
    for r in range(0, x2.shape[0], ROW_BLOCK):
        blk = x2[r : r + ROW_BLOCK].float()
        s[r : r + ROW_BLOCK] = (blk.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
        q[r : r + ROW_BLOCK] = (blk / s[r : r + ROW_BLOCK]).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return q, s


class FP8Linear:
    """E4M3 linear: weights per output channel (quantised on first use), activations per row.

    The first use is a warm-up forward, outside any graph capture, so the capture only reads the cache.
    """

    def __init__(self) -> None:
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, x32: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x @ w.T + b`` through ``_scaled_mm``; ``x32``, when given, is the fp32 value to quantise ``x`` from."""
        if id(w) not in self.cache:
            w32 = w.float()
            sw = (w32.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
            self.cache[id(w)] = ((w32 / sw).to(E4M3), sw.t().contiguous())
        wq, sw = self.cache[id(w)]
        shp = x.shape
        src = (x32 if x32 is not None else x).reshape(-1, shp[-1])
        m = src.shape[0]
        pad = (-m) % 16
        if pad:
            src = F.pad(src, (0, 0, 0, pad))
        xq, sa = quant_rows_e4m3(src)
        y = torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16, use_fast_accum=True)
        return (y[:m] + b).reshape(*shp[:-1], -1)


def _attention(layer: nn.TransformerEncoderLayer, x: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
    b, t, e = x.shape
    attn = layer.self_attn
    q, k, v = qkv.view(b, t, 3, attn.num_heads, e // attn.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, e)


def fp8_all4_layer() -> Callable:
    """The record's all-four-linears layer forward."""
    lin = FP8Linear()

    def layer_fwd(
        layer: nn.TransformerEncoderLayer, x: Tensor, add_layer_norm: Callable[..., Tensor], _fp8: object = None
    ) -> Tensor:
        attn = layer.self_attn
        o = _attention(layer, x, lin(x, attn.in_proj_weight, attn.in_proj_bias))
        n1, n2 = layer.norm1, layer.norm2
        x = add_layer_norm(x, lin(o, attn.out_proj.weight, attn.out_proj.bias), n1.weight, n1.bias, n1.eps)
        h = torch.relu(lin(x, layer.linear1.weight, layer.linear1.bias))
        return add_layer_norm(x, lin(h, layer.linear2.weight, layer.linear2.bias), n2.weight, n2.bias, n2.eps)

    return layer_fwd


def fp8_stage1_layer() -> Callable:
    """The record's Stage 1 layer forward: FP8 QKV and FFN up, fed from the fp32 add + LayerNorm output."""
    lin = FP8Linear()
    last: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # id of a bf16 LayerNorm output -> (it, its fp32 value)

    def add_ln(x: Tensor, r: Tensor, w: Tensor, bias: Tensor, eps: float) -> Tensor:
        y32 = F.layer_norm(x.float() + r.float(), (x.shape[-1],), w.float(), bias.float(), eps)
        yb = y32.to(torch.bfloat16)
        last.clear()
        last[id(yb)] = (yb, y32)
        return yb

    def src32(x: Tensor) -> Tensor | None:
        hit = last.get(id(x))
        return hit[1] if hit is not None and hit[0] is x else None

    # Ignores the production add + LayerNorm it is passed: it needs the fp32 output, so it runs its own.
    def layer_fwd(
        layer: nn.TransformerEncoderLayer, x: Tensor, _add_layer_norm: Callable[..., Tensor], _fp8: object = None
    ) -> Tensor:
        attn = layer.self_attn
        o = _attention(layer, x, lin(x, attn.in_proj_weight, attn.in_proj_bias, src32(x)))
        n1, n2 = layer.norm1, layer.norm2
        x = add_ln(x, F.linear(o, attn.out_proj.weight, attn.out_proj.bias), n1.weight, n1.bias, n1.eps)
        h = torch.relu(lin(x, layer.linear1.weight, layer.linear1.bias, src32(x)))
        return add_ln(x, F.linear(h, layer.linear2.weight, layer.linear2.bias), n2.weight, n2.bias, n2.eps)

    return layer_fwd


# variant -> (layer forward factory, chunking on, the store precision the model is built for)
VARIANTS: dict[str, tuple[Callable[[], Callable], bool, Precision]] = {
    "bf16": (lambda: PRODUCTION_LAYER, True, "bf16"),
    "bf16_whole": (lambda: PRODUCTION_LAYER, False, "bf16"),
    "fp8": (lambda: PRODUCTION_LAYER, True, "fp8"),
    "fp8_all4": (fp8_all4_layer, True, "bf16"),
    "fp8_stage1": (fp8_stage1_layer, True, "bf16"),
    "fp8_stage1_whole": (fp8_stage1_layer, False, "bf16"),
}


def embed(cfg: InferenceConfig, ckpt: str, lattice: dict, variant: str, batch: int) -> tuple[np.ndarray, dict]:
    """One variant over one tile's lattice, on a freshly built production model."""
    dev = torch.device("cuda")
    make_layer, chunked, precision = VARIANTS[variant]
    student_v2._encoder_layer_forward = make_layer()
    try:
        model = build_inference_model(dataclasses.replace(cfg, precision=precision), dev, checkpoint_path=ckpt)
        _prepare_gpu(model, dev)
        if not chunked:
            model.chunk_pixels = None
        zs = []
        with torch.no_grad():
            for b in lattice["batches"]:
                for s in range(0, b["s2"].shape[0], batch):
                    zs.append(model(b["s2"][s : s + batch].to(dev), b["s1"][s : s + batch].to(dev)).float().cpu())
        fp8_built = any(e.fp8_layers is not None for e in (model.s2_backbone, model.s1_backbone))
        meta = {"chunk_pixels": model.chunk_pixels, "graphs": len(model._graphs), "fp8_layers": fp8_built}
    finally:
        student_v2._encoder_layer_forward = PRODUCTION_LAYER
    del model
    torch.cuda.empty_cache()
    return torch.cat(zs).numpy(), meta


def _git_sha() -> str:
    """The code under test: ``TE_CODE_SHA`` on a box running a ``git archive`` of it, else the checkout's HEAD."""
    if os.environ.get("TE_CODE_SHA"):
        return os.environ["TE_CODE_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("card")
    p.add_argument("lattice_dir")
    p.add_argument("out_dir")
    p.add_argument("tiles", nargs="+")
    p.add_argument("--variants", required=True, help=f"comma-separated, from {sorted(VARIANTS)}")
    p.add_argument("--checkpoint", default="/opt/ckpt/v2_student_large.pt")
    p.add_argument("--batch", type=int, help="sub-batch size; default: production's for this card")
    a = p.parse_args(argv)
    variants = a.variants.split(",")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        p.error(f"unknown variants {sorted(unknown)}")
    if any(v.startswith("fp8") for v in variants) and torch.cuda.get_device_capability() < (8, 9):
        p.error("FP8 variants need compute capability 8.9 or later")

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = InferenceConfig(
        time_window=parse_time_window("December 2024"),
        checkpoint_path=a.checkpoint,
        inputs_bucket="s3://x",
        output_bucket="s3://x",
        num_gpus=1,
        model_version="v2-large",
    )
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch = a.batch or batch_size_for_gpu(cfg.batch_size, total_gib)
    name = torch.cuda.get_device_name(0)
    print(f"{a.card}: {name}, torch {torch.__version__}, batch {batch}, code {_git_sha()}", flush=True)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile in a.tiles:
        lattice = torch.load(Path(a.lattice_dir) / f"lattice_{tile}.pt", weights_only=False)
        out = {
            "Y": lattice["Y"],
            "X": lattice["X"],
            "card": name,
            "torch": torch.__version__,
            "code": _git_sha(),
            "batch": batch,
        }
        for v in variants:
            t0 = time.monotonic()
            z, meta = embed(cfg, a.checkpoint, lattice, v, batch)
            out[f"{v}_q"], out[f"{v}_s"] = quantize_rows(z)
            out[f"{v}_chunk_pixels"] = -1 if meta["chunk_pixels"] is None else meta["chunk_pixels"]
            out[f"{v}_fp8_layers"] = meta["fp8_layers"]
            print(
                f"{tile} {v}: {time.monotonic() - t0:.0f}s, chunk_pixels {meta['chunk_pixels']}, "
                f"{meta['graphs']} graphs, FP8 layers {meta['fp8_layers']}, {z.shape[0]:,} px",
                flush=True,
            )
        # The gate scores against "bf16": it must be a bf16 forward, and on an FP8 card the shipped forward must differ.
        if out.get("bf16_fp8_layers"):
            raise RuntimeError("the bf16 reference was built with FP8 layers")
        if out.get("fp8_fp8_layers") and "bf16_q" in out and np.array_equal(out["fp8_q"], out["bf16_q"]):
            raise RuntimeError("the fp8 variant embedded exactly as bf16: its FP8 layers did not run")
        np.savez(out_dir / f"lattice_{a.card}_{tile}.npz", **out)
        print(f"PROGRESS tile {tile} done", flush=True)
        del lattice
    print("EMBED COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
