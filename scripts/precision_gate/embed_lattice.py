"""Embed ADR 012's lattice with the production forward of the code this runs from, for the precision gate.

The model is built through ``build_inference_model`` and ``_prepare_gpu``, so it runs the card's
production forward, CUDA graph replay included. ``side`` names what the embedding stands for in the
gate: ``l40s``, ``a10g`` and ``l4`` are the reference (today's code on each card), ``cand`` is the
candidate (the branch under test, on an L40S). The script refuses a box whose GPU is not the side's
card. Embeddings are quantised with ``quantize_rows``, as the store's are.

usage: embed_lattice.py <side> <lattice_dir> <out_dir> <tile> [<tile> ...]
       -> <out_dir>/lattice_<side>_<tile>.npz: q, s, Y, X and what produced them
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tessera_embeddings.config.inference import InferenceConfig, batch_size_for_gpu
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.inference import _prepare_gpu
from tessera_embeddings.inference.models.builder import build_inference_model
from tessera_embeddings.inference.quantization import quantize_rows

SIDES = {"l40s": "NVIDIA L40S", "a10g": "NVIDIA A10G", "l4": "NVIDIA L4", "cand": "NVIDIA L40S"}
"""The sides calibrate.py reads, by the device name each must run on."""


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
    p.add_argument("side", choices=sorted(SIDES))
    p.add_argument("lattice_dir")
    p.add_argument("out_dir")
    p.add_argument("tiles", nargs="+")
    p.add_argument("--checkpoint", default="/opt/ckpt/v2_student_large.pt")
    p.add_argument("--batch", type=int, help="sub-batch size; default: production's for this card")
    a = p.parse_args(argv)
    name = torch.cuda.get_device_name(0)
    if name != SIDES[a.side]:  # the side names the file calibrate.py reads, so it must be the real card
        p.error(f"side {a.side} runs on {SIDES[a.side]}, but this box has {name}")

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
    dev = torch.device("cuda")
    model = build_inference_model(cfg, dev, checkpoint_path=a.checkpoint)
    _prepare_gpu(model, dev)
    print(f"{a.side}: {name}, torch {torch.__version__}, batch {batch}, code {_git_sha()}", flush=True)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile in a.tiles:
        t0 = time.monotonic()
        lattice = torch.load(Path(a.lattice_dir) / f"lattice_{tile}.pt", weights_only=False)
        zs = []
        with torch.no_grad():
            for b in lattice["batches"]:
                for s in range(0, b["s2"].shape[0], batch):
                    zs.append(model(b["s2"][s : s + batch].to(dev), b["s1"][s : s + batch].to(dev)).float().cpu())
        q, scale = quantize_rows(torch.cat(zs).numpy())
        np.savez(
            out_dir / f"lattice_{a.side}_{tile}.npz",
            q=q,
            s=scale,
            Y=lattice["Y"],
            X=lattice["X"],
            card=name,
            torch=torch.__version__,
            code=_git_sha(),
            batch=batch,
            chunk_pixels=-1 if model.chunk_pixels is None else model.chunk_pixels,
        )
        print(f"PROGRESS tile {tile} done: {time.monotonic() - t0:.0f}s, {q.shape[0]:,} px", flush=True)
        del lattice
    print("EMBED COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
