"""Time and size the production forward pass on whatever GPU it is run on.

    python -m tessera_embeddings.profiling.inference.forward_bench

**Why this exists.** Two questions about a candidate GPU cannot be answered from a
campaign run, and both decide whether the card is usable at all:

1. **Does the deepest possible sequence fit?** Peak VRAM scales with sequence
   length, the model's hard ceiling is ``max(num_obs_checkpoints)`` = 256, and no
   real cell reaches it — ``iowa_epsg5070`` tops out near 120 and the deepest
   ``t_kept`` ever recorded anywhere is 206. So a fleet run cannot exercise the
   worst case, and "it did not OOM on the chunks we happened to draw" is not the
   same claim as "it cannot OOM". This sweeps sequence length to the ceiling and
   reports the peak, or the OOM, at each rung.
2. **How fast is the card at the work, with nothing else in the way?** An
   end-to-end tok/sec figure carries S3 weather, host CPU feed, chunk geography
   and optical depth. Two cards measured on one cell in one run remove most of
   that by construction, but not all of it. A synthetic forward at a fixed
   ``(B, T)`` removes all of it: same tensors, same shapes, same dtype, no I/O.

It is deliberately **not** a correctness tool. Weights are random — the model is
built by :func:`_build_inference_model` and never loaded from a checkpoint —
because FLOPs, memory and time depend on shapes and dtype, not on values. Nothing
it prints says anything about embedding quality.

Every result line is machine-readable::

    FORWARD_BENCH: {"t_s2": 256, "vram_peak_gib": 16.9, "status": "ok", ...}

so a caller can parse it out of an SSM command's output or a CloudWatch stream
without a regex over prose.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.models.builder import _build_inference_model
from tessera_embeddings.inference.profiling import _card_ceiling, transformer_flops

logger = logging.getLogger(__name__)

#: Sequence lengths swept by default. The last one is the model's own ceiling
#: (``max(num_obs_checkpoints)``), which is the only value that answers "can this
#: card ever OOM on this model" — every real chunk buckets at or below it.
DEFAULT_SEQ_LENS = (8, 32, 64, 96, 128, 160, 192, 224, 256)


def _one_rung(
    model: torch.nn.Module,
    config: InferenceConfig,
    *,
    batch_size: int,
    t_s2: int,
    t_s1: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    """Time ``iters`` forwards at one ``(batch_size, t_s2, t_s1)`` and report peaks.

    Returns a dict with ``status`` ``"ok"`` or ``"oom"``. An OOM is a RESULT, not
    an error: the whole point of sweeping to the model's ceiling is to find where
    a smaller card stops, and a traceback would lose every rung measured before
    it. The caching allocator's pool is dropped after one so the next rung starts
    from a clean device.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        s2 = torch.randn(batch_size, t_s2, 11, device=device, dtype=dtype)
        # The merged S1 stream is 2 bands + DOY. A radar-free chunk still runs the
        # S1 backbone (both backbones always execute; `fusion_method="concat"`),
        # so t_s1=0 is not a shortcut and is not offered.
        s1 = torch.randn(batch_size, max(t_s1, 1), 3, device=device, dtype=dtype)
        with torch.no_grad():
            for _ in range(warmup):
                model(s2, s1)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            for _ in range(iters):
                model(s2, s1)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t0
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        torch.cuda.empty_cache()
        return {
            "t_s2": t_s2,
            "t_s1": t_s1,
            "status": "oom",
            "vram_peak_gib": round(peak, 2),
            "vram_reserved_peak_gib": round(reserved, 2),
            "error": str(exc).splitlines()[0][:200],
        }

    ms = elapsed / iters * 1000
    tokens = batch_size * (t_s2 + max(t_s1, 1))
    # Dimensions from the CONFIG, not read back off the model: this is the same
    # arithmetic `log_effective_tflops` does in production
    # (`d_model = latent_dim * 4`), and taking them from the same source is what
    # makes the two figures comparable.
    flops = transformer_flops(
        batch_size,
        t_s2,
        max(t_s1, 1),
        d_model=config.latent_dim * 4,
        dim_feedforward=config.dim_feedforward,
        num_layers=config.num_encoder_layers,
    )
    result = {
        "t_s2": t_s2,
        "t_s1": t_s1,
        "status": "ok",
        "forward_ms": round(ms, 1),
        # Combined tokens per second: optical plus radar timesteps, the same
        # basis the campaign's per-chunk figures use. px/sec is deliberately not
        # reported — it mixes machine speed with observation depth.
        "tok_per_sec": round(tokens / (ms / 1000)),
        "eff_tflops": round(flops / (ms / 1000) / 1e12, 2),
        "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "vram_reserved_peak_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }
    del s2, s1
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    """Sweep sequence length on this host's GPU and print one JSON line per rung."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=None, help="default: the production InferenceConfig value")
    ap.add_argument(
        "--seq-lens",
        default=",".join(str(t) for t in DEFAULT_SEQ_LENS),
        help="comma-separated S2 sequence lengths to sweep",
    )
    ap.add_argument(
        "--s1-frac",
        type=float,
        default=0.4,
        help=(
            "S1 sequence length as a fraction of the S2 one. Default 0.4 is the "
            "campaign's shape (iowa_epsg5070: 103 ascending against 263 optical)."
        ),
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if not torch.cuda.is_available():
        print("FORWARD_BENCH_ERROR: no CUDA device")
        return 1

    device = torch.device("cuda")
    # The time window is required by the config and irrelevant here: nothing is
    # read from a store, so it only has to parse. Every dimension that decides
    # FLOPs, memory and time comes from the config's own defaults, which is the
    # point — a benchmark on hand-picked dimensions measures a different model.
    config = InferenceConfig(time_window=parse_time_window("December 2024"))
    batch_size = args.batch_size or config.batch_size
    # BF16 unconditionally, matching production: `inference.py` picks bf16 whenever
    # `torch.cuda.is_bf16_supported()`, and every card under consideration (L40S,
    # L4, A10G — SM 8.6 and 8.9) reports it. A card that did not would be a
    # different measurement, so it refuses rather than silently benchmarking fp16.
    if not torch.cuda.is_bf16_supported():
        print("FORWARD_BENCH_ERROR: this device has no bf16; production dtype is bf16")
        return 1
    dtype = torch.bfloat16

    model = _build_inference_model(config, device).to(dtype).eval()
    name = torch.cuda.get_device_name(0)
    ceiling = _card_ceiling(name)
    header = {
        "device": name,
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "batch_size": batch_size,
        "dtype": "bfloat16",
        "model_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "card_bf16_tflops": ceiling[1] if ceiling else None,
        "card_bandwidth_gbs": ceiling[2] if ceiling else None,
    }
    print("FORWARD_BENCH_HEADER: " + json.dumps(header, sort_keys=True))

    for t_s2 in (int(t) for t in args.seq_lens.split(",") if t.strip()):
        row = _one_rung(
            model,
            config,
            batch_size=batch_size,
            t_s2=t_s2,
            t_s1=max(round(t_s2 * args.s1_frac), 1),
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        eff = row.get("eff_tflops")
        if ceiling and isinstance(eff, (int, float)):
            row["frac_of_card_tflops"] = round(eff / ceiling[1], 3)
        print("FORWARD_BENCH: " + json.dumps(row, sort_keys=True))
        sys.stdout.flush()
        if row["status"] == "oom":
            # Deeper rungs would OOM too, and each one costs a minute of GPU time
            # to prove it. The first refusal is the answer.
            print(f"FORWARD_BENCH_STOP: OOM at t_s2={t_s2}; deeper rungs not attempted")
            break
    return 0


if __name__ == "__main__":  # pragma: no cover — module entrypoint
    raise SystemExit(main())
