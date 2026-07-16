"""ADR-012 equivalence gate: compare staged embeddings from two inference runs.

Compares int8 embeddings + per-pixel scales between a reference run and a test
run (same chunks, different code paths) and applies the validated-equivalence
thresholds from ``context_docs/decisions/012-validated-equivalence-for-
inference-outputs.md``:

    int8 values exactly equal            >= 99.5%
    max int8 deviation                   <= 1 level
    per-pixel scale relative drift       <= 0.1%
    cosine similarity (dequantized)      >= 0.9999

Usage::

    # Compare two staged run directories (all chunk labels present in both):
    python scripts/inference_perf/compare_outputs.py \
        s3://bucket/staging/<ref_run_id> s3://bucket/staging/<test_run_id>

    # Compare two specific staged chunk zarrs:
    python scripts/inference_perf/compare_outputs.py \
        s3://.../ref/chunk_0_2.zarr s3://.../test/chunk_0_2.zarr

Exit code 0 = all chunks pass, 1 = any failure. Bit-identical changes
(pipelining, resampler vectorization) should report 100% exact / 0 drift;
harness-gated changes (GRU restructure) must stay within the thresholds.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import fsspec
import numpy as np
import xarray as xr

# ADR-012 thresholds.
MIN_EXACT_FRAC = 0.995
MAX_ABS_DELTA = 1
MAX_SCALE_REL_DRIFT = 1e-3
MIN_COSINE = 0.9999

ROW_BLOCK = 500  # stream comparison in row slabs to bound memory


@dataclass
class ChunkComparison:
    """Aggregated comparison metrics for one chunk."""

    label: str
    n_values: int
    exact_frac: float
    max_abs_delta: int
    within_one_frac: float
    scale_max_rel_drift: float
    nan_mask_mismatches: int
    cosine_min: float
    cosine_mean: float

    @property
    def passed(self) -> bool:
        """Whether every ADR-012 threshold is met."""
        return (
            self.exact_frac >= MIN_EXACT_FRAC
            and self.max_abs_delta <= MAX_ABS_DELTA
            and self.scale_max_rel_drift <= MAX_SCALE_REL_DRIFT
            and self.nan_mask_mismatches == 0
            and self.cosine_min >= MIN_COSINE
        )

    def report(self) -> str:
        """One-line PASS/FAIL summary with all metrics."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.label}: {status} | exact={self.exact_frac:.6%} "
            f"max|d|={self.max_abs_delta} within1={self.within_one_frac:.6%} "
            f"scale_drift_max={self.scale_max_rel_drift:.2e} "
            f"nan_mismatch={self.nan_mask_mismatches} "
            f"cos_min={self.cosine_min:.6f} cos_mean={self.cosine_mean:.6f} "
            f"(n={self.n_values:,})"
        )


def compare_chunk(ref_path: str, test_path: str, label: str) -> ChunkComparison:
    """Stream-compare one staged chunk zarr pair in row slabs."""
    ref = xr.open_zarr(ref_path)
    test = xr.open_zarr(test_path)
    if ref["embeddings"].shape != test["embeddings"].shape:
        msg = f"{label}: shape mismatch {ref['embeddings'].shape} vs {test['embeddings'].shape}"
        raise ValueError(msg)

    h = ref["embeddings"].shape[0]
    n_values = 0
    n_exact = 0
    n_within_one = 0
    max_abs_delta = 0
    scale_max_rel_drift = 0.0
    nan_mask_mismatches = 0
    cosine_min = 1.0
    cosine_sum = 0.0
    cosine_count = 0

    for row0 in range(0, h, ROW_BLOCK):
        rows = slice(row0, min(row0 + ROW_BLOCK, h))
        e_ref = ref["embeddings"].isel(northing=rows).values.astype(np.int16)
        e_test = test["embeddings"].isel(northing=rows).values.astype(np.int16)
        s_ref = ref["scales"].isel(northing=rows).values
        s_test = test["scales"].isel(northing=rows).values

        delta = np.abs(e_ref - e_test)
        n_values += delta.size
        n_exact += int((delta == 0).sum())
        n_within_one += int((delta <= 1).sum())
        max_abs_delta = max(max_abs_delta, int(delta.max(initial=0)))

        ref_nan = np.isnan(s_ref)
        test_nan = np.isnan(s_test)
        nan_mask_mismatches += int((ref_nan != test_nan).sum())
        valid = ~ref_nan & ~test_nan
        if valid.any():
            drift = np.abs(s_ref[valid] - s_test[valid]) / np.abs(s_ref[valid])
            scale_max_rel_drift = max(scale_max_rel_drift, float(drift.max()))

            # Cosine over dequantized embeddings for valid pixels in this slab.
            deq_ref = e_ref[valid].astype(np.float32) * s_ref[valid][:, None]
            deq_test = e_test[valid].astype(np.float32) * s_test[valid][:, None]
            num = (deq_ref * deq_test).sum(axis=1)
            den = np.linalg.norm(deq_ref, axis=1) * np.linalg.norm(deq_test, axis=1)
            nz = den > 0
            if nz.any():
                cos = num[nz] / den[nz]
                cosine_min = min(cosine_min, float(cos.min()))
                cosine_sum += float(cos.sum())
                cosine_count += int(nz.sum())

    return ChunkComparison(
        label=label,
        n_values=n_values,
        exact_frac=n_exact / n_values if n_values else 1.0,
        max_abs_delta=max_abs_delta,
        within_one_frac=n_within_one / n_values if n_values else 1.0,
        scale_max_rel_drift=scale_max_rel_drift,
        nan_mask_mismatches=nan_mask_mismatches,
        cosine_min=cosine_min if cosine_count else 1.0,
        cosine_mean=cosine_sum / cosine_count if cosine_count else 1.0,
    )


def _staged_labels(run_dir: str) -> set[str]:
    """List chunk labels staged under a run directory (``<label>.zarr`` entries)."""
    fs, _, (path,) = fsspec.get_fs_token_paths(run_dir)
    return {
        entry.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")
        for entry in fs.ls(path)
        if entry.rstrip("/").endswith(".zarr")
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref", help="Reference run dir or single staged chunk .zarr")
    parser.add_argument("test", help="Test run dir or single staged chunk .zarr")
    parser.add_argument("--labels", help="Comma-separated chunk labels (default: all common)")
    args = parser.parse_args(argv)

    if args.ref.rstrip("/").endswith(".zarr"):
        label = args.ref.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")
        pairs = [(args.ref, args.test, label)]
    else:
        ref_dir = args.ref.rstrip("/")
        test_dir = args.test.rstrip("/")
        if args.labels:
            labels = sorted(args.labels.split(","))
        else:
            labels = sorted(_staged_labels(ref_dir) & _staged_labels(test_dir))
            if not labels:
                print("No common staged chunk labels between the two runs", file=sys.stderr)
                return 1
        pairs = [(f"{ref_dir}/{lb}.zarr", f"{test_dir}/{lb}.zarr", lb) for lb in labels]

    failures = 0
    for ref_path, test_path, label in pairs:
        cmp_result = compare_chunk(ref_path, test_path, label)
        print(cmp_result.report())
        failures += 0 if cmp_result.passed else 1

    print(f"\n{len(pairs) - failures}/{len(pairs)} chunks pass ADR-012 thresholds")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
