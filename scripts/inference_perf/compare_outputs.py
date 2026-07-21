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

# ADR-012 thresholds for SAME-CONFIG comparisons (the code gate).
MIN_EXACT_FRAC = 0.995
MAX_ABS_DELTA = 1
MAX_SCALE_REL_DRIFT = 1e-3
MIN_COSINE = 0.9999

# Cross-config thresholds (--cross-config): runs differing in batch size or
# library stack are not bit-comparable — cuBLAS kernel selection alone moves
# outputs. Measured 2026-07-16 (main@3584 vs Phase-1@7168, 3 chunks x 512M
# values): exact 95.4-98.3%, max|d|=2 at ~0.002% rate, scale drift <= 0.78%
# (2 BF16 ULPs), cosine >= 0.99990, zero NaN-mask mismatches. These bounds
# sit just outside that envelope; anything worse indicates a real defect,
# not config shimmer.
XCFG_MIN_WITHIN_ONE_FRAC = 0.9999
XCFG_MAX_ABS_DELTA = 3
XCFG_MAX_SCALE_REL_DRIFT = 1.6e-2  # 4 BF16 ULPs
XCFG_MIN_COSINE = 0.9999

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
    # Obs-count layers (s2/s1_asc/s1_desc) are deterministic per-pixel counts,
    # independent of batch size or inference — they must match EXACTLY in both
    # comparison classes. Guards the sparse-read fidelity machinery (full-width
    # SAR reads + bundle-sourced S2 obs under the easting-bbox crop).
    obs_count_mismatches: int = 0

    # Scales that are neither NaN (invalid-pixel fill) nor finite-positive
    # (generated): zero, negative, or infinite. These indicate a corrupt
    # artifact in EITHER store and must be zero — without this counter a
    # malformed scale is lumped with "not generated" and skipped (e.g.
    # ref=NaN, test=0.0 would sail through every metric).
    malformed_scales: int = 0

    cross_config: bool = False

    @property
    def passed(self) -> bool:
        """Whether every threshold for the selected comparison class is met."""
        # Zero generated pixels (no finite-positive scale in either store) is an
        # invalid artifact — a no-valid-pixel chunk must be a .skipped marker,
        # not an embeddings zarr. Without this guard n_values == 0 defaults every
        # fraction to 1.0, so an empty/corrupt staged pair would "pass".
        if self.n_values == 0:
            return False
        if self.cross_config:
            return (
                self.within_one_frac >= XCFG_MIN_WITHIN_ONE_FRAC
                and self.max_abs_delta <= XCFG_MAX_ABS_DELTA
                and self.scale_max_rel_drift <= XCFG_MAX_SCALE_REL_DRIFT
                and self.nan_mask_mismatches == 0
                and self.obs_count_mismatches == 0
                and self.malformed_scales == 0
                and self.cosine_min >= XCFG_MIN_COSINE
            )
        return (
            self.exact_frac >= MIN_EXACT_FRAC
            and self.max_abs_delta <= MAX_ABS_DELTA
            and self.scale_max_rel_drift <= MAX_SCALE_REL_DRIFT
            and self.nan_mask_mismatches == 0
            and self.obs_count_mismatches == 0
            and self.malformed_scales == 0
            and self.cosine_min >= MIN_COSINE
        )

    def report(self) -> str:
        """One-line PASS/FAIL summary with all metrics."""
        status = ("PASS" if self.passed else "FAIL") + (" (cross-config)" if self.cross_config else "")
        return (
            f"{self.label}: {status} | exact={self.exact_frac:.6%} "
            f"max|d|={self.max_abs_delta} within1={self.within_one_frac:.6%} "
            f"scale_drift_max={self.scale_max_rel_drift:.2e} "
            f"nan_mismatch={self.nan_mask_mismatches} obs_mismatch={self.obs_count_mismatches} "
            f"malformed_scales={self.malformed_scales} "
            f"cos_min={self.cosine_min:.6f} cos_mean={self.cosine_mean:.6f} "
            f"(n={self.n_values:,})"
        )


def compare_chunk(ref_path: str, test_path: str, label: str, *, cross_config: bool = False) -> ChunkComparison:
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
    malformed_scales = 0
    cosine_min = 1.0
    cosine_sum = 0.0
    cosine_count = 0

    for row0 in range(0, h, ROW_BLOCK):
        rows = slice(row0, min(row0 + ROW_BLOCK, h))
        e_ref = ref["embeddings"].isel(northing=rows).values.astype(np.int16)
        e_test = test["embeddings"].isel(northing=rows).values.astype(np.int16)
        s_ref = ref["scales"].isel(northing=rows).values
        s_test = test["scales"].isel(northing=rows).values

        # "Generated" = a real embedding was written: a finite, POSITIVE scale.
        # Invalid pixels are zero-filled with a NaN scale. Counting them would
        # (a) dilute int8 exactness toward 100% on sparse chunks (most values
        # are guaranteed-equal zeros) and (b) admit malformed scales — a zero,
        # negative, or infinite scale would slip past a bare `~isnan` and make
        # drift/cosine NaN. So compute EVERY metric over generated pixels only.
        gen_ref = np.isfinite(s_ref) & (s_ref > 0)
        gen_test = np.isfinite(s_test) & (s_test > 0)
        nan_mask_mismatches += int((gen_ref != gen_test).sum())
        # A scale must be exactly one of: NaN (invalid pixel) or generated
        # (finite positive). Zero / negative / infinite means a corrupt store;
        # count it explicitly — the "generated pixels only" masking below
        # would otherwise silently skip it (e.g. ref=NaN vs test=0.0 agrees
        # on the mask and touches no metric).
        malformed_scales += int((~np.isnan(s_ref) & ~gen_ref).sum())
        malformed_scales += int((~np.isnan(s_test) & ~gen_test).sum())
        valid = gen_ref & gen_test
        if valid.any():
            # Gather each masked array once per slab — the boolean-mask gathers
            # (especially the (n_valid, D) embeddings) allocate copies.
            er, et = e_ref[valid], e_test[valid]
            sr, st = s_ref[valid], s_test[valid]
            delta = np.abs(er - et)  # (n_valid, D)
            n_values += delta.size
            n_exact += int((delta == 0).sum())
            n_within_one += int((delta <= 1).sum())
            max_abs_delta = max(max_abs_delta, int(delta.max(initial=0)))

            drift = np.abs(sr - st) / np.abs(sr)
            scale_max_rel_drift = max(scale_max_rel_drift, float(drift.max()))

            # Cosine over dequantized embeddings for valid pixels in this slab.
            deq_ref = er.astype(np.float32) * sr[:, None]
            deq_test = et.astype(np.float32) * st[:, None]
            num = (deq_ref * deq_test).sum(axis=1)
            den = np.linalg.norm(deq_ref, axis=1) * np.linalg.norm(deq_test, axis=1)
            nz = den > 0
            if nz.any():
                cos = num[nz] / den[nz]
                cosine_min = min(cosine_min, float(cos.min()))
                cosine_sum += float(cos.sum())
                cosine_count += int(nz.sum())

    # Obs-count layers: deterministic counts, must be EXACT in both classes and
    # PRESENT in both. (H, W) uint16 — small enough to compare whole. A required
    # layer missing from one store (or from BOTH) is a FAILURE, not a silent
    # skip: a run that dropped an obs layer must not pass the gate. A one-sided
    # absence charges the present layer's whole size as mismatches; a two-sided
    # absence (neither run wrote it) charges 1 so obs_count_mismatches still goes
    # non-zero rather than validating nothing.
    obs_count_mismatches = 0
    for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
        in_ref, in_test = var in ref.data_vars, var in test.data_vars
        if in_ref and in_test:
            obs_count_mismatches += int((ref[var].values != test[var].values).sum())
        elif in_ref != in_test:
            present = ref[var] if in_ref else test[var]
            obs_count_mismatches += int(present.size)
        else:
            obs_count_mismatches += 1  # required layer absent from BOTH stores

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
        obs_count_mismatches=obs_count_mismatches,
        malformed_scales=malformed_scales,
        cross_config=cross_config,
    )


def _staged_labels(run_dir: str) -> tuple[set[str], set[str]]:
    """Return (zarr labels, skip-marker labels) staged under a run directory.

    Skip markers (``<label>.skipped``) are returned separately so a chunk that
    produced embeddings in one run but was skip-marked in the other counts as a
    label-set difference, not a silent omission.
    """
    fs, _, (path,) = fsspec.get_fs_token_paths(run_dir)
    # detail=False so s3fs returns path strings, not entry dicts (its default);
    # refresh so a listing cached from before the run staged isn't reused.
    zarr_labels, skip_labels = set(), set()
    for entry in fs.ls(path, detail=False, refresh=True):
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            zarr_labels.add(name.removesuffix(".zarr"))
        elif name.endswith(".skipped"):
            skip_labels.add(name.removesuffix(".skipped"))
    return zarr_labels, skip_labels


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref", help="Reference run dir or single staged chunk .zarr")
    parser.add_argument("test", help="Test run dir or single staged chunk .zarr")
    parser.add_argument(
        "--labels", help="Comma-separated chunk labels — an explicit, intentional subset (skips the label-set check)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="In directory mode, compare the common labels even if the two runs' label sets "
        "differ (default: a label-set mismatch is a failure, so an incompletely-staged run "
        "cannot be reported as passing).",
    )
    parser.add_argument(
        "--cross-config",
        action="store_true",
        help="Runs differ in batch size / library stack: judge against the relaxed "
        "cross-config envelope (within-1 >= 99.99%%, max|d| <= 3, scale <= 1.6%%, "
        "cosine >= 0.9999) instead of the same-config bit-drift gate.",
    )
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
            ref_zarr, ref_skip = _staged_labels(ref_dir)
            test_zarr, test_skip = _staged_labels(test_dir)
            # A label present as BOTH a .zarr and a .skipped marker within one
            # run is an invalid artifact state that assembly's
            # verify_staged_completeness rejects — comparing the .zarr could
            # otherwise report PASS for a run assembly will refuse. Fail
            # regardless of --allow-partial (this is corruption, not partial
            # staging).
            both_staged = (ref_zarr & ref_skip) | (test_zarr & test_skip)
            if both_staged:
                print(
                    "Invalid staging — chunk(s) present as both .zarr and .skipped in a run: "
                    f"{', '.join(sorted(both_staged))}",
                    file=sys.stderr,
                )
                return 1
            # A run is only a valid equivalence reference if it staged the same
            # chunks with the same skip decisions; otherwise a run that dropped
            # or skip-marked chunks could "pass" on whatever remained in common.
            if not args.allow_partial and (ref_zarr != test_zarr or ref_skip != test_skip):
                only_ref = sorted((ref_zarr | ref_skip) - (test_zarr | test_skip))
                only_test = sorted((test_zarr | test_skip) - (ref_zarr | ref_skip))
                flipped = sorted((ref_zarr & test_skip) | (test_zarr & ref_skip))
                print(
                    "Label sets differ between the two runs (use --labels for an intentional subset, "
                    "or --allow-partial to compare the common set anyway):",
                    file=sys.stderr,
                )
                if only_ref:
                    print(f"  only in ref:  {', '.join(only_ref)}", file=sys.stderr)
                if only_test:
                    print(f"  only in test: {', '.join(only_test)}", file=sys.stderr)
                if flipped:
                    print(f"  staged in one, skip-marked in the other: {', '.join(flipped)}", file=sys.stderr)
                return 1
            labels = sorted(ref_zarr & test_zarr)
            if not labels:
                print("No common staged chunk labels between the two runs", file=sys.stderr)
                return 1
        pairs = [(f"{ref_dir}/{lb}.zarr", f"{test_dir}/{lb}.zarr", lb) for lb in labels]

    failures = 0
    for ref_path, test_path, label in pairs:
        cmp_result = compare_chunk(ref_path, test_path, label, cross_config=args.cross_config)
        print(cmp_result.report())
        failures += 0 if cmp_result.passed else 1

    kind = "cross-config" if args.cross_config else "ADR-012 same-config"
    print(f"\n{len(pairs) - failures}/{len(pairs)} chunks pass {kind} thresholds")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
