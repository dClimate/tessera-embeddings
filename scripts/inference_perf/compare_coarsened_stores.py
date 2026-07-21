"""Bit-identity check for two coarsened (500 m) embedding icechunk stores.

Sibling to :mod:`compare_outputs` (which gates the *full-resolution, int8 +
scales, per-chunk staged* runs against the ADR-012 validated-equivalence
thresholds). This script targets the **coarsened** aggregate stores instead:
a single icechunk store per run whose ``embeddings`` variable is already
float32 / pre-dequantized (no ``scales`` variable) and whose obs-count layers
are uint32. See ``scripts/plot_corn_sample_500m_regions.py`` for that format.

The question here is stricter and simpler than ADR-012: are the two stores
**bit-identical**? "Bit-identical" means every array — coordinates and data —
matches its counterpart byte-for-byte (floats compared on their raw bit
pattern, so a ``-0.0`` vs ``0.0`` or a differing NaN payload counts as a
mismatch). Because the inference speed-up campaign only guarantees *validated
equivalence* for the int8 outputs (batch-size changes move cuBLAS kernel
selection, so the 10 m int8 values differ by <=1 level on a small fraction of
pixels), the dequantized-then-mean-pooled 500 m floats are unlikely to be
bit-identical. So when they are not, the script still reports HOW they diverge
(max/mean abs diff, exactness fractions, per-pixel cosine, NaN-mask agreement)
so the divergence can be judged as either "quantization shimmer" or a real
defect.

Usage::

    AWS_PROFILE=yield python scripts/inference_perf/compare_coarsened_stores.py \
        s3://arbol-tessera-embeddings-dev/embeddings/iowa_epsg5070-reference_500m.zarr \
        s3://arbol-tessera-embeddings-dev/embeddings/iowa_epsg5070-inference-speedup-phase5_500m.zarr

Exit code 0 = fully bit-identical, 1 = any difference (structural, coordinate,
or array). Data volume is small (~0.4 GB/store for Iowa), so the full arrays
are compared by default; ``--sample-rows N`` compares an evenly-strided subset
of northing rows if pointed at a much larger store.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from tessera_embeddings.storage.zarr_store import open_store

# Stream the (northing, easting, band) arrays in row slabs to bound peak
# memory. Sized to the stores' 384-row northing chunk so each on-disk chunk is
# downloaded exactly once (a smaller, unaligned slab re-reads a straddled chunk
# on the next slab, ~doubling the S3 transfer). One 384-row embeddings slab is
# ~384 * 1066 * 128 * 4 B ~= 210 MB per store; two stores + uint bit-views stay
# comfortably under ~0.6 GB.
ROW_BLOCK = 384

# Obs-count / provenance layers are keyed by name so a variable present in one
# store but absent in the other is caught rather than silently skipped.
OBS_VARS = ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count")

# Root attrs that legitimately differ between two runs of the same ROI — these
# are provenance, not data, so a mismatch here is reported but not a failure.
BENIGN_ATTR_DIFFS = frozenset({"run_id"})

# Abs-diff thresholds reported for float embeddings when NOT bit-identical, so
# a near-miss (quantization shimmer) is distinguishable from a real divergence.
ABS_DIFF_THRESHOLDS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)


@dataclass
class VarComparison:
    """Bit-identity + numeric-divergence metrics for one data variable."""

    name: str
    dtype: str
    n: int = 0
    n_bit_equal: int = 0
    nan_mask_mismatches: int = 0  # positions where exactly one store is NaN
    max_abs_diff: float = 0.0
    sum_abs_diff: float = 0.0
    n_finite_pairs: int = 0  # elements finite in BOTH stores (abs-diff domain)
    # element count at each abs-diff threshold, over finite pairs
    within: dict[float, int] = field(default_factory=dict)
    cosine_min: float = 1.0
    cosine_sum: float = 0.0
    cosine_count: int = 0

    @property
    def bit_identical(self) -> bool:
        return self.n > 0 and self.n_bit_equal == self.n

    @property
    def mean_abs_diff(self) -> float:
        return self.sum_abs_diff / self.n_finite_pairs if self.n_finite_pairs else 0.0

    @property
    def cosine_mean(self) -> float:
        return self.cosine_sum / self.cosine_count if self.cosine_count else 1.0

    def report(self) -> str:
        """Multi-line summary: bit verdict first, numeric divergence if needed."""
        head = f"  {self.name} ({self.dtype}): "
        if self.bit_identical:
            return head + f"BIT-IDENTICAL ({self.n:,} elements)"
        bit_frac = self.n_bit_equal / self.n if self.n else 1.0
        lines = [
            head + f"NOT bit-identical — {self.n - self.n_bit_equal:,} of {self.n:,} "
            f"elements differ ({bit_frac:.6%} bit-equal)",
            f"      max|d|={self.max_abs_diff:.6g} mean|d|={self.mean_abs_diff:.6g} "
            f"nan_mask_mismatch={self.nan_mask_mismatches:,}",
        ]
        if self.within:
            frac = ", ".join(
                f"|d|<={t:g}:{self.within.get(t, 0) / self.n_finite_pairs:.4%}"
                for t in ABS_DIFF_THRESHOLDS
                if self.n_finite_pairs
            )
            lines.append(f"      finite-pair abs-diff cdf: {frac}")
        if self.cosine_count:
            lines.append(f"      per-pixel cosine: min={self.cosine_min:.8f} mean={self.cosine_mean:.8f}")
        return "\n".join(lines)


def _bit_equal(a: np.ndarray, b: np.ndarray) -> int:
    """Count byte-for-byte equal elements (raw bit pattern, NaN-payload aware)."""
    ua = a.view(_uint_of(a.dtype))
    ub = b.view(_uint_of(b.dtype))
    return int((ua == ub).sum())


def _uint_of(dtype: np.dtype) -> np.dtype:
    """Return an unsigned integer dtype of the same width for bitwise compare."""
    return np.dtype(f"u{dtype.itemsize}")


def compare_coords(ref: xr.Dataset, test: xr.Dataset) -> list[str]:
    """Return a list of coordinate/structure mismatches (empty = identical)."""
    problems: list[str] = []
    if dict(ref.sizes) != dict(test.sizes):
        problems.append(f"dims differ: {dict(ref.sizes)} vs {dict(test.sizes)}")
    ref_coords, test_coords = set(ref.coords), set(test.coords)
    if ref_coords != test_coords:
        problems.append(f"coord set differs: only-ref={ref_coords - test_coords} only-test={test_coords - ref_coords}")
    for c in sorted(ref_coords & test_coords):
        a, b = ref[c].values, test[c].values
        if a.shape != b.shape:
            problems.append(f"coord '{c}' shape {a.shape} vs {b.shape}")
        elif a.dtype != b.dtype:
            problems.append(f"coord '{c}' dtype {a.dtype} vs {b.dtype}")
        elif a.dtype.kind in "fiu":
            if _bit_equal(a, b) != a.size:
                problems.append(f"coord '{c}' values not bit-identical")
        elif not np.array_equal(a, b):  # datetime64, strings, etc.
            problems.append(f"coord '{c}' values differ")
    return problems


def compare_attrs(ref: xr.Dataset, test: xr.Dataset) -> tuple[list[str], list[str]]:
    """Return (benign, meaningful) attribute-difference messages."""
    benign: list[str] = []
    meaningful: list[str] = []
    for key in sorted(set(ref.attrs) | set(test.attrs)):
        rv, tv = ref.attrs.get(key, "<absent>"), test.attrs.get(key, "<absent>")
        if np.all(rv == tv):
            continue
        msg = f"{key}: {rv!r} vs {tv!r}"
        (benign if key in BENIGN_ATTR_DIFFS else meaningful).append(msg)
    return benign, meaningful


def _accumulate_float(cmp: VarComparison, a: np.ndarray, b: np.ndarray) -> None:
    """Fold one float slab into the running numeric-divergence metrics."""
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    cmp.nan_mask_mismatches += int((nan_a != nan_b).sum())
    both = ~nan_a & ~nan_b
    if both.any():
        diff = np.abs(a[both] - b[both])
        cmp.n_finite_pairs += int(diff.size)
        cmp.max_abs_diff = max(cmp.max_abs_diff, float(diff.max()))
        cmp.sum_abs_diff += float(diff.sum())
        for t in ABS_DIFF_THRESHOLDS:
            cmp.within[t] = cmp.within.get(t, 0) + int((diff <= t).sum())
    # Per-pixel cosine over the trailing band axis (finite in both stores).
    if a.ndim >= 2:
        band = a.shape[-1]
        ra, rb = a.reshape(-1, band), b.reshape(-1, band)
        px = np.isfinite(ra).all(1) & np.isfinite(rb).all(1)
        if px.any():
            ra, rb = ra[px], rb[px]
            num = (ra * rb).sum(1)
            den = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
            nz = den > 0
            if nz.any():
                cos = num[nz] / den[nz]
                cmp.cosine_min = min(cmp.cosine_min, float(cos.min()))
                cmp.cosine_sum += float(cos.sum())
                cmp.cosine_count += int(nz.sum())


def _accumulate_int(cmp: VarComparison, a: np.ndarray, b: np.ndarray) -> None:
    """Fold one integer slab into the running metrics (bitwise == numeric)."""
    diff = np.abs(a.astype(np.int64) - b.astype(np.int64))
    cmp.n_finite_pairs += int(diff.size)
    cmp.max_abs_diff = max(cmp.max_abs_diff, float(diff.max(initial=0)))
    cmp.sum_abs_diff += float(diff.sum())


def compare_variable(
    ref: xr.Dataset, test: xr.Dataset, name: str, rows: list[slice]
) -> VarComparison:
    """Stream-compare one data variable across the given northing row slabs."""
    da_ref, da_test = ref[name], test[name]
    cmp = VarComparison(name=name, dtype=str(da_ref.dtype))
    is_float = da_ref.dtype.kind == "f"
    for rows_slice in rows:
        a = da_ref.isel(northing=rows_slice).values
        b = da_test.isel(northing=rows_slice).values
        cmp.n += a.size
        cmp.n_bit_equal += _bit_equal(a, b)
        if is_float:
            _accumulate_float(cmp, a, b)
        else:
            _accumulate_int(cmp, a, b)
    return cmp


def _row_slabs(height: int, sample_rows: int | None) -> list[slice]:
    """Row slabs covering [0, height), or an evenly-strided sampled subset."""
    if sample_rows and sample_rows < height:
        idx = np.linspace(0, height - 1, sample_rows, dtype=int)
        # Group the sampled indices into contiguous-ish slabs is overkill for a
        # sample; one 1-row slab per sampled index reads only what's needed.
        return [slice(int(i), int(i) + 1) for i in idx]
    return [slice(r0, min(r0 + ROW_BLOCK, height)) for r0 in range(0, height, ROW_BLOCK)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code (0 = bit-identical)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref", help="Reference coarsened store (s3:// icechunk path)")
    parser.add_argument("test", help="Test coarsened store (s3:// icechunk path)")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Compare only this many evenly-strided northing rows (default: all rows).",
    )
    args = parser.parse_args(argv)

    # chunks=None: slice lazily on metadata, no dask graph (see open_store docs).
    ref = open_store(args.ref, chunks=None)
    test = open_store(args.test, chunks=None)

    print(f"ref : {args.ref}")
    print(f"test: {args.test}\n")

    coord_problems = compare_coords(ref, test)
    if coord_problems:
        print("STRUCTURE / COORDINATES — NOT identical:")
        for p in coord_problems:
            print(f"  - {p}")
        print("\nVERDICT: NOT bit-identical (structural mismatch; skipping array comparison).")
        return 1
    print("STRUCTURE / COORDINATES: identical (dims, coord values, dtypes all match)")

    benign, meaningful = compare_attrs(ref, test)
    if benign:
        print("  root attrs differ (expected/provenance): " + "; ".join(benign))
    if meaningful:
        print("  root attrs differ (UNEXPECTED): " + "; ".join(meaningful))

    ref_vars, test_vars = set(ref.data_vars), set(test.data_vars)
    if ref_vars != test_vars:
        print(f"\nVERDICT: NOT bit-identical — variable sets differ: "
              f"only-ref={ref_vars - test_vars} only-test={test_vars - ref_vars}")
        return 1

    height = int(ref.sizes["northing"])
    rows = _row_slabs(height, args.sample_rows)
    if args.sample_rows and args.sample_rows < height:
        print(f"\n(sampling {len(rows)} of {height} northing rows)")

    # Compare embeddings first, then obs-count layers, in a stable order.
    order = [v for v in ("embeddings", *OBS_VARS) if v in ref_vars]
    order += sorted(ref_vars - set(order))
    print("\nDATA VARIABLES:")
    results = [compare_variable(ref, test, v, rows) for v in order]
    for cmp in results:
        print(cmp.report())

    all_bit = all(c.bit_identical for c in results) and not meaningful
    print()
    if all_bit and args.sample_rows:
        print("VERDICT: bit-identical on the sampled rows (run without --sample-rows to prove the whole store).")
    elif all_bit:
        print("VERDICT: bit-identical — every coordinate and data array matches byte-for-byte.")
    else:
        print("VERDICT: NOT bit-identical (see per-variable divergence above).")
    return 0 if all_bit else 1


if __name__ == "__main__":
    sys.exit(main())
