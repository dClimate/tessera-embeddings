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

    te-compare-stores \
        s3://<bucket>/embeddings/<roi>-reference_500m.zarr \
        s3://<bucket>/embeddings/<roi>-candidate_500m.zarr

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

# The coarsened-store format contract: these variables must be present in BOTH
# stores — two identically-malformed stores (e.g. both missing `embeddings`)
# must fail, not vacuously pass. `scales` must be ABSENT: its presence means a
# full-resolution int8 store (compare those with compare_outputs.py), not a
# pre-dequantized coarsened one.
REQUIRED_VARS = ("embeddings", *OBS_VARS)
FORBIDDEN_VARS = ("scales",)

# The format's expected dtypes. Pinning them (not just checking the two stores
# agree) means two identically-malformed stores — e.g. both with int16
# embeddings, or uint16 obs counts that would overflow at 50x50 coarsening
# (up to ~315k obs/px ≫ 65535) — fail rather than certify each other.
EXPECTED_DTYPES = {
    "embeddings": "float32",
    "s2_obs_count": "uint32",
    "s1_asc_obs_count": "uint32",
    "s1_desc_obs_count": "uint32",
}

# ...and the format's expected dimension layout, again pinned in both stores:
# a wrong-but-agreeing axis order (e.g. embeddings as (time, northing, band,
# easting)) would pass a dtype-and-agreement check yet break the northing-slab
# reader. All required vars carry (northing, easting), optionally led by time;
# embeddings additionally trails band.
EXPECTED_DIMS = {
    "embeddings": ("northing", "easting", "band"),
    "s2_obs_count": ("northing", "easting"),
    "s1_asc_obs_count": ("northing", "easting"),
    "s1_desc_obs_count": ("northing", "easting"),
}

# Root attrs that legitimately differ between two runs of the same ROI — pure
# run provenance, not data or format. `run_id` plus anything run-scoped
# (``run_*``, e.g. run_started_at/run_completed_at timestamps) is reported but
# never fails the gate; every other attr difference (format, version, grid)
# still does.
BENIGN_ATTR_DIFFS = frozenset({"run_id"})


def _is_benign_attr(key: str) -> bool:
    """Whether an attr key is run-scoped provenance (reported, never a failure)."""
    return key in BENIGN_ATTR_DIFFS or key.startswith("run_")


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
        """Every compared element matched byte-for-byte (and none were compared vacuously)."""
        return self.n > 0 and self.n_bit_equal == self.n

    @property
    def mean_abs_diff(self) -> float:
        """Mean |Δ| over finite pairs (0.0 when nothing was comparable)."""
        return self.sum_abs_diff / self.n_finite_pairs if self.n_finite_pairs else 0.0

    @property
    def cosine_mean(self) -> float:
        """Mean per-pixel cosine (1.0 when no pixel had a computable cosine)."""
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
    # str() because xarray types its keys as Hashable, which is neither sortable
    # nor usable as a name in the messages below; every key in these stores is a
    # string.
    ref_coords, test_coords = {str(c) for c in ref.coords}, {str(c) for c in test.coords}
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
        (benign if _is_benign_attr(key) else meaningful).append(msg)
    return benign, meaningful


def compare_var_structure(ref: xr.Dataset, test: xr.Dataset) -> list[str]:
    """Per-variable dims/shape/dtype mismatches between the two stores.

    Must run before any bit comparison: ``_bit_equal`` views each side through
    an unsigned integer of *its own* width, so a dtype change that preserves
    small values (uint16 vs uint32 counts, or all-zero float32 vs float64)
    would otherwise compare "equal" and falsely certify non-identical stores;
    a shape/dims change would crash mid-read instead of reporting cleanly.
    """
    problems: list[str] = []
    # str(): see compare_coords — xarray keys are typed Hashable, ours are strings.
    for v in sorted({str(v) for v in ref.data_vars} & {str(v) for v in test.data_vars}):
        a, b = ref[v], test[v]
        if a.dims != b.dims:
            problems.append(f"var '{v}' dims {a.dims} vs {b.dims}")
        elif a.shape != b.shape:
            problems.append(f"var '{v}' shape {a.shape} vs {b.shape}")
        elif a.dtype != b.dtype:
            problems.append(f"var '{v}' dtype {a.dtype} vs {b.dtype}")
    return problems


def _accumulate_float(cmp: VarComparison, a: np.ndarray, b: np.ndarray) -> None:
    """Fold one float slab into the running numeric-divergence metrics."""
    # "not comparable" = either side is non-finite (NaN OR ±inf). The mask must
    # exclude inf too: inf - inf = NaN would poison max/mean/CDF on a store that
    # carries infinities. The NaN mask-agreement counter still keys on isnan
    # (NaN is the invalid-pixel fill); an inf is neither valid data nor the fill,
    # so a one-sided inf shows up as a non-finite (dropped) element and, if it
    # differs bitwise, in n_bit_equal — it can't masquerade as agreement.
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    cmp.nan_mask_mismatches += int((nan_a != nan_b).sum())
    both = fin_a & fin_b
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
    ref: xr.Dataset, test: xr.Dataset, name: str, rows: list[tuple[slice, list[int] | None]]
) -> VarComparison:
    """Stream-compare one data variable across the given northing row slabs.

    Each entry is ``(slab, selection)``: the slab is read whole (one backing
    zarr chunk's worth of rows), then ``selection`` — when sampling — picks
    the sampled rows out of it in memory. ``None`` compares the whole slab.
    """
    da_ref, da_test = ref[name], test[name]
    cmp = VarComparison(name=name, dtype=str(da_ref.dtype))
    is_float = da_ref.dtype.kind == "f"

    # A variable without a `northing` dim (a scalar or non-spatial auxiliary
    # var an unexpected store might carry) can't be row-slabbed — read it whole
    # in one shot rather than crashing on isel(northing=...). The known format
    # vars (embeddings, obs counts) all have northing and take the streaming
    # path below.
    if "northing" not in da_ref.dims:
        a, b = da_ref.values, da_test.values
        cmp.n += a.size
        cmp.n_bit_equal += _bit_equal(a, b)
        (_accumulate_float if is_float else _accumulate_int)(cmp, a, b)
        return cmp

    for rows_slice, sel in rows:
        sub_ref = da_ref.isel(northing=rows_slice)
        sub_test = da_test.isel(northing=rows_slice)
        if sel is not None:
            # Select the sampled rows by DIM NAME, not a positional numpy index:
            # northing's axis position depends on whether a `time` dim is present
            # ((time, northing, easting, band) vs (northing, easting, band)), so
            # positional indexing would silently select easting on a time-less
            # store. `sel` are slab-relative northing offsets.
            sub_ref = sub_ref.isel(northing=sel)
            sub_test = sub_test.isel(northing=sel)
        a = sub_ref.values
        b = sub_test.values
        cmp.n += a.size
        cmp.n_bit_equal += _bit_equal(a, b)
        if is_float:
            _accumulate_float(cmp, a, b)
        else:
            _accumulate_int(cmp, a, b)
    return cmp


def _row_slabs(height: int, sample_rows: int | None) -> list[tuple[slice, list[int] | None]]:
    """Row slabs covering [0, height), or a chunk-aligned sampled subset.

    Sampling groups the sampled row indices by their containing
    ``ROW_BLOCK``-aligned slab (= the stores' northing chunk), so each backing
    zarr chunk downloads and decompresses ONCE; the sampled rows are then
    selected from the slab in memory. Per-row slices would re-read the whole
    containing chunk for every sampled row — far slower than a full scan once
    several samples land in one chunk.
    """
    if sample_rows and sample_rows < height:
        idx = np.linspace(0, height - 1, sample_rows, dtype=int)
        by_block: dict[int, set[int]] = {}
        for i in idx:
            by_block.setdefault(int(i) // ROW_BLOCK, set()).add(int(i) % ROW_BLOCK)
        return [
            (slice(blk * ROW_BLOCK, min((blk + 1) * ROW_BLOCK, height)), sorted(rows))
            for blk, rows in sorted(by_block.items())
        ]
    return [(slice(r0, min(r0 + ROW_BLOCK, height)), None) for r0 in range(0, height, ROW_BLOCK)]


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

    # str(): see compare_coords — xarray keys are typed Hashable, ours are strings.
    ref_vars, test_vars = {str(v) for v in ref.data_vars}, {str(v) for v in test.data_vars}
    if ref_vars != test_vars:
        print(
            f"\nVERDICT: NOT bit-identical — variable sets differ: "
            f"only-ref={ref_vars - test_vars} only-test={test_vars - ref_vars}"
        )
        return 1

    # Format contract: both stores must carry the coarsened layout. Without
    # this, two identically-malformed stores (both missing `embeddings`, say)
    # would vacuously "pass" on whatever variables remain in common.
    missing = [v for v in REQUIRED_VARS if v not in ref_vars]
    if missing:
        print(f"\nVERDICT: INVALID coarsened store(s) — required variable(s) absent from both: {missing}")
        return 1
    forbidden = [v for v in FORBIDDEN_VARS if v in ref_vars]
    if forbidden:
        print(
            f"\nVERDICT: NOT a coarsened store — {forbidden} present (full-resolution int8 "
            "format; compare those runs with compare_outputs.py instead)."
        )
        return 1

    # Pin the required variables to the format's expected dtypes in BOTH stores
    # (compare_var_structure only checks the two AGREE — two identically-wrong
    # stores would agree and pass).
    dtype_problems = [
        f"{tag} '{v}' is {ds[v].dtype} (expected {EXPECTED_DTYPES[v]})"
        for v in REQUIRED_VARS
        for tag, ds in (("ref", ref), ("test", test))
        if str(ds[v].dtype) != EXPECTED_DTYPES[v]
    ]
    if dtype_problems:
        print("\nVERDICT: INVALID coarsened store — required variable(s) have unexpected dtype:")
        for p in dtype_problems:
            print(f"  - {p}")
        return 1

    # ...and the dims layout (again in both stores): a wrong-but-agreeing axis
    # order passes the dtype + agreement checks yet breaks the slab reader / the
    # format's meaning. A leading `time` axis is allowed and ignored.
    dim_problems = [
        f"{tag} '{v}' dims {tuple(ds[v].dims)} (expected {EXPECTED_DIMS[v]}, optionally led by time)"
        for v in REQUIRED_VARS
        for tag, ds in (("ref", ref), ("test", test))
        if tuple(d for d in ds[v].dims if d != "time") != EXPECTED_DIMS[v]
    ]
    if dim_problems:
        print("\nVERDICT: INVALID coarsened store — required variable(s) have unexpected dims:")
        for p in dim_problems:
            print(f"  - {p}")
        return 1

    var_problems = compare_var_structure(ref, test)
    if var_problems:
        print("\nDATA-VARIABLE STRUCTURE — NOT identical:")
        for p in var_problems:
            print(f"  - {p}")
        print("\nVERDICT: NOT bit-identical (variable structure mismatch; skipping value comparison).")
        return 1

    height = int(ref.sizes["northing"])
    rows = _row_slabs(height, args.sample_rows)
    if args.sample_rows and args.sample_rows < height:
        sampled = sum(len(sel) for _, sel in rows if sel is not None)
        print(f"\n(sampling {sampled} of {height} northing rows, across {len(rows)} chunk-aligned slabs)")

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
