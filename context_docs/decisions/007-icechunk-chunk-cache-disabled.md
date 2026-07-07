# 007 — Icechunk chunk cache left at default (disabled)

**Status:** Accepted

## Context

During striped inference development a 512 MB icechunk chunk cache was added to
absorb the per-strip re-reads of S2 band data.  A real-store A/B test showed it
did not help, so it was removed.

## Decision

Leave icechunk's chunk cache at its (small) default — do not configure a custom
size in `_default_repo_config`.

## Why the cache did not help

1. **Decompression is not cached.** Icechunk caches *compressed* chunk bytes and
   decompresses above the cache layer.  A cache hit saves the S3 GET but not the
   zstd decode, so it cannot relieve a decompression-bound workload.

2. **Working set exceeds cache by 8–16×.** A dense strip reads all 10 S2 bands ×
   T_kept timesteps band-major.  At 512 MB the cache is ~8–16× smaller than the
   working set, so cross-strip reuse thrashes to a ~0% hit rate.  Measured: a
   cache 8× smaller than the working set was no faster than no cache and slightly
   slower from bookkeeping overhead.

3. **Density-based striping eliminates the re-read pattern.** `actors._strip_height_for_density`
   now loads the sparse majority of chunks in a single full-height strip, so
   there are no per-strip re-reads for a cache to serve in the first place.

4. **Memory cost.** 512 MB per store × three stores opened per chunk ≈ 1.5 GB
   resident, directly competing with the per-strip band budget on a 16 GB
   worker.

## Rejected alternatives

**512 MB cache (previous default):** No measurable throughput benefit; ~1.5 GB
memory cost on the critical worker tier.

**Cache sized to a whole strip (~9 GB):** Would make hits land but only fits 32 GB
boxes, not the standard 16 GB worker fleet.

**Band-interleaved reads:** The principled fix — would reduce reuse distance from
10 × T_kept chunks to 10 chunks, making even a modest cache effective.  Not
implemented yet; tracked as a future optimization.

## Consequences

- Workers keep ~1.5 GB of headroom that the cache would have consumed.
- If band-interleaved reads are implemented, this decision should be revisited —
  that access pattern would make a strip-sized cache worthwhile.

## Related

- `storage/zarr_store.py` — `_default_repo_config`
- [Inference spatial striping](../memory/inference-spatial-striping.md)
- [Striping perf regression](../memory/striping-perf-regression.md)
