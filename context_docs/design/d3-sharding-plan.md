# D3 settlement plan — shard or not shard the global embeddings store

> **RESOLVED (2026-07-14): ADOPT `c256_sharded`, and shard `scales` too.** Runs
> `d3` + `d3v2` (t8) passed all gates: reads leaner *and* faster (wire 1.23 vs
> 8.69 MB/pt, reproducible; scattered p95 131 vs 217 ms), the shard-aligned +
> land-masked writer builds at **0.46×** the unsharded time with lean ~3.5 GB
> shards, and the embeddings array consolidates 403→9 objects (post-GC). E4 also
> showed the object-count win is capped unless `scales` is sharded too, so it is.
> Production writer must be shard-aligned + land-masked. See ADR-008 D3 for the
> full write-up; the plan below is retained as the method of record.

Companion to ADR-008 decision **D3**. Run 1 flipped sharding from "optional,
probably not" to "strong contender" — this plan defines the experiments that
turn that into a decision, the criteria for calling it, and the tooling
(`scripts/scale_tests/t8_sharding.py`) that runs them.

## Outcome

**Adopted `c256_sharded`, and `scales` is sharded too** (2026-07-14). Runs `d3`/`d3v2` passed every
gate: reads leaner *and* faster — 1.23 MB per point on the wire against 8.69, scattered p95 131 ms
against 217 — and the shard-aligned, land-masked writer builds at **0.46×** the unsharded time.

The decision is ADR-008 D3, which is where it belongs and where a reader will look; the scale test
that produced it is `scripts/scale_tests/t8_sharding.py`, and the geometry it settled is in
`config/store_layout.py`. The spec that stood here described what to build, and it is built.

**The one thing worth carrying out of it:** sharding `scales` was not obvious and was nearly skipped.
It is a small array beside the embeddings, so the instinct is that its layout does not matter — but
it is read on every dequantisation, so an unsharded `scales` puts a small-object read in front of
every single chunk read. Layout follows access pattern, not array size.
