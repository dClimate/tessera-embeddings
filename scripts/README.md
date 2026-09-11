# scripts/

Code that runs against production on an ad-hoc basis, so it has no home inside the
orchestrated stack. Nothing here is imported by `src/tessera_embeddings/`, and the
architecture rules do not scan it ([ADR 014](../context_docs/decisions/014-architecture-rules-scan-the-package.md)).

## Two tiers, and why the distinction is written down

Every script below is either **supported tooling** — a failure is a bug to repair — or a
**kept-for-reference instrument**, written to settle one question, kept because its findings
are published and anyone re-opening those findings wants the thing that produced them, and
explicitly not maintained. A failure there raises whether to keep the script, not how to fix it.

[ADR 016](../context_docs/decisions/016-scripts-layout-and-shared-modules.md) makes this index
a safety mechanism rather than a courtesy. On 2026-08-17 three scripts were shortlisted for
deletion because nothing referenced them; reading them showed all three were still useful.
**Reference counting finds undocumented code, not dead code** — so an unlisted script is
indistinguishable from an abandoned one.

## `build/` — supported tooling

Produces artifacts the campaign depends on.

| script | what it does |
|---|---|
| `build_landmask_coverage.py` | Builds, verifies and validates the campaign land-mask coverage store from the partner TIFF delivery ([ADR 010](../context_docs/decisions/010-landmask-registry-coverage.md)). The mask is the campaign's work list. |
| `record_stac_cassettes.py` | Re-records the VCR cassettes the integration and parity tests replay. Hits the STAC endpoints only, never COG bodies. Needs Earthdata credentials. |

## `publish/` — supported tooling that MUTATES a published artifact

One script, and the directory exists to keep it away from the read-only tools. Anything here
changes something consumers are already using, so it dry-runs by default and refuses to write a
store whose state is not the one its safety evidence was gathered against.

| script | what it does |
|---|---|
| `set_published_store_reader_config.py` | Switches the published store's SAVED manifest preload off, so consumers stop inheriting a setting sized for writing that costs them ~2.5 s of every open and returns nothing. On spec version 2 that rewrites the one object holding every tag and the branch pointers, so run `diagnostic/repo_config_write_safety.py` first and read its evidence. Dry run unless `--apply`; `--rollback` undoes it. |

## `diagnostic/` — supported tooling

Answers "why is this environment behaving that way?" and is expected to work on demand.

| script | what it does |
|---|---|
| `check_env.py` | Prints the installed torch variant and CUDA availability — which lock file this environment was actually built from. |
| `probe_edl_bearer.py` | Whether a Bearer Earthdata token survives the full ASF redirect chain. Written for a specific auth failure and kept because that failure recurs. |
| `published_store_census.py` | Whether the published global store opens, conforms to the declared layout in all 120 zone groups, and agrees with its own completion tags about which cells are done. Exits non-zero on any disagreement. |
| `published_store_read_bench.py` | Read performance of the published store — open latency, point-vector percentiles, region throughput, bytes on the wire — using the same workloads and concurrency sweep as `scoping/scale_tests/t1_read_bench.py`, so the built store can be compared against the scoped one. |
| `published_registry_census.py` | Whether the published Parquet registry is shaped as designed, whether a whole-dataset read loses columns, how fast an area-of-interest coverage query answers, and whether the registry's coverage agrees with the store's. |
| `repo_config_write_safety.py` | Whether rewriting a store's saved configuration can lose anything. On spec version 2 that write rebuilds the one object holding every tag, the branch pointers and every snapshot record, so this proves — against a throwaway store, and optionally against a clone of a real store's reference state — that the tags, the branch tip, the data and the other settings survive it. Run it before changing configuration on anything published. |

## `scoping/` — kept-for-reference instruments

Each answered one sizing question whose answer is published in `context_docs/`. Kept so a
figure can be re-derived rather than re-argued. **Not maintained**: re-running one against a
changed upstream may need repair first, and that is a decision, not a defect.

| script | the question it answered | where the answer lives |
|---|---|---|
| `census_s1_coverage.py` | Global OPERA radar coverage on an equal-area land grid | `context_docs/campaign/campaign-cost-model.md` §6 |
| `census_s2_coverage.py` | Global Sentinel-2 usable-observation counts on the same grid | `context_docs/inference/minimum-optical-depth.md` |
| `cluster_work_spread.py` | How evenly N Ray clusters divide the work, from the real mask — the campaign ends when the LAST cluster does | `context_docs/campaign/campaign-plan.md` §10, cost model §5b |
| `scale_tests/` | Whether Icechunk holds up at campaign scale: read, write, prealloc, group count, contention, GC, ramp, sharding | [ADR 008](../context_docs/decisions/008-global-store-architecture.md), `context_docs/storage/icechunk-api-ledger.md`, and its own `README.md` |

`cluster_work_spread.py` is the exception worth knowing: it is the **only** thing that
re-derives the cluster-balance figure the campaign plan depends on, so it is closer to
supported than the rest of this tier.
