# Reading the published store: what a consumer needs, what it costs, and what would surprise them

**The finished global store needs no credentials.** The delivery bucket's policy grants
`s3:Get*` and `s3:List*` to every principal over TLS, so anyone can open
`s3://tessera-embeddings/v1.1/dclimate.icechunk`, enumerate all 120 UTM zone groups, and read
embeddings, with no AWS account at all. That was verified end to end with every AWS environment
variable cleared and the shared-credentials file pointed at an empty directory, so it is a
statement about the public path and not about a profile that happened to be lying around.

This document is the reader's counterpart to
[`writing-to-the-global-store.md`](writing-to-the-global-store.md). That one is about how the bytes
got there. This one answers the four questions somebody who did not run the campaign will ask:
can I open it, is it what was promised, how fast is it, and how do I find out whether my area is
covered.

| § | question |
|---|---|
| 1 | the recipe, and the one thing that will not work |
| 2 | what is published, and what "complete" means |
| 3 | three ways to ask what is covered, and what each costs |
| 4 | measured read performance, in two regions, against what scoping predicted |
| 5 | the registry beside the store |
| 6 | what is missing, and what would surprise a consumer |
| 7 | re-running every measurement here |

Everything below was measured on 2026-09-10, the day the campaign's last cell
(17S/2017) landed, against snapshot `QR7F41A6WYZ03VC92T6G`.

---

## 1. The recipe

```python
from tessera_embeddings.storage.global_store import open_global_repo
import zarr

repo = open_global_repo("s3://tessera-embeddings/v1.1/dclimate.icechunk",
                        region="us-west-2", anonymous=True)
session = repo.readonly_session(branch="main")
zone = zarr.open_group(session.store, mode="r")["33N"]

print(zone.attrs["years_complete"])          # which years this zone holds
vector = zone["embeddings"][8, 500_000, 30_000, :]   # one pixel, all 128 bands, 2025
```

Two details carry the rest of the document.

**The store is Icechunk, not plain Zarr.** `xr.open_zarr("s3://.../dclimate.icechunk")` does not
work and does not fail in an obvious way — the URI is a repository, not a Zarr hierarchy. A reader
has to open the repository, take a session, and hand *that* session's store to Zarr or xarray. The
`anonymous=True` above is the whole of the credential story; without it the default AWS chain is
used, which also works for anyone whose account has been granted access, and fails confusingly for
somebody who has AWS configured for something else entirely.

**The time axis is nine preallocated slots, and an unfilled one reads back as fill.** All nine
2017–2025 timesteps exist in every zone group from the moment it was seeded, because unwritten
chunks cost nothing ([ADR 008](../decisions/008-global-store-architecture.md) D1). So a read of a
year that was never filled succeeds and quietly returns the fill value. **`years_complete` is the
authority**, and it distinguishes nothing else can: a year deliberately left empty — a zone with no
qualifying land that year — *is* in the list, while a year that never landed is absent from it.

Rounding out the recipe: reading `zone.attrs` pulls the group's whole metadata document, which
includes a `runs` entry per year carrying that fill's run id, its input coverage per source store,
and the commit it was built from. That is useful provenance and it is not small; a reader who only
wants `years_complete` still pays for all of it, once per zone.

## 2. What is published

One Icechunk repository, one `main` branch, 120 Zarr groups named `01N` … `60S`, each its own UTM
CRS. Per group:

| array | shape | dtype | inner chunk | shard |
|---|---|---|---|---|
| `embeddings` | (9, northing, easting, 128) | int8 | (1, 256, 256, 128) | (1, 2048, 2048, 128) |
| `scales` | (9, northing, easting) | float32 | (1, 256, 256) | (1, 2048, 2048) |
| `s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count` | (9, northing, easting) | uint16 | (1, 256, 256) | (1, 2048, 2048) |
| `s2_month_covered`, `s1_asc_month_covered`, `s1_desc_month_covered` | (9, northing, easting, 12) | int8 | (1, 256, 256, 12) | (1, 2048, 2048, 12) |

plus the coordinate arrays `time`, `time_bnds`, `northing`, `easting`, `band` and `month`.

**All 120 groups conform to the declared layout** — dtype, inner chunks and shards, per array,
compared against `config.store_layout.GLOBAL` with the expectations clamped to each array's own
shape exactly as the writer clamped them. Zero departures. The `scales` array is sharded on the
same grid as `embeddings`, which was the second half of the sharding decision and the easier half
to get wrong.

**Coverage, as delivered:**

| | |
|---|---|
| zone-years possible | 1,080 (120 zones × 9 years) |
| zone-years marked complete | **1,066** |
| zone-years never filled | **14** |
| shards holding embeddings, whole store | **3,229,545** |
| completion tags (`zone-<ZONE>-<YEAR>`) | 1,066 |
| cells marked complete but untagged, or tagged but unmarked | 0 |

The attribute and the tag are written in separate commits, so they *can* disagree; they do not. The
14 unfilled cells are 03S 2017/2018/2020/2021, 08S 2017/2018, 09S 2017, 24N 2017, 26S 2017,
29S 2017, and 31S 2018/2019/2020/2021 — every one of them a year where the optical archive over
that zone's land could not meet the depth rule. Thirteen were confirmed unfillable across three
catalogues; 24N/2017 is the exception and is a gap in one *mirror* rather than in the archive (see
`corrections-register.md`).

**Only four `year-<YEAR>-complete` tags exist** — 2022 through 2025. Every cell in 2017–2021 is
complete and tagged individually, but those five years never got their roll-up tag, so a consumer
using year tags to find finished years would see four of nine. The per-zone attribute is right;
the roll-up is incomplete.

## 3. Three ways to ask what is covered

**Per zone-year, from the attribute.** `zone.attrs["years_complete"]` — one metadata read, already
paid for by opening the group. This is the only one of the three that distinguishes a deliberately
empty year from one that never landed.

**Per shard, from the store's own chunk index.** `Session.chunk_coordinates("/<zone>/scales")`
returns the coordinates of every initialized chunk, and on a sharded array **those coordinates are
shard-grid, not inner-chunk-grid** — which is what makes this cheap rather than a
billion-entry enumeration. It reads manifests only; no chunk bytes move. In-region this takes
**0.4 s for 33N's 8,593 shards in one year** and about a second for all nine years of a large zone,
so the whole globe is a couple of minutes.

Ask `scales`, not `embeddings`: both carry the same shard set, but `scales` is float32 with a NaN
fill, so "no chunk here" and "a chunk of zeros here" can never be confused. The obs-count arrays
answer a *different* question, and the difference is informative — a shard can hold observation
counts and no embeddings at all, which is what a tile that was imaged and then wholly refused looks
like. In 33N/2017, `s2_obs_count` has 8,471 shards where `scales` has 7,982.

This is wrapped as `storage.published_store.live_shards`.

**Per tile, from the registry.** The Parquet dataset beside the store carries one row per 2048-pixel
tile per year with a WGS84 bounding box, so "is my area of interest covered, and how well" is a
filter rather than a grid calculation. §5.

**A fourth way that does not work, and why it is tempting.** `Array.nchunks_initialized` on a
sharded array reports *shards × chunks-per-shard* — 9,792 for 16S's `embeddings`, which is
153 shards times the 64 inner chunks a full shard would hold. Since a live shard's ocean inner
chunks are elided, that figure counts positions nobody wrote, and a consumer using it to estimate
data volume will overstate it by whatever fraction of each shard is ocean.

## 4. Measured read performance

See §4.1 for the table. Three things about the method first, because each of them is a way to get a
wrong number that looks right.

**Probes must land in pixels that hold embeddings.** Icechunk answers a read of an absent chunk from
the manifest without issuing a request, so a probe on elided ocean is nearly free — and a latency
sample that mixes those with real reads reports the mixture. The benchmark takes live shards from
the chunk index, samples candidate pixels inside them, and keeps only those whose `scales` value is
finite.

**Cold means a fresh process.** Icechunk caches manifests and the HTTP client pools connections, so
a second read in the same process measures the cache. Every cold phase runs in a subprocess that
then exits. The parent picks the probe pixels once and passes them down, so the cold arm, the warm
arm and both regions read the same addresses.

**The throughput column is not a wire rate.** It is decompressed elements per second, which is what
the scoping harness reported (`elements / wall`); for the int8 `embeddings` one element is one byte,
so it is a logical MB/s and can exceed the host's network bandwidth, because zstd means fewer bytes
cross the wire than reach the array. The `band_subset` workload counts only the bands asked for
while the reader must fetch and decode all 128 in the chunk, so its figure understates the work by
design — kept because changing it would break comparability with the scoping run, not because it is
a bandwidth claim.

### 4.1 The numbers

*(inserted below)*

### 4.2 Against what scoping predicted

[ADR 008](../decisions/008-global-store-architecture.md) settled the chunk and shard geometry on a
synthetic store: one timestep, one group, 70% land, random int8, on an `r7i.4xlarge` in us-west-2.
The figures it published for the adopted `c256_sharded` variant, as medians across a 10/64/128
concurrency sweep, were a cold point-vector p50 of **29 ms** and p95 of **204 ms**, a zone open of
**128 ms**, bulk throughput peaking at **2,014 MB/s**, and — from the `d3`/`d3v2` runs — **1.23 MB
on the wire per point read**.

**The store built is not the store scoped**, in three ways that all push the same direction, and a
comparison that ignores them is not a comparison. The published store has 120 groups rather than
one and nine years rather than one, so its snapshot and manifests are far larger; its data is real
quantized embeddings rather than random bytes, so it compresses differently; and the benchmark
reads it from a machine matched to the scoping host in instance type but not in anything else.
The like-for-like part is the workload definitions, their extents and the concurrency sweep, which
were copied deliberately.

## 5. The registry beside the store

`s3://tessera-embeddings/v1.1/dclimate.registry` — 994 Parquet parts, 142 MB, at
`parts/zone=<ZONE>/year=<YEAR>/<run_id>.parquet`. **3,247,410 rows**, one per tile per year, each
with a WGS84 bounding box, whether the tile was embedded, how many of its pixels the depth rule
refused and for which of three reasons, and how deep the imagery was where it fell short.

**It is written as designed, on every check.** All 994 parts carry all 23 declared columns at the
declared types; every part's own key-value metadata agrees with the zone and year in its path; no
part's path fails to parse; and no cell has more than one part, which means nothing was filled
twice.

**It agrees with the store.** Per cell, the count of rows with `embedded` true equals the count of
shards holding embeddings — checked over 16S (17 tiles in each of nine years) and 33N (8,593 tiles
in each of nine years, of which 7,982 are embedded in 2017 and all 8,593 in every later year). The
2017 shortfall is 611 tiles that were evaluated and produced nothing, and the store and the
registry put the same number on it from opposite directions. That is the strongest evidence
available that the registry is a faithful convenience layer rather than a second, drifting source
of truth.

**Row counts differ between the two, and both are right.** 3,247,410 registry rows against
3,229,545 shards holding embeddings: the difference, 17,865, is tiles that were evaluated and
refused outright. A consumer who reads `embedded` alone as coverage will overstate it, which is why
the registry carries `refused_px` beside it — a tile can be embedded and still be largely holes.

**Reading the whole dataset needs the schema stated**, or a column added mid-campaign is silently
dropped: `pyarrow` infers a dataset's schema from the first file it finds in sorted path order, so
whether a newer column is visible depends on whether zone 01N was written before or after it was
added.

```python
import pyarrow.dataset as ds
from tessera_embeddings.storage.registry import dataset_schema
dataset = ds.dataset(f"{root}/parts", schema=dataset_schema(), partitioning="hive")
```

**On this dataset the hazard is latent, not live.** Reading it both ways returns the same 25 columns
and the same 3,247,410 rows, because the whole campaign ran on one commit
(`bbb9d836b3667f579c18a052e8d456d9a58c4e11`) and every part therefore has the same schema. It will
bite the first time a column is added and only some zones are rewritten.

## 6. What is missing, and what would surprise a consumer

**There is no compacted master and no `_common_metadata`.** Nothing sits beside `parts/`. The
registry's own design says a consumer without this package should read the compacted master, and
that a dataset-level `_common_metadata` is where the compaction would put a schema a reader could
take from the dataset rather than from a part it has to know is current. Neither exists, so today
an outside consumer either states the schema from a part they have chosen themselves or accepts
whatever `pyarrow` infers. Fine while every part agrees; a trap the first time they do not.

**Nothing in the store or the registry records the run's parameters.** The store's root carries
`optical_min_obs = 15`, which is the depth rule. It does not carry `allow_s2_only` or
`min_valid_coverage`, so exactly reproducing a cell needs the run's parameters from elsewhere —
they are in `context_docs/campaign/campaign-plan.md`, not in the published product.

**Point access is expensive, and the store is not built for it.** One pixel's 128-band vector costs
a whole 256×256×128 inner chunk on the wire. That is the geometry working as intended — it is what
makes block reads fast and object counts manageable — but a consumer whose access pattern is
scattered single pixels will find it slow, and the fix is to read a region and index into it, not
to tune the client.

**Five of the nine years have no roll-up completion tag** (§2).

## 7. Re-running all of this

Three scripts under `scripts/diagnostic/`, all of which take `--anonymous` and none of which need
credentials:

```bash
uv run python scripts/diagnostic/published_store_census.py --anonymous --shards --json census.json
uv run python scripts/diagnostic/published_store_read_bench.py --anonymous --zone 33N --year 2025 \
    --points 1000 --json bench.json
uv run python scripts/diagnostic/published_registry_census.py --anonymous \
    --verify-zones 16S,33N --aoi=-93.8,41.9,-93.4,42.2 --json registry.json
```

The census and the registry audit exit non-zero when anything disagrees, so either can be run as a
check rather than read as a report. The accessibility claims are also pinned as tests:
`tests/unit/storage/test_published_store.py` for the reader logic against a synthetic store, and
`tests/integration/test_published_store_access.py` against the live store — opt-in twice, via the
`integration` marker and `TESSERA_TEST_PUBLISHED_STORE=1`, for the reasons in
`tests/integration/README.md`.

The cross-region figures in §4 came from one `r7i.4xlarge` in each region, matching the scoping
host's instance type, each running the benchmark from user-data and terminating itself afterwards.
