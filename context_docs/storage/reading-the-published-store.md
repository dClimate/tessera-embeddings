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

That works, and it is not the fastest way. **Add one line to turn a 2.6-second open into a
0.15-second one** — §4.3 has the measurement and the four lines of code. Everything else about the
recipe above is right.

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

§4.1 is the read path in both regions, §4.2 the metadata and coverage questions, §4.3 the one
inherited setting that dominates the open, and §4.4 the comparison against what scoping predicted.
Three things about the method first, because each is a way to get a wrong number that looks right.

**Probes must land in pixels that hold embeddings.** Icechunk answers a read of an absent chunk from
the manifest without issuing a request, so a probe on elided ocean is nearly free — and a latency
sample that mixes those with real reads reports the mixture. The benchmark takes live shards from
the chunk index, samples candidate pixels inside them, and keeps only those whose `scales` value is
finite.

**Cold means a fresh process.** Icechunk caches manifests and the HTTP client pools connections, so
a second read in the same process measures the cache. Every cold phase runs in a subprocess that
then exits. The parent picks the probe pixels once and passes them down, so the cold arm, the warm
arm and both regions read the same addresses.

**The throughput column is decompressed elements per second**, which is what the scoping harness
reported (`elements / wall`). For the int8 `embeddings` one element is one byte, so it reads as a
logical MB/s. In general that can exceed a host's network bandwidth, because compression means
fewer bytes cross the wire than reach the array — but **on this store it does not**, because the
quantized embeddings turn out to be effectively incompressible (§4.4: 8.0 MB on the wire for an
8.39 MB inner chunk). So here the logical rate and the wire rate are within a few percent of each
other, and the distinction matters for how the figure is defined rather than for how it reads.

The one exception is `band_subset`, which counts only the 8 bands asked for while the reader must
fetch and decode all 128 in the chunk — so its figure understates the work by a factor of sixteen.
Kept as it is because changing the definition would break comparability with the scoping run, not
because it is a bandwidth claim.

### 4.1 The read path, in both regions

One `r7i.4xlarge` per region — the scoping host's instance type — reading zone 33N's 2025 slot
through 1,000 probe pixels, every figure a median across the 10/64/128 concurrency sweep. The
bucket is in us-west-2, so us-east-1 is the cross-region case and its S3 traffic necessarily
crosses the public internet: a gateway endpoint is region-local, so there is no configuration that
makes a remote region's reads local, and the asymmetry is part of what cross-region costs rather
than a flaw in the arrangement.

| | us-west-2 (in-region) | us-east-1 (cross-region) | |
|---|---|---|---|
| open, to first read | 2,638 ms cold / 2,571 warm | 3,198 / 3,183 | 1.2× |
| one pixel's 128-band vector, p50 / p95 | **302 / 463 ms** | **463 / 779 ms** | 1.5× |
| …bytes on the wire, per pixel | 8.04 MB | 8.15 MB | — |
| patch, 100×100×128 | 4 MB/s (0.4 s) | 1 MB/s (1.2 s) | 3.3× |
| tile, 1000×1000×128 | 280 MB/s (0.5 s) | 83 MB/s (1.5 s) | 3.4× |
| band subset, 512×512×8 | 9 MB/s (0.2 s) | 1 MB/s (1.4 s) | 6.7× |
| bulk, 4096×4096×128 | **763 MB/s (2.8 s)** | **194 MB/s (11.0 s)** | 3.9× |

Concurrency barely moves any of it: in-region bulk throughput is 741–781 MB/s across all three
settings and both cache states, and the open path is 2.5–2.7 s at every setting. The non-fill
fraction of every region read was 0.99, which is how these rows are known to have landed on data
rather than on ocean.

**Cross-region costs between 1.2× and 6.7×, and which end you land on depends on the shape of the
read, not its size.** Bulk pays 3.9× because it is bandwidth-bound and the bandwidth is what
distance takes away. The open path pays only 1.2×, because most of it is neither bandwidth nor
distance (§4.3). A single pixel pays 1.5×.

### 4.2 Metadata and coverage navigation

| | us-west-2 | us-east-1 |
|---|---|---|
| repository open | 0.21 s | 0.61 s |
| root group open | 2.58 s | 2.65 s |
| a zone group, after the root is open (median of 120) | 14 ms | 14 ms |
| shard enumeration, one zone, all nine years (median of 120) | 0.51 s | 0.89 s |
| shard enumeration, **all 120 zones** | **60 s** | 104 s |
| registry: list all 994 parts | 0.16 s | 0.62 s |
| registry: read all 994 schema footers | 73 s | 191 s |
| registry: whole dataset, 3,247,410 rows | 14.9 s | 51.5 s |
| registry: **"is my area covered", one box, one year** | **0.74 s** | 2.58 s |

Two of these are the answers somebody will actually want. **A complete shard-level coverage map of
the whole globe takes a minute in-region** and under two cross-region, which makes "what exists"
a question to ask rather than a table to maintain. And **the registry answers a coverage question
about an area of interest in under a second in-region**, which is the whole reason it exists.

Both regions independently found 3,229,545 live shards and 3,247,410 registry rows. That the two
hosts agree is the check that neither run was partial.

### 4.3 The one inherited setting that dominates the open path

The store saves the repository configuration the campaign wrote with, so a reader who passes none
of their own inherits the writer's manifest preload — a budget of 1,000,000 refs across up to 2,400
arrays. Measured on one host, 33N/2025, 150 probes, two alternated rounds, six samples per cell:

| us-west-2 | preload as saved | preload disabled |
|---|---|---|
| open, to first read (cold) | 2,706 ms | **158 ms** |
| one pixel, p50 | 286 ms | 278 ms |
| one pixel, p95 | 426 ms | 392 ms |
| one pixel, bytes on the wire | 8.02 MB | 8.02 MB |
| bulk | 755 MB/s | 754 MB/s |
| tile (cold) | 255 MB/s | 293 MB/s |

**It costs 2.5 seconds of every open and buys the reader nothing measurable.** Not point latency,
not bytes on the wire, not region throughput — every read column is the same or marginally better
without it. That is not a criticism of the setting: it was chosen so a *fill* did not re-fetch
manifests it was about to write into, and there it earns its keep. It is simply the wrong default
to inherit as a reader, and inheriting it is automatic.

Confirmed in both regions, three samples each:

| open, to first read | us-west-2 | us-east-1 |
|---|---|---|
| following the documented recipe | 2,490 ms | 3,592 ms |
| with preload disabled | **143 ms** | **932 ms** |
| opening the zone group by path instead of the root | 2,484 ms | 3,411 ms |

**Opening the zone group by path does not avoid it**, which is the natural guess and wrong: the cost
simply moves out of the root-group step and into the zone-group step. It is not the 120 groups being
enumerated; it is the preload, and the only thing that avoids it is asking for a different config.

Note also what the preload-free numbers say about distance: 143 ms in-region against 932 ms
cross-region is **6.5×**, against the 1.2× the full open path shows. The preload is largely
region-independent work, so leaving it in place hides most of the cross-region penalty behind
something slower than the penalty.

**For a reader, then:** fetch the saved config, turn preload off, and keep everything else.

```python
import icechunk

storage = icechunk.s3_storage(
    bucket="tessera-embeddings", prefix="v1.1/dclimate.icechunk",
    region="us-west-2", anonymous=True,
)
config = icechunk.Repository.fetch_config(storage)
config.manifest = icechunk.ManifestConfig(
    preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0),
    splitting=config.manifest.splitting,
)
repo = icechunk.Repository.open(storage, config=config)
```

**Start from the saved config, never from a fresh one.** A `RepositoryConfig` handed to
`Repository.open` replaces what the store saved rather than layering onto it, so building one to
change a single setting silently reverts every other. This document made that mistake once while
being written: an arm meant to differ only in its chunk cache also reverted the preload, and the
open time moved for the wrong reason.

### 4.4 Against what scoping predicted

[ADR 008](../decisions/008-global-store-architecture.md) settled the chunk and shard geometry on a
synthetic store: one timestep, one group, 70% land, random int8, on an `r7i.4xlarge` in us-west-2 —
the same instance type this benchmark used, which is the only thing about the two hosts that is
known to match.

**The store built is not the store scoped.** The published store has 120 groups rather than one and
nine years rather than one, so its snapshot and manifests are far larger; its data is real
quantized embeddings rather than random int8. The like-for-like part is the workload definitions,
their extents and the concurrency sweep, copied deliberately so that at least the questions match.

| scoping said | measured | verdict |
|---|---|---|
| zone open **128 ms** | 2,638 ms as documented, **143 ms** with preload off | **reconciled** — see below |
| bulk throughput, 13–2,014 MB/s across 24 samples, median **202** | 763 MB/s bulk, 280 tile, 9 band-subset, 4 patch | **consistent** — inside the range, comparable median |
| point p50 **29 ms**, p95 **204 ms** | p50 **302 ms**, p95 **463 ms** | **reconciled** — see below |
| **1.23 MB** on the wire per point read | **8.04 MB** | **reconciled** — see below |

**The open time reconciles, and the explanation is the preload.** Scoping's store was one group and
one year, so a 1,000,000-ref preload budget had almost nothing to fetch and cost almost nothing;
against a 120-group, nine-year repository the same budget is 2.5 s of work. With preload disabled
the published store opens to first read in 143 ms in-region, against the 128 ms scoping published.
The geometry is behaving as scoped; what changed is what the inherited configuration has to do.

**The throughput figures are consistent, and cannot be compared more sharply than that.** Scoping
retained only the distribution over its 24 `c256_sharded` throughput samples — four workloads ×
three concurrencies × two cache states — not the per-workload breakdown, and the raw run data is
gone. So "2,014 MB/s" is the maximum over that whole set and not a bulk-read figure; quoting it
against our bulk read would compare a maximum with a median. What can honestly be said is that our
four workloads span 4–763 MB/s with a median in the low hundreds, inside scoping's 13–2,014 range
around a comparable median.

**The point figures reconcile too, and the mechanism is the chunk cache against the working set.**

Start from what is solid. A point-vector read fetches **one whole 256×256×128 int8 inner chunk**,
which is 8.39 MB, and the quantized embeddings are effectively incompressible, so 8.39 MB is also
what crosses the wire. At in-region single-stream S3 rates that is a few hundred milliseconds. So
the measured 8.04 MB and 302 ms are one fact told twice, and any explanation has to fit both.

Three candidate explanations were eliminated first. **Compressibility** is not it: `synth.py`
generates random int8, deliberately worst-case, so scoping's data was as incompressible as ours.
**A codec difference between variants** is not it either: `variants.py` fixes only chunk and shard
geometry per variant and takes dtype, serializer and compressor from the library. And **partial
reads within an inner chunk** cannot be it, because a pixel cannot be returned without
decompressing the whole chunk that holds it — which is what makes ADR 008's reading of the gap,
that sharding does lean partial reads, not tenable as stated.

What remained was **cache reuse**, and it is measurable. If probes revisit inner chunks and the
cache holds them, the average bytes per read falls to `distinct chunks touched / probes` × 8.39 MB.
Measured in us-west-2 on 16S/2025, where 1,000 probes land in 1,088 inner chunks:

| | p50 | bytes on the wire per point |
|---|---|---|
| icechunk's default chunk cache | 209 ms | 7.97 MB |
| a 16 GiB chunk cache | **142 ms** | **5.14 MB** |

And the predicted figure: 1,000 uniform draws over 1,088 chunks touch
`1088 × (1 − e^(−1000/1088)) = 654` distinct ones, so a cache large enough to hold them all gives
`654 / 1000 × 8.39 = 5.49 MB` per point. **Measured 5.14 MB** — slightly better than the uniform
prediction, as it should be, since probes are drawn shard by shard and therefore cluster a little.
The arithmetic predicts the measurement.

Run the same arithmetic backwards on scoping's number: 1.23 MB/point implies a distinct-chunk
ratio of `1.23 / 8.39 = 0.147`, which for 1,000 probes means roughly **150 distinct inner chunks**.
That is what a small synthetic store gives when 1,000 probes are sampled clustered rather than
scattered — which is exactly what `t8_sharding.py` does for that measurement (`scattered=False`;
the scattered case is a different experiment). **So scoping's 1.23 MB/point is the average cost of
a read on a working set small enough to cache, and our 8.04 MB is the cost of a read that misses.
Both are right, and neither is the other's answer.**

**The practical consequence is the useful part.** A consumer whose reads revisit the same
neighbourhood should size the chunk cache to the working set, because icechunk's default is small
against an 8.39 MB chunk and a miss costs a whole chunk. And a consumer reading scattered pixels
should stop and read regions instead (§6) — no cache helps a working set that never repeats.

**Two things this leaves open.** ADR 008's own two variant figures — 8.69 MB/point unsharded
against 1.23 sharded, on identical geometry and codec — are still not explained by anything here,
since cache reuse should have applied equally to both; the raw run data is gone, so this is
recorded rather than chased. And the 16 GiB arm was measured within a single pass, not across a
genuinely warm second pass, so it shows reuse *within* a workload and not what a second traversal
would cost.

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
a whole 256×256×128 inner chunk on the wire — measured at 8.0 MB and 302 ms in-region, and it does
not get cheaper when probes revisit the same chunk (§4.4). That is the geometry working as intended:
it is what makes block reads fast and object counts manageable. But a consumer whose access pattern
is scattered single pixels will find it slow, and the fix is to read a region and index into it, not
to tune the client. Reading a 1000×1000 tile delivers 280 MB/s; reading its million pixels one at a
time would take three days.

**A reader inherits the writer's manifest preload, which costs 2.5 s of every open and returns
nothing** (§4.3). Nothing warns about it, opening a zone group directly does not avoid it, and the
only fix is to pass a configuration — starting from the saved one, or every other setting silently
reverts too.

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
