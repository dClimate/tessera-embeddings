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

That works, and it is not the fastest way. **Add one line to turn a 2.7-second open into a
fifth of a second** — §4.3 has the measurement and the four lines of code. Everything else about the
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
billion-entry enumeration. It reads manifests only; no chunk bytes move. In-region the median zone
takes **0.52 s for all nine of its years**, the heaviest (35N, 82,107 shards) 0.83 s, and **the
whole globe 60 s** — so the complete shard-level coverage map of the published store is a question
to ask rather than a table to keep.

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

**Cold means a fresh process; warm means a second pass through the same open.** Icechunk caches
manifests and pools connections, so every cold phase runs in a subprocess that then exits. The warm
arm is a second pass over the group the first pass opened. An earlier version of the benchmark
re-opened everything for the warm arm, which made it a second cold reader and discarded the caches
the arm exists to observe — found in review, and the reason the warm and cold columns of an earlier
draft of this document agreed everywhere. The open phases have no warm figure at all, because their
measurement *is* the open, and a reader pays it once per handle rather than once per read. The
parent picks the probe pixels once and passes them down, so both arms and both regions read exactly
the same addresses.

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
through 300 probe pixels, every figure a median across the 10/64/128 concurrency sweep. The bucket
is in us-west-2, so us-east-1 is the cross-region case and its S3 traffic necessarily crosses the
public internet: a gateway endpoint is region-local, so no configuration makes a remote region's
reads local, and the asymmetry is part of what cross-region costs rather than a flaw in the setup.

| | us-west-2 (in-region) | us-east-1 (cross-region) | |
|---|---|---|---|
| open, to first read | 2,684 ms | 3,502 ms | 1.3× |
| one pixel's 128-band vector, p50 / p95 | **278 / 454 ms** | **441 / 618 ms** | 1.6× |
| …on a second pass over the same pixels | 277 / 396 ms | 428 / 546 ms | |
| patch, 100×100×128 | 4 MB/s (0.3 s) → 10 warm | 1 MB/s (1.1 s) → 4 warm | 3.9× |
| tile, 1000×1000×128 | 346 MB/s (0.4 s) → 504 warm | 82 MB/s (1.6 s) → 83 warm | 4.2× |
| band subset, 512×512×8 | 10 MB/s → 14 warm | 2 MB/s → 6 warm | 4.4× |
| bulk, 4096×4096×128 | **735 MB/s (2.9 s)** | **232 MB/s (9.2 s)** | 3.2× |

**Cross-region costs between 1.3× and 4.4×, and where a read lands depends on its shape rather
than its size.** Bulk pays 3.2× because it is bandwidth-bound and bandwidth is what distance takes
away. The open path pays only 1.3×, because most of it is neither bandwidth nor distance (§4.3). A
single pixel pays 1.6×.

**A second pass through the same open handle helps the region reads and not the point reads.**
In-region a tile goes 346 → 504 MB/s and a patch 4 → 10 MB/s, while a point read stays at 278 ms.
The wire bytes say why: a repeated tile read moves the same 133 MB either way, so the gain is warm
connections and threads rather than cached data, and a repeated point read re-fetches its whole
inner chunk because 300 chunks is 2.5 GB and icechunk's default cache does not hold that (§4.4).
Cross-region the warm figures are also noisier — a 2.1 GB bulk read came back at 232 MB/s cold and
168 MB/s on the second pass, so at that distance a single repeat is not reliably an improvement.

**Bytes on the wire, cold, confirm two things the throughput column cannot say on its own:**

| workload | logical bytes asked for | on the wire (us-west-2 / us-east-1) | |
|---|---|---|---|
| bulk, 4096×4096×128 | 2,147 MB | 2,087 / 2,111 MB | the embeddings barely compress |
| band subset, 512×512×8 | 2.1 MB | 34.9 / 35.0 MB | all 128 bands are fetched to return 8 |
| patch, 100×100×128 | 1.3 MB | 10.0 / 10.5 MB | one whole inner chunk, plus its shard index |
| one pixel | 128 B | 8.04 / 8.18 MB | one whole inner chunk |

The bulk row is the useful one: 2,087 MB crossing the wire to deliver 2,147 MB of array means a
compression ratio of 1.03, so on this store the logical throughput figures above *are* wire rates
to within a few percent.

### 4.2 Metadata and coverage navigation

| | us-west-2 | us-east-1 |
|---|---|---|
| repository open | 0.12 s | 0.46 s |
| root group open | 2.54 s | 2.77 s |
| a zone group, after the root is open (median of 120) | 14 ms | 14 ms |
| shard enumeration, one zone, all nine years (median of 120) | 0.52 s | 1.06 s |
| shard enumeration, **all 120 zones** | **60 s** | 121 s |
| registry: list all 994 parts | 0.17 s | 0.55 s |
| registry: read all 994 schema footers | 46 s | 189 s |
| registry: whole dataset, 3,247,410 rows | 14.9 s | 49.3 s |
| registry: **"is my area covered", one box, one year** | **0.80 s** | 2.81 s |

Two of these are the answers somebody will actually want. **A complete shard-level coverage map of
the whole globe takes a minute in-region** and two cross-region, which makes "what exists" a
question to ask rather than a table to maintain. And **the registry answers a coverage question
about an area of interest in under a second in-region**, which is the whole reason it exists.

Both regions independently found 3,229,545 live shards and 3,247,410 registry rows, with no
missing groups, no layout departures and no store-versus-registry disagreements. That the two hosts
agree is the check that neither run was partial.

### 4.3 The one inherited setting that dominates the open path

The store saves the repository configuration the campaign wrote with, so a reader who passes none
of their own inherits the writer's manifest preload — a budget of 1,000,000 refs across up to 2,400
arrays. (A reader going through `open_global_repo` gets the library's `global_store_config()`
instead, which is byte-identical to what the store saved; an integration test pins that, because
these figures describe one path and would quietly stop describing the other if they diverged.)

Both arms, same host, same 300 probes, medians across the sweep:

| | us-west-2 | | us-east-1 | |
|---|---|---|---|---|
| | as saved | preload off | as saved | preload off |
| open, to first read | 2,684 ms | **224 ms** | 3,502 ms | **1,403 ms** |
| opening the zone by path | 2,659 ms | 221 ms | 3,443 ms | 1,362 ms |
| one pixel, p50 | 278 ms | 268 ms | 441 ms | 463 ms |
| tile | 346 MB/s | 366 MB/s | 82 MB/s | 99 MB/s |
| bulk | 735 MB/s | 751 MB/s | 232 MB/s | 235 MB/s |

**It costs 2.5 seconds of every in-region open and buys the reader nothing measurable.** Point
latency, region throughput and bytes on the wire are all the same either way, within the run-to-run
spread. That is not a criticism of the setting: it exists so a *fill* did not re-fetch manifests it
was about to write into, and there it earns its keep. It is simply the wrong default to inherit as
a reader, and inheriting it is automatic.

**Opening the zone group by path does not avoid it.** That is the natural guess — that the root open
is slow because it enumerates 120 groups — and it is wrong: the cost moves out of the root-group
step and into the zone-group step, 2,659 ms against 2,684. The only thing that avoids it is asking
for a different configuration.

Two further readings. Across four separate runs the preload-free in-region open measured 143, 156,
158 and 224 ms, against the 128 ms scoping published — so the spread is real but the agreement is
close, and **the geometry was never what made the open slow**. And the cross-region penalty on the
preload-free path is 224 → 1,403 ms, about **6×**, against the 1.3× the full open path shows: the
preload is largely region-independent work, so leaving it in place hides most of the distance
penalty behind something slower than the penalty.

**For a reader, then:** fetch the saved config, turn preload off, keep everything else.

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
| zone open **128 ms** | 2,684 ms as documented, **224 ms** with preload off | **reconciled** — see below |
| bulk throughput, 13–2,014 MB/s across 24 samples, median **202** | 735 MB/s bulk, 346 tile, 10 band-subset, 4 patch | **consistent** — inside the range, comparable median |
| point p50 **29 ms**, p95 **204 ms** | p50 **278 ms**, p95 **454 ms** | **reconciled** — see below |
| **1.23 MB** on the wire per point read | **8.04 MB** | **reconciled** — see below |

**The open time reconciles, and the explanation is the preload.** Scoping's store was one group and
one year, so a 1,000,000-ref preload budget had almost nothing to fetch and cost almost nothing;
against a 120-group, nine-year repository the same budget is 2.5 s of work. With preload disabled
the published store opens to first read in 143–224 ms in-region across four runs, against the
128 ms scoping published. The geometry is behaving as scoped; what changed is what the inherited
configuration has to do.

**The throughput figures are consistent, and cannot be compared more sharply than that.** Scoping
retained only the distribution over its 24 `c256_sharded` throughput samples — four workloads ×
three concurrencies × two cache states — not the per-workload breakdown, and the raw run data is
gone. So "2,014 MB/s" is the maximum over that whole set and not a bulk-read figure; quoting it
against our bulk read would compare a maximum with a median. What can honestly be said is that our
four workloads span 4–735 MB/s with a median in the low hundreds, inside scoping's 13–2,014 range
around a comparable median.

**The point figures reconcile too, and the mechanism is the chunk cache against the working set.**

Start from what is solid. A point-vector read fetches **one whole 256×256×128 int8 inner chunk**,
which is 8.39 MB, and the quantized embeddings are effectively incompressible, so 8.39 MB is also
what crosses the wire. At in-region single-stream S3 rates that is a few hundred milliseconds. So
the measured 8.04 MB and 278 ms are one fact told twice, and any explanation has to fit both.

Three candidate explanations were eliminated first. **Compressibility** is not it: `synth.py`
generates random int8, deliberately worst-case, so scoping's data was as incompressible as ours.
**A codec difference between variants** is not it either: `variants.py` fixes only chunk and shard
geometry per variant and takes dtype, serializer and compressor from the library. And **partial
reads within an inner chunk** cannot be it, because a pixel cannot be returned without
decompressing the whole chunk that holds it — which is what makes ADR 008's reading of the gap,
that sharding does lean partial reads, not tenable as stated.

What remained was **cache reuse — and reuse alone is not enough, which is why a first attempt at
this explanation appeared to fail.** Asking whether probes revisit inner chunks, with the store's
own configuration, showed nothing: 1,000 probes over 16S's 1,088 inner chunks cost 7.96 MB each,
statistically identical to 25 probes over 33N's 549,952 at 8.28 MB. The reuse was there and the
saving was not, because the store **saves no caching setting** and icechunk's default cache is
small against an 8.39 MB chunk — every revisit had been evicted before it came round again. Reuse
only shows when the cache can hold the working set.

With that, the arithmetic is simple: if probes revisit inner chunks and the cache holds them, the
average bytes per read falls to `distinct chunks touched / probes` × 8.39 MB. Measured in
us-west-2 on 16S/2025, where 1,000 probes land in 1,088 inner chunks:

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
a whole 256×256×128 inner chunk on the wire — 8.39 MB, measured at 8.04 MB and 278 ms in-region,
because the quantized embeddings barely compress. That is the geometry working as intended: it is
what makes block reads fast and object counts manageable. But a consumer whose access pattern is
scattered single pixels will find it slow, and the fix is to read a region and index into it, not to
tune the client. Reading a 1000×1000 tile delivers 346 MB/s; reading its million pixels one at a
time would take three days.

**If reads do revisit the same neighbourhood, size the chunk cache to the working set.** The store
saves no caching setting, so a reader gets icechunk's own default, which is small against an
8.39 MB chunk. A 16 GiB cache on a workload whose 1,000 probes land in 1,088 inner chunks cut bytes
on the wire per read from 7.97 MB to 5.14 MB and p50 from 209 ms to 142 ms — almost exactly the
saving the repeat fraction predicts (§4.4). No cache helps a working set that never repeats, which
is why this is a separate recommendation from the one above rather than a replacement for it.

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
