# One area, or the whole world

This library does one thing: it turns satellite imagery into **embeddings** — a compact numerical
summary of what each patch of ground looked like over a year, which you can then feed to a
downstream model instead of the raw imagery. A patch is 10 metres square by default, and on the
single-area path the pixel size is a parameter you can change.

There are two ways to run it. You can point it at **one area you care about**, or you can run the
**global campaign** that covers the world's land between 59.45°S and 83.65°N — Antarctica is
excluded by decision. This page explains their points of similarity (many) and difference (few).

## The short version

Both paths run the same three stages in the same order, with the same code:

```
   ingest    →    inference    →    assembly    →   (validation)
   ------         ---------         --------          ----------
   fetch and      run the model     write the         currently global path
   mosaic the     on a GPU, or on   results into      only, and
   imagery        CPU for small     a store you       only if
                  runs              can read          configured
```

The global campaign can add a fourth step the single-area path does not have: once a zone-year is
written and tagged, it dispatches a validation run that produces figures and a machine-readable
verdict on the published cell. That unearths bad data in a job too large to
inspect by hand — but it is **configured, not automatic**. The validation deployment defaults to
unset, and with nothing set the dispatch does nothing, so a campaign you run yourself gets no
validation unless you supply one.

Otherwise, the **model, ingest, and inference are identical**. Assembly shares its code up to the
point of writing, and then the two write differently — the global path lays down whole 2048-pixel
tiles into slots that were set aside in advance,in order to let many machines add to one
dataset at once; the single-area path creates or extends a store of its own. That difference
reaches the data in one place, and it is worth knowing before you compare the two: where a tile
was looked at and refused — no pixel in it passed the quality rules — the global path still writes
the observation counts it measured there, while the single-area path leaves them at fill. So a
refused footprint reads as zero observations in a single-area output and as the real count in the
global store. The embeddings are fill either way; it is the counts beside them that disagree. The
asymmetry is deliberate, and the reason is in a comment at the write site. What changes otherwise
is how much you run at once, how you say which ground you want, and what the output store looks
like when it lands.

If you are uncertain, you almost certainly should use the single-area path. If you need global coverage,
first check if the published multi-year global dataset at s3://tessera-embeddings/v1.1/dclimate.icechunk
works for you. The single path is simpler and more flexible about time periods, can be flexibly run
on different cluster sizes or single machines, and is vastly more cost-effective for small area analysis.

## What is genuinely the same

**The same inference code runs in both.** The single-area flow and the global campaign both call
the same `run_inference` function, producing the same 128-dimensional embedding per pixel. There is
no "global model" and no "small model" — by default they run the same checkpoint too, though the
single-area path lets you point at another one with `checkpoint_url`, in which case the outputs are
of course no longer comparable.

**The same ingest code runs in both, and it sees the same kind of input.** This is the part that
surprises people. The global campaign holds its coverage information in one format and the
single-area path in another — but before ingest runs, the global path *converts* its coverage into
exactly the artefact the single-area path uses: a plain boolean grid saying where you want
embeddings. From that point on, the two are indistinguishable to the ingest code.

**Cost follows the area you keep far more closely than the box you draw**, in both paths and with
no flag to set. Be precise about the unit, though: work is skipped a **chunk at a time**, not a
pixel at a time — ingest works in 4096-pixel windows and inference in 2048-pixel tiles, and any
window your mask touches at all is processed whole. So a compact area is close to free beyond its
own extent (one sparse island zone drops from 3,706 chunks per band-date to 4), while an area that
is small but *scattered* — a thousand separate field boundaries across a region — can touch many
tiles and cost accordingly. If your area is sparse, its cost is set by how many tiles it lands on.

### But it is not configured the same, and that part does change the answer

The shared code is run with **different default settings**, and three of them change what comes out.
The same area and the same year can therefore differ depending on which path produced it:

| setting | single area | the global campaign |
|---|---|---|
| `allow_s2_only` | `false` — a pixel with no radar observation produces nothing | **`true`** — optical-only pixels are embedded |
| `optical_min_obs` | unset — no minimum | **15** — a pixel with fewer than fifteen clear optical observations in the year is left empty |
| `min_valid_coverage` | **5%** of the area's pixels must be cloud-free for a date to be kept | **0.1%** |

The first two are the only quality rules applied per pixel, and both were deliberate choices for a
global run: about a fifth of the land has no radar for 2022–24, so without `allow_s2_only` those
pixels would be holes, and fifteen observations is the line below which an embedding was judged
untrustworthy.

**The third is easy to miss and matters as much**, because it throws away whole dates before
inference ever sees them. A zone-sized area is mostly ocean and edge, so the single-area default of
5% would discard nearly every date over a zone — hence 0.1% for the campaign. Run a small, compact
area at 0.1% and you keep dates the 5% default would have dropped, and the embeddings differ
accordingly. **Note the trap:** the two values live in two constants that share the name
`DEFAULT_MIN_VALID_COVERAGE`, in `config/ingest.py` (0.1) and `ingest/roi_processing.py` (5.0), so
reading one of them tells you nothing about which applies to your run.

**What this means in practice.** If you compare your own single-area output against the published
global store and the pixels disagree, check these settings before looking for a bug.

**Exact parity is not reachable from the single-area entry points today, and you should know that
before trying.** `allow_s2_only` is a parameter you can pass. `optical_min_obs` is **not exposed**
on either documented single-area path — neither the plain runner nor the Prefect flow reads it — so
a single-area run applies no optical-depth floor and there is currently no way to ask it for the
campaign's fifteen. Wiring it through would be a small change to the library and is on the roadmap.

**What the published store lets you check, and what it does not.** Only one of the three is
recorded, and it is on the store's ROOT attributes rather than on a zone:

```python
ds_root = xr.open_zarr(session.store, consolidated=False)   # no group=
ds_root.attrs["optical_min_obs"]     # 15 — the line every cell was measured against
ds_root.attrs["checkpoint_id"]       # which model produced it, plus the geoemb: provenance
```

`allow_s2_only` and `min_valid_coverage` are **not** published. The coverage threshold is kept in
the private ingest manifest and never copied into the store, so a consumer cannot recover it from
the data. **Exact reproduction of a published cell therefore needs the run's own parameters, not
just the store** — ask whoever produced it. What the store does answer is the question that matters
most often: which optical-depth line a cell was held to, and which model wrote it.

## What actually differs

| | one area | the whole world |
|---|---|---|
| what you name | an area of interest you supply | UTM zones and years from a fixed list* |
| how you say where | a boolean grid you make, at pixel resolution | a prepared coverage store, at 2048-pixel tile resolution |
| time period | **any 12 months**, ending in the month you choose | **calendar years only**, January to December |
| output | one store per area, one entry per window | one store for the world, one entry per zone per year |
| scale | one machine, one GPU or a few | **a peak of 1,307 single-GPU machines**, plus a large container fleet, for about two weeks |
| you run it | yourself, when you want | as a campaign, with restart and recovery machinery |

\* A **UTM zone** is a common convention in geography. The UTM system divides the world into 
60 northern and southern strips for mapping, each six degrees of longitude wide. 
Each is split at the equator into a northern and a southern half with
its own flat coordinate system, which gives the **120** groups this store has, named like `33N` and
`33S` — the same longitude band, opposite hemispheres. The campaign works a zone at a time because
that flat coordinate system is what lets imagery be processed without distortion. You do not need
to care about any of this for the single-area path: it works out the right coordinate system from
your area.

Everything in that table is about *packaging and scale*. None of it is about how an embedding is
computed.

## Time periods: the one difference that will bite you

**The single-area path takes any 12-month window.** You name the month it ends in, and you get the
twelve months up to and including it. Ask for `"June 2025"` and you get July 2024 through June
2025. This is genuinely useful: a growing season, a monsoon year, or a window chosen to sit
between two events rarely lines up with January.

> **You set the window and the imagery range separately, and on the plain runner nothing checks
> they agree.** A config carries both a `time_window_end`, which is the label the output is written
> under, and a `time_range`, which is the span actually ingested. The Prefect flow refuses a window
> its inputs do not cover; the plain runner does not check at all, so it will happily label one
> month of imagery as a twelve-month embedding — the shipped quickstart config does exactly that,
> deliberately, to keep a laptop run short. **If you intend a real twelve-month embedding, set
> `time_range` to span those twelve months yourself.**

**The global store takes calendar years only.** Ask the global path for anything but January to
December and it refuses before it spends any money on GPUs.

**The reason is the store's convention, not the model.** The model computes a twelve-month window either way.
The global store's time axis has one slot per calendar year, and by each slot carries a label saying
it covers 1 January to 31 December. If a non-calendar window were written into a slot labelled
that way, anyone reading the store later would be told something untrue about what they were
looking at.

Global stores that support non-calendar year time windows are technically entirely possible
and we are comfortable making the required code changes if a genuine use case
(and corresponding budget) are expressed to us.

Single-area stores use a different and equally explicit convention: one entry per window, labelled
by the month the window ended. Nothing is lost — the two stores just make different promises.

**Variable time segments are on the roadmap for the global store.** It is a change to how the
store labels and lays out time, not a change to the model or the pipeline, which is why the
single-area path can already do it.

## Selecting which areas you do, and don't, want

### For one area: a boolean grid, and it does not have to be land

The single-area path wants a **mask**: a grid of true and false values, the same shape as the
imagery, true wherever you want embeddings. That is the whole contract. Anything the pipeline can
turn into that grid will work.

**It carries no notion of "land".** The global campaign's mask happens to describe land, because
that is what that campaign is for. Yours can be a city, a catchment, a set of farm boundaries, a
protected area, a coastline, a study plot, or a lake — anything you can draw. The code only ever
asks "is this pixel wanted?"

**What the mask does not free you from is radar, unless you ask.** Both single-area entry points
resolve a Sentinel-1 orbit for the run, and by default neither accepts "none" — so an area with no
Sentinel-1 store behind it will not complete, and you should ingest both sensors even if your
interest is optical. The Prefect flow does have a way out: pass `require_s1=False` and it will
finish optical-only where there is genuinely no radar. The plain runner has no such switch; it
demands an orbit, and that is not configurable.

Two supported ways to make one, both a single flow run:

1. **From a polygon.** Supply a GeoJSON file and it is rasterised onto the imagery grid for you.
   This is the usual route, and the one the quickstart uses for its square kilometre over Denver.
2. **From Sentinel-2 tile names.** Name one or more Sentinel-2 tiles — the roughly 110 km squares
   the mission publishes its imagery in, with codes like `13TDE` — and their footprints become your
   area. Useful when you want your outputs to line up with someone else's tiling exactly.
   **This route needs a tile index the repository does not ship:** it reads
   `sentinel2_tiles.geojson` from the root of your ROI bucket, so you have to put a Sentinel-2
   tiling-grid GeoJSON there first. Without it the run fails on a missing file before anything is
   rasterised. The polygon route has no such prerequisite.

Either way the result is a small **Zarr** file. Using Zarr lets us store our big boolean
grids in chunks so a reader can fetch only the part it needs and apply them.

You can write one yourself, but the pipeline needs more than the array. It reads three attributes
off it and fails without them:

| attribute | what it is |
|---|---|
| `crs` | the coordinate reference system the grid is in, as a string |
| `transform` | the six affine coefficients mapping array indices to projected coordinates |
| `bbox_wgs84` | the bounding box in ordinary longitude and latitude, used to query the catalogues |

The array itself is a chunked boolean of shape `(height, width)`. If your mask is in another form,
the least error-prone route is still to hand the pipeline a GeoJSON outline and let it rasterise —
it produces all of this for you, on the right grid.

**Two things to get right.** The mask must sit on the same coordinate system and grid as the
imagery you are ingesting — the flow handles this when it rasterises for you if you supply a
polygon.

And **a mask that selects nothing does not fail.** The rasteriser reports how many pixels it
selected and carries on, and a run over an empty mask completes having produced nothing of
substance. So read the valid-pixel count in the log of your mask-building run before you spend
anything on GPUs — a polygon in the wrong coordinate system, or one that misses the imagery's
extent, looks exactly like this.

### For the whole world: a prepared coverage store

The global campaign does not rasterise a polygon. Its coverage comes from a prepared store holding
one bitmap per UTM zone, marking which 2048-pixel tiles contain land. That store is built once
from a global delivery of small per-cell files, and the campaign reads it to decide what exists.

Two consequences worth knowing:

- **It is tile-granular, not pixel-granular.** A coastal tile with any land in it is included
  whole, so the ocean pixels inside that tile come along — and they are **embedded, not dropped**.
  Water is a valid surface class as far as the cloud mask is concerned, so a sea pixel in a live
  tile gets an embedding like any other. That is deliberate: it keeps the unit of work a whole
  tile. What gets skipped is whole tiles and chunks *outside* the coverage mask, which is where
  nearly all the saving comes from.
- **Coverage extends a little way offshore** — roughly 11 km — because the upstream mask was built
  with a generous margin for other users. So "land" here is slightly more than land.

The global run covers **land between 59.45°S and 83.65°N**. Antarctica is excluded on purpose: the
coverage source has no Antarctic land, and the UTM grid used for the zones cannot represent it
anyway.

## The published global dataset

The global campaign writes to one **Icechunk** store — Zarr arrays plus a transaction log, so
every write is atomic and the history is readable, which is what makes it safe for many machines
to add to one dataset at once:

```
s3://tessera-embeddings/v1.1/dclimate.icechunk/
```

This is **global TESSERA v1.1**. It holds 120 groups, one per UTM zone, named by hemisphere
(`01N`–`60N`, `01S`–`60S`). 112 of them contain land; the other eight are ocean. Each group has an
annual time axis for **2017 to 2025** and holds embeddings as 8-bit integers with 128 values per
pixel, stored in 2048-pixel tiles, alongside the scale factors needed to interpret them and
per-pixel counts of how many observations fed each result.

The store is hosted by AWS Open Data, which sponsors its storage, so reading it does not bill the
project that produced it.

**Check that the cell you want was filled before you read it** — see
[Which should I use?](#which-should-i-use) below. A cell that was never filled opens without
complaint and hands back fill values rather than an error.

### Reading a zone group

Open a zone group through a read-only Icechunk session, and ask xarray to decode the variables
that CF conventions link to a coordinate:

```python
import xarray as xr
from tessera_embeddings.storage.global_store import open_global_repo

repo = open_global_repo(
    "s3://tessera-embeddings/v1.1/dclimate.icechunk",
    region="us-west-2", anonymous=True,
)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, group="33N", consolidated=False, decode_coords="all",
                  chunks=None)
```

```
<xarray.Dataset>
Coordinates:
  * time         (time) datetime64[ns] 2017-01-01 2018-01-01 ... 2025-01-01
  * northing     (northing) float64 ...
  * easting      (easting) float64 ...
  * band         (band) int64 0 1 ... 127
    time_bnds    (time, bnds) datetime64[ns] ...     ← [Jan 1, Dec 31] per slot
Data variables:
    embeddings   (time, northing, easting, band) int8 ...
    scales       (time, northing, easting) float32 ...
    s2_obs_count (time, northing, easting) uint16 ...
```

`anonymous=True` is what sends an unsigned request, which is why this needs no AWS credentials
at all. There is nothing else to pass: the published store is already configured for readers.
See [Which should I use?](#which-should-i-use) if your copy of the library does not accept the
argument.

**`chunks=None` matters on a zone this size.** Without it xarray hands back Dask-backed arrays,
and a zone is large enough that the graph describing one runs to millions of chunks: reading a
single pixel through it took about three seconds and peaked near two gigabytes, against a fifth of
a second and under 200 MB with `chunks=None`. Neither is free — xarray still builds the zone's
variables and materialises a 933,888-element `northing` coordinate either way — but one of them
scales with the zone and the other does not. The same advice, and the same reason, is in the
[inference README](../src/tessera_embeddings/inference/README.md#write-units-vs-read-units-per-layout).

**`decode_coords="all"`** is what promotes `time_bnds` from a data variable to a coordinate:
xarray treats variables referenced by a `bounds` or `grid_mapping` attribute as coordinates.
Without it the dataset holds exactly the same numbers; `time_bnds` just lists lower down.

**What each `time` point means, guaranteed.** Each one is **January 1 of its calendar year — the
start of the exact January-to-December window that slot holds**
(`time_convention="calendar_year"`, and the fill runner rejects any other window, so the label
always matches the data). The companion `time_bnds` variable, shape `(time, 2)` and linked from
`time.attrs["bounds"]` as CF requires, states each slot's interval outright:
`[YYYY-01-01, YYYY-12-31]`. Rolling twelve-month windows are never written here; they belong in a
single-area store, whose convention is one entry per window, labelled by the month it ended.

**How many observations fed each pixel.** Three count layers record it —
`s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count` — always written, with `0` meaning none.
By default every embedded pixel has at least one radar observation. Where a fill ran with
`allow_s2_only=True`, as the global campaign does, optical-only pixels are embedded too, and they
are exactly those with a finite `scales` value and
`s1_asc_obs_count + s1_desc_obs_count == 0`. The quality of an optical-only embedding has not been
validated against a radar-informed one — see
[ADR-013](../context_docs/decisions/013-optional-s1-s2-only-pixels.md).

Reference docs:
[`xarray.open_zarr` / `decode_coords`](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html) ·
[xarray weather & climate (CF) guide](https://docs.xarray.dev/en/stable/user-guide/weather-climate.html) ·
[CF conventions §7.1 Cell Boundaries](https://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries) ·
[cf-xarray bounds handling](https://cf-xarray.readthedocs.io/en/latest/bounds.html)

### Why chunk size dominates everything

Both paths use the same grid, so none of this is specific to the global store — but the
global store is where getting it wrong is most expensive, and it is the reason the shard
sizes above are the numbers they are.

A subtle reality of distributed array workloads: **the task graph your scheduler has to
plan grows quadratically with how finely you chunk the data.** Chunk too small and the
scheduler spends more time managing tasks than the tasks spend doing work — on a
20 km × 20 km area of interest, 200-pixel chunks build a graph of ten thousand nodes,
which costs tens of seconds and about a gigabyte of scheduler memory before any data is
read, and leaves overhead as most of the wall clock. Chunk too large and a worker cannot
fit one chunk in memory at all.

Storage and read granularity are tuned separately. Ingest writes `INGEST_CHUNK_SIZE =
4096` storage chunks to keep the satellite-ingest Dask graph small (a quarter of the
spatial tasks), while inference reads a smaller sub-tile out of them — small enough to
keep peak GPU-node RAM in check. Zarr's `oindex` reads a sub-tile out of a 4096 chunk with
no alignment requirement, so the two sizes are independent.

The read tile divides the output chunking, and both paths use the same one:
`INFERENCE_CHUNK_SIZE = 2048`, so one inference tile is exactly one 2048-pixel shard
([ADR-008](../context_docs/decisions/008-global-store-architecture.md) D3). The global
campaign also passes it explicitly, since that path requires the identity rather than
merely matching it. Go smaller on the ingest chunk and the
satellite-ingest scheduler drowns in tasks; go larger on the read tile and you exhaust the
memory of a single-GPU worker. If you change either, profile.

The powers of two are not cosmetic — they align every stage of the pipeline on one grid,
so no stage rechunks its input:

```
ingest chunk    4096 px  = 2×2 inference tiles
inference tile  2048 px  = 1 output shard
shard           2048 px  = 8×8 inner chunks
inner chunk      256 px  = the unit downstream readers decode

one ingest store chunk (4096²) — what one satellite read/write touches
┌─ inference tile (2048²) ─┬─ inference tile (2048²) ─┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░ │                          │
│ ░ 8×8 grid of 256²     ░ │   each tile is read out  │
│ ░ inner chunks — the   ░ │   of the ingest chunk by │
│ ░ same grid the output ░ │   one GPU actor, staged  │
│ ░ shard will store     ░ │   as one file, and lands │
│ ░░░░░░░░░░░░░░░░░░░░░░░░ │   as ONE shard object    │
├──────────────────────────┼──────────────────────────┤
│                          │                          │
│    inference tile        │    inference tile        │
│                          │                          │
└──────────────────────────┴──────────────────────────┘
```

### How the store is laid out

You do not need any of this to read the store. It is here because the layout is what makes a
job this size possible at all, and because two of the choices are visible to a consumer.

**Zones are named, not numbered.** Zone groups, mosaic paths and campaign tags all use the UTM
**common name** — `canonicalize_zone` parses `"33n"` or `" 7s "` into `"33N"` and `"07S"`. This is
a deliberate deviation from the geoembeddings `utm_zones` specification, whose `utm{NN}` group
name cannot say which hemisphere it means. The EPSG code (326xx north, 327xx south) is kept, but
only as the coordinate reference system.

**Zones are pure six-degree longitude bands.** Every pixel centre falls in exactly one zone, and
the per-zone pixel grids come from the EPSG registry (`storage/zone_grid.py`), snapped to the
20,480 m shard pitch. The Norway and Svalbard width exceptions that MGRS makes (32V, and 31X–37X)
are deliberately **not** honoured: they exist for navigation, not for data grids. **If you work
near those zones, do not assume MGRS behaviour** — the dataset says so in each group's
`zone_scheme: "utm_6deg_nominal"` attribute.

**A shard is one S3 object holding an 8×8 grid of smaller chunks.** That is the geometry that
makes a point read cheap without making a write expensive:

```
zone group "33N" ▸ embeddings ▸ year 2025 ▸ one shard
┌─ shard object (2048² px × 128 bands ≈ 0.5 GB max on S3) ────────────┐
│   8×8 inner chunks, 256² px × 128 bands (~8.4 MB int8+zstd each)    │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐                         │
│   │▓▓▓▓│▓▓▓▓│▓▓▓▓│    │    │▓▓▓▓│▓▓▓▓│▓▓▓▓│   ▓ = data: encoded    │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤       bytes + an index  │
│   │▓▓▓▓│▓▓▓▓│    │    │    │    │▓▓▓▓│▓▓▓▓│       entry             │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤   blank = all-fill (no  │
│   │▓▓▓▓│    │    │    │    │    │    │▓▓▓▓│     valid observations):│
│   └────┴────┴────┴────┴────┴────┴────┴────┘       zero bytes stored │
│   + shard index: inner chunk → (offset, length)    — a "lean" shard │
└──────────────────────────────────────────────────────────────────────┘

WRITE  one staged inference tile (2048²) is exactly one shard: the assembly worker
       emits the whole object once — no read-modify-write — and an all-ocean tile
       costs nothing, because it is never staged and never written.
READ   a point or window read fetches the shard index, then asks for only the byte
       ranges of the inner chunks it overlaps — about 8 MB for a point, not 0.5 GB.
```

Single-area stores use the same geometry; the two presets are one definition under two names.

**Manifests are split by year, so a commit costs one year rather than the whole store.** An
Icechunk **manifest** is the index that maps every chunk to the object holding it. By default
there is one per array, so every commit rewrites the entire index no matter how little changed.
The global store splits manifests at `time@1`:

```
    unsplit (default)                     split time@1 (global store)
    one manifest per array                one manifest per (array, year)

    MANIFEST: all 9 years                 M2017 M2018 ⋯ M2024 M2025
    ┌────────────────────────┐            ┌────┐┌────┐  ┌────┐┌────┐
    │ every (year, y, x)     │            │ ρρ ││ ρρ │  │ ρρ ││ ρρ │
    │ chunk → object ref     │            └────┘└────┘  └────┘└─▲──┘
    └───────────▲────────────┘                                  │
                │                         WRITE  filling 2025 rewrites
    WRITE  ANY commit rewrites                   only M2025 — commit
           the whole thing:                      cost stays O(one year)
           O(entire store)                       for all nine years
                                          READ   opening a group loads
                                                 only the manifests of
                                                 the arrays/years read
```

Single-area stores use the same idea spatially: a 32-chunk-per-axis two-dimensional split, so
rewriting a region rewrites only the manifest tiles it touches (`zarr_store.manifest_split`).

**Four write paths, all committing atomically.** The first three are in
`storage/zarr_store.py`, the fourth in `inference/assembly.py` and
`storage/shard_writer.py`:

1. **create** — `write_dataset` on a fresh store. It adopts a repository an interrupted attempt
   left behind, rather than failing forever on a dirty prefix.
2. **append** — extend the time axis of an existing store.
3. **region overwrite** — rewrite a slice, in time or space, in place.
4. **shard-assemble** — staged inference tiles written as whole, lean 2048-pixel shards into a
   pre-allocated zone group, one fork-and-merge commit per (zone, year). These commits are
   ungated: they contend on the repository's single branch tip, which costs seconds and never a
   conflict. The mechanics are in
   [`context_docs/storage/writing-to-the-global-store.md`](../context_docs/storage/writing-to-the-global-store.md).

The full write-path reference, including what a fork worker does and how a resume tells a
finished tile from an interrupted one, is in the
[inference README](../src/tessera_embeddings/inference/README.md).

## Which should I use?

**Use the single-area path if** you have a study area, you want one or several time windows that are
not a calendar year, you want to iterate quickly, or you are evaluating whether these embeddings help
you at all.

**Use the global dataset if** the area you want is already filled. Reading a published store is
free and instant compared with computing anything.

> Note that the store is an Icechunk repository, so you open it through a session rather than by 
> handing the URI straight to Zarr or xarray:
>
> ```python
> import zarr
> from tessera_embeddings.storage.global_store import open_global_repo
>
> repo = open_global_repo(
>     "s3://tessera-embeddings/v1.1/dclimate.icechunk",
>     region="us-west-2", anonymous=True,
> )
> session = repo.readonly_session(branch="main")
> zone = zarr.open_group(session.store, mode="r")["33N"]
> print(zone.attrs)      # check all the attrs for a zone
> ```
>
> **Ask Zarr for this rather than xarray.** You want one line of metadata, not data, and
> `xr.open_zarr` insists on describing every array in the zone before it will show you an
> attribute — millions of pieces for a zone this size, four times the wait and eight times the
> memory. Use xarray when you actually want the embeddings, as the
> [example above](#reading-a-zone-group) does.
>
> **You do not need an AWS account to run this.** The bucket's policy grants anyone read access,
> and `anonymous=True` is what makes the library send an unsigned request, so the whole example
> works with nothing set in your environment. **Nothing else needs passing.** The published store
> is configured for readers, so opening it does no extra work on your behalf and there is no
> performance argument to tune.
>
> **Read the zone.attrs["year_complete"] list carefully,**
> **because it distinguishes two different things from a third.** A year
> *in* the list either holds data or was deliberately marked as having none — an all-ocean zone, or
> land where the campaign looked and found nothing usable. Either way the question has been
> answered. A year *missing* from the list is different: that cell never landed, which is a gap in
> the publication rather than a statement about the imagery. **Investigate a missing year; do not
> read it as "there was nothing there".**

**Run the global campaign yourself only if** you need coverage or years the published store does
not have, and you have the infrastructure for it. It is a large, long, expensive job with its own
operational machinery; [`context_docs/campaign/campaign-plan.md`](../context_docs/campaign/campaign-plan.md)
describes what that involves.

**What one costs is now measured rather than estimated.** The completed campaign published
992 cells — 3,247,400 tile-years, 99.96% of what it set out to cover — for about **$828,000**
at on-demand list prices, using 360,282 graphics-card hours over 36 days of billed compute.
Graphics cards were only about two thirds of that; storage, object-store requests and the
container fleet made up most of the rest.
[`context_docs/campaign/campaign-cost-model.md`](../context_docs/campaign/campaign-cost-model.md)
§12 is the record, with every line and the usage it was derived from, and it is the figure to
quote rather than any of the planning estimates earlier in that document.

## Where to go next

- [`quickstart.md`](quickstart.md) — the single-area path, end to end, on a laptop
- [`configuration.md`](configuration.md) — the configuration objects and what each field does,
  including the time window
- [top-level README](../README.md#the-global-embeddings-store) — the overview of the store and the
  code that fills it
- [`context_docs/decisions/008-global-store-architecture.md`](../context_docs/decisions/008-global-store-architecture.md)
  — why the global store is shaped the way it is
- [`context_docs/decisions/010-landmask-registry-coverage.md`](../context_docs/decisions/010-landmask-registry-coverage.md)
  — why the global coverage store is tile-granular
