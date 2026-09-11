# One area, or the whole world

This library does one thing: it turns satellite imagery into **embeddings** — a compact numerical
summary of what each patch of ground looked like over a year, which you can then feed to a
downstream model instead of the raw imagery. A patch is 10 metres square by default, and on the
single-area path the pixel size is a parameter you can change.

There are two ways to run it. You can point it at **one area you care about**, or you can run the
**global campaign** that covers the world's land between 59.45°S and 83.65°N — Antarctica is
excluded by decision, not omitted by accident. This page explains how they relate, because
the honest answer is that they are far more alike than different, and knowing which differences
are real will save you guessing.

## The short version

Both paths run the same three stages in the same order, with the same code:

```
   ingest    →    inference    →    assembly    →   (validation)
   ------         ---------         --------          ----------
   fetch and      run the model     write the         global path
   mosaic the     on a GPU, or on   results into      only, and
   imagery        CPU for small     a store you       only if
                  runs              can read          configured
```

The global campaign can add a fourth step the single-area path does not have: once a zone-year is
written and tagged, it dispatches a validation run that produces figures and a machine-readable
verdict on the published cell. That is how bad published data gets noticed in a job too large to
inspect by hand — but it is **configured, not automatic**. The validation deployment defaults to
unset, and with nothing set the dispatch does nothing, so a campaign you run yourself gets no
validation unless you supply one.

The **model is identical** and the **ingest is identical**. Assembly shares its code up to the
point of writing, and then the two write differently — the global path lays down whole 2048-pixel
tiles into slots that were set aside in advance, which is what lets many machines add to one
dataset at once; the single-area path creates or extends a store of its own. That difference
reaches the data in one place, and it is worth knowing before you compare the two: where a tile
was looked at and refused — no pixel in it passed the quality rules — the global path still writes
the observation counts it measured there, while the single-area path leaves them at fill. So a
refused footprint reads as zero observations in a single-area output and as the real count in the
global store. The embeddings are fill either way; it is the counts beside them that disagree. The
asymmetry is deliberate, and the reason is in a comment at the write site. What changes otherwise
is how much you run at once, how you say which ground you want, and what the output store looks
like when it lands.

If you are choosing: use the single-area path unless you actually need global coverage. It is
simpler and more flexible about time periods, and it can run entirely on one machine — the
quickstart does. (The full Prefect pipeline flow provisions a Dask cluster for the ingest stage and
auto-sizes it from the area, so "one machine" describes the plain runner rather than every
single-area entry point.)

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

So: if you are wondering whether the global campaign does something cleverer to the imagery, or
uses a better model, or has a different definition of an embedding — it does not.

### But it is not configured the same, and that part does change the answer

The shared code is run with **different settings**, and three of them change what comes out. The
same area and the same year can therefore differ depending on which path produced it:

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
campaign's fifteen. Wiring it through would be a small change to the library, not a configuration
choice.

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
| scale | one machine, one GPU or a few | **around 2,500 single-GPU machines**, days of running |
| you run it | yourself, when you want | as a campaign, with restart and recovery machinery |

\* A **UTM zone** is one of 60 north–south strips the world is divided into for mapping, each six
degrees of longitude wide. Each is split at the equator into a northern and a southern half with
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

**The reason is the store, not the model.** The model computes a twelve-month window either way.
The global store's time axis has one slot per calendar year, and each slot carries a label saying
it covers 1 January to 31 December. If a non-calendar window were written into a slot labelled
that way, anyone reading the store later would be told something untrue about what they were
looking at. So the refusal protects readers of a dataset published for other people to use, and it
is checked once, early, rather than discovered late.

Single-area stores use a different and equally explicit convention: one entry per window, labelled
by the month the window ended. Nothing is lost — the two stores just make different promises.

**Variable time segments are on the roadmap for the global store.** It is a change to how the
store labels and lays out time, not a change to the model or the pipeline, which is why the
single-area path can already do it.

## Saying which ground you want

### For one area: a boolean grid, and it does not have to be land

The single-area path wants a **mask**: a grid of true and false values, the same shape as the
imagery, true wherever you want embeddings. That is the whole contract. Anything the pipeline can
turn into that grid will work.

**It carries no notion of "land".** The global campaign's mask happens to describe land, because
that is what that campaign is for. Yours can be a city, a catchment, a set of farm boundaries, a
protected area, a coastline, a study plot, or a lake — anything you can draw. The code only ever
asks "is this pixel wanted?"

**What the mask does not free you from is radar.** The single-area path resolves a Sentinel-1 orbit
for the run and will not accept "none", so an area with no Sentinel-1 store behind it cannot
complete — ingest both sensors, even if your interest is optical.

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

Either way the result is a small **Zarr** file — a directory-shaped array format that stores big
grids in chunks so a reader can fetch only the part it needs.

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

Reading one zone is an ordinary `xarray` open — the top-level
[README](../README.md#reading-a-zone-group-xarray) has a worked example, including which options
you need for the time bounds and coordinate reference to come through correctly.

The store is hosted by AWS Open Data, which sponsors its storage, so reading it does not bill the
project that produced it.

## Which should I use?

**Use the single-area path if** you have a study area, you want a time window that is not a
calendar year, you want to iterate quickly, or you are evaluating whether these embeddings help
you at all. It runs on one machine and the quickstart finishes on a laptop in a few minutes.

**Use the global dataset if** the area you want is already filled. Reading a published store is
free and instant compared with computing anything.

> **Check before you trust it, and do not infer coverage from latitude and year.** Every
> zone-and-year slot in the store is created in advance, before any imagery is processed. That
> means a cell that was never filled **opens perfectly happily and hands back fill values** — you
> will not get an error, you will get plausible-looking nothing.
>
> The authority is each zone group's `years_complete` attribute. Note that the store is an
> Icechunk repository, so you open it through a session rather than by handing the URI straight to
> Zarr or xarray:
>
> ```python
> import zarr
> from tessera_embeddings.storage.global_store import open_global_repo
>
> repo = open_global_repo(
>     "s3://tessera-embeddings/v1.1/dclimate.icechunk",
>     region="us-west-2", anonymous=True, preload_manifests=False,
> )
> session = repo.readonly_session(branch="main")
> zone = zarr.open_group(session.store, mode="r")["33N"]
> print(zone.attrs["years_complete"])      # the years you can read
> ```
>
> **Ask Zarr for this rather than xarray.** You want one line of metadata, not data, and
> `xr.open_zarr` insists on describing every array in the zone before it will show you an
> attribute — millions of pieces for a zone this size, four times the wait and eight times the
> memory. Use xarray when you actually want the embeddings, as the
> [README example](../README.md#reading-a-zone-group-xarray) does.
>
> **You do not need an AWS account to run this.** The bucket's policy grants anyone read access,
> and `anonymous=True` is what makes the library send an unsigned request, so the whole example
> works with nothing set in your environment. `preload_manifests=False` is worth passing as well:
> the store carries a saved setting, sized for the job of writing it, that costs a reader about two
> and a half seconds on every open and buys nothing back — the measurement is in
> `context_docs/storage/reading-the-published-store.md` §4.3. Both of those arguments arrive with
> the change that opened the store to anonymous readers, so on an older copy of the library they
> will not be accepted: drop them and supply any AWS credentials instead, which is all the previous
> version needed.
>
> **Read that list carefully, because it distinguishes two different things from a third.** A year
> *in* the list either holds data or was deliberately marked as having none — an all-ocean zone, or
> land where the campaign looked and found nothing usable. Either way the question has been
> answered. A year *missing* from the list is different: that cell never landed, which is a gap in
> the publication rather than a statement about the imagery. **Investigate a missing year; do not
> read it as "there was nothing there".**

**Run the global campaign yourself only if** you need coverage or years the published store does
not have, and you have the infrastructure for it. It is a large, long, expensive job with its own
operational machinery; [`context_docs/campaign/campaign-plan.md`](../context_docs/campaign/campaign-plan.md)
describes what that involves and
[`context_docs/campaign/campaign-cost-model.md`](../context_docs/campaign/campaign-cost-model.md)
estimates what one costs. Read that as a planning model rather than a bill: it was written before
the campaign ran, and what the completed run actually cost is still being written up.

## Where to go next

- [`quickstart.md`](quickstart.md) — the single-area path, end to end, on a laptop
- [`configuration.md`](configuration.md) — the configuration objects and what each field does,
  including the time window
- [top-level README](../README.md#the-global-embeddings-store) — how the global store is laid out
- [`context_docs/decisions/008-global-store-architecture.md`](../context_docs/decisions/008-global-store-architecture.md)
  — why the global store is shaped the way it is
- [`context_docs/decisions/010-landmask-registry-coverage.md`](../context_docs/decisions/010-landmask-registry-coverage.md)
  — why the global coverage store is tile-granular
