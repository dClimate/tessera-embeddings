# One area, or the whole world

This library does one thing: it turns satellite imagery into **embeddings** — a compact numerical
summary of what each 10-metre patch of ground looked like over a year, which you can then feed to
a downstream model instead of the raw imagery.

There are two ways to run it. You can point it at **one area you care about**, or you can run the
**global campaign** that covers all the world's land. This page explains how they relate, because
the honest answer is that they are far more alike than different, and knowing which differences
are real will save you guessing.

## The short version

Both paths run the same three stages in the same order, with the same code:

```
   ingest    →    inference    →    assembly    →   (validation)
   ------         ---------         --------          ----------
   fetch and      run the model     write the         global path
   mosaic the     on a GPU          results into      only: check
   imagery                          a store you       what landed
                                    can read
```

The global campaign adds a fourth step the single-area path does not have: once a zone-year is
written and tagged, it dispatches a validation run that produces figures and a machine-readable
verdict on the published cell. That is how bad published data gets noticed in a job too large to
inspect by hand.

The **model is identical** and the **ingest is identical**. Assembly shares its code up to the
point of writing, and then the two write differently — the global path lays down whole 2048-pixel
tiles into slots that were set aside in advance, which is what lets many machines add to one
dataset at once; the single-area path creates or extends a store of its own. What changes overall
is how much you run at once, how you say which ground you want, and what the output store looks
like when it lands.

If you are choosing: use the single-area path unless you actually need global coverage. It is
simpler, it runs on one machine, and it is more flexible about time periods.

## What is genuinely the same

**The same inference code runs in both.** The single-area flow and the global campaign both call
the same `run_inference` function, on the same model checkpoint, producing the same
128-dimensional embedding per pixel. There is no "global model" and no "small model".

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

The shared code is run with **different settings**, and two of them decide which pixels get an
embedding at all. The same area and the same year can therefore come out differently depending on
which path produced it:

| setting | library default (single area) | the global campaign |
|---|---|---|
| `allow_s2_only` | `false` — a pixel with no radar observation produces nothing | **`true`** — optical-only pixels are embedded |
| `optical_min_obs` | unset — no minimum | **15** — a pixel with fewer than fifteen clear optical observations in the year is left empty |

Those are the only two quality rules in the system, and both were deliberate choices for a global
run: about a fifth of the land has no radar for 2022–24, so without `allow_s2_only` those pixels
would be holes; and fifteen observations is the line below which an embedding was judged not
trustworthy.

**What this means in practice.** If you compare your own single-area output against the published
global store and the pixels disagree, check these two settings before looking for a bug. If you
want your own run to match the published dataset, set both to the campaign's values. And because
they are per-run settings, the global store stamps `optical_min_obs` when it is seeded, so a
consumer can read which line a cell was actually measured against rather than assuming.

## What actually differs

| | one area | the whole world |
|---|---|---|
| what you name | an area of interest you supply | UTM zones and years from a fixed list* |
| how you say where | a boolean grid you make, at pixel resolution | a prepared coverage store, at 2048-pixel tile resolution |
| time period | **any 12 months**, ending in the month you choose | **calendar years only**, January to December |
| output | one store per area, one entry per window | one store for the world, one entry per zone per year |
| scale | one machine, one GPU or a few | dozens of machines, hundreds of GPUs, days of running |
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
> will not get an error, you will get plausible-looking nothing. The authority is each zone
> group's `years_complete` attribute, which lists only the years that actually landed:
>
> ```python
> import xarray as xr
> zone = xr.open_zarr("s3://tessera-embeddings/v1.1/dclimate.icechunk/", group="33N")
> print(zone.attrs["years_complete"])      # the years you can trust
> ```
>
> A small number of cells are legitimately unfilled — mostly tiny or remote land in the earliest
> years, where the satellite archive holds nothing to work from — so an absent year is usually a
> fact about the imagery rather than a gap waiting to be closed.

**Run the global campaign yourself only if** you need coverage or years the published store does
not have, and you have the infrastructure for it. It is a large, long, expensive job with its own
operational machinery; [`context_docs/campaign/campaign-plan.md`](../context_docs/campaign/campaign-plan.md)
describes what that involves and
[`context_docs/campaign/campaign-cost-model.md`](../context_docs/campaign/campaign-cost-model.md)
records what the last one cost.

## Where to go next

- [`quickstart.md`](quickstart.md) — the single-area path, end to end, on a laptop
- [`configuration.md`](configuration.md) — every setting, including the mask and window parameters
- [top-level README](../README.md#the-global-embeddings-store) — how the global store is laid out
- [`context_docs/decisions/008-global-store-architecture.md`](../context_docs/decisions/008-global-store-architecture.md)
  — why the global store is shaped the way it is
- [`context_docs/decisions/010-landmask-registry-coverage.md`](../context_docs/decisions/010-landmask-registry-coverage.md)
  — why the global coverage store is tile-granular
