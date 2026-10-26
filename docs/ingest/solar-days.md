# Solar days: which day an image belongs to

A mosaic is indexed by local **solar day**: the calendar day at the place the image was taken. A
catalogue speaks UTC. Where a zone's solar offset crosses UTC midnight, in the far-eastern and
far-western zones, the two disagree.

This document is for anyone touching a date anywhere in the ingest (a query bound, a dedupe key, a
store coordinate), or asking why a day near the antimeridian came out short. It covers the one rule
that keeps the conventions apart (`solar_days.py`), how a query is padded so a day straddling a
chunk boundary is written whole, how OPERA's per-burst timestamps become one slice, and which day a
slice is labelled with.

Three things about a date have to be decided before a query is sent, and they are one subject:
which day an acquisition belongs to, which day a group of them is labelled with, and which day
range a catalogue is asked for. Getting any of them from a different convention than the others
loses imagery silently.

## The rule

**One rule, and everything else follows: the solar offset is applied exactly once, by
`normalize_to_solar_day`, at the catalogue chokepoint. After that an item's `datetime` IS
its solar day (at noon UTC), and every date derivation downstream is a plain
`strftime("%Y-%m-%d")` with no offset.**

Applied at more than one site the conventions drift: keyed on the UTC date while the loader
groups by solar day, a group straddling UTC midnight is not sorted as intended and half its
baseline entries never match.

## Every date in the pipeline, and its convention

The point of one chokepoint is that this table has no exceptions:

| where a date lives | form | convention |
|---|---|---|
| catalogue item, before the chokepoint | real acquisition instant | UTC — the only place a raw timestamp exists |
| catalogue item, after `normalize_to_solar_day` | **noon UTC of its solar day** | solar |
| loaded array coordinate (`odc.stac.load` output) | inherits the item stamp | solar (noon) |
| written store coordinate | `np.datetime64(solar_day)` | solar (midnight) |
| `existing_dates`, `written_dates`, baseline keys, `assessed_window` | `YYYY-MM-DD` strings | solar |
| chunk `own_start` / `own_end` | `YYYY-MM-DD` strings | solar |
| chunk `query_start` / `query_end` | `YYYY-MM-DD` strings | **UTC** — the only thing a catalogue understands |

The two timestamp forms differ (noon in flight, midnight in the store) and never meet as
numbers: everything crossing that boundary compares `YYYY-MM-DD` strings. The one deliberately
UTC row is the query bound, because a catalogue has no other vocabulary — which is why
ownership, not the query bound, decides what gets written. Noon rather than midnight is what
makes the stamp read as the solar day both directly and after `odc.stac.load` groups on it: noon
leaves half a day of margin, and no offset the grid produces (±11 h nearest the antimeridian)
crosses midnight.

## What enforces it

Three things enforce this rather than describing it:

- `solar_day_of` **raises** on an item that is not stamped noon UTC. A raw item's UTC date
  is a plausible-looking wrong answer — right at central longitudes, wrong only where the
  offset crosses midnight — so a path that skips the chokepoint would look correct
  everywhere anyone checks interactively.
- An **architecture rule** (`solar-offset-applied-only-in-solar-days`) fails CI if
  `solar_day_offset_seconds` is called outside `solar_days.py`. One application is the
  invariant; a second is a bug in the opposite direction.
- Every consumption point **re-normalises defensively** rather than trusting call order, because
  every supplier (`query_fn`, `item_provider_fn`) is injectable and `normalize_to_solar_day` is
  idempotent.

## A chunk owns solar days and queries UTC dates

A catalogue query is bounded in **UTC**; an ingest window and every chunk of it is a range of
**solar** days. Where a zone's offset crosses UTC midnight the two do not line up, and both ways
of ignoring that have been in this codebase: querying the chunk's own range and writing what
comes back splits a straddling solar day, so the earlier chunk writes it from its half, the later
chunk's half is dropped as an already-written date, and the day lands looking complete while
missing acquisitions; padding the query but clamping the pad to the window loses the first and
last solar day of a zone-year, since the padding vanishes at the window's own edges. Both are
silent — `assessed_window` still covers the days and the coverage gate still passes.

A chunk **owns** a range of solar days and **queries** a wider range of UTC dates:

```text
                 own:            2024-01-31 .............. 2024-02-29
                 query:   2024-01-30 ........................... 2024-03-01
                          └─ pad ─┘                              └─ pad ─┘

  a solar day landing on the cut is drawn from BOTH sides by the batch that owns it,
  and is owned by exactly one batch, so it is written once and written whole
```

Owned ranges tile the window exactly, so nothing is processed twice and nothing outside the
window is written — **ownership is what guarantees that, never the query bound**, which cannot
see solar days at all. The pads are deliberately *not* clamped to the window, because a solar day
owned at its very edge still draws on the UTC day beyond it. One day of padding is always enough:
the offset is a whole number of hours in `[-12, +12]`, so solar day `D` lies inside UTC
`[D-1, D+1]`.

`owned_items` is applied **between the query and the loader**, never after loading. Filtered
there, the loader builds a group only for an owned day and that group holds every image of it.
Filtered afterwards, a straddling day has already been split into two partial groups and what is
needed to rejoin them is gone.

Three chunkings share it today, and a new provider adds a fourth by writing a span producer and
nothing else:

| producer | used by | chunk |
|---|---|---|
| `month_ranges` | S2, streaming (default) | one calendar month, to bound item retention |
| `fixed_day_ranges` | S1 | `batch_days` solar days, to bound Dask graph size |
| `whole_window_range` | S2, `stream_stac_monthly=False` | the window in one query |

## The offset arithmetic must match the loader's

This rests on our offset arithmetic agreeing exactly with the loader's — both truncate
longitude over fifteen to whole hours. If they diverged an image could be filtered out as another
chunk's while the loader would have grouped it into this one, dropping it from the run entirely.
`solar_day_offset_seconds` is the single definition, and it is why `solar_grouping_longitude`
prefers the geobox centroid over a bbox midpoint, why `group_items_by_date` takes a
`mid_longitude`, and why the pre-sort uses the same key: the sort carries the
[fusion order](../../src/tessera_embeddings/ingest/README.md#the-fusion-order-decides-which-source-wins-a-pixel)
(clearest tile FIRST within a group), so sorting on a different notion of "day" than the grouping
would silently let a cloudier pixel win. Central longitudes image far from UTC midnight and are
unaffected, which kept it latent.

## Dates already in the store

`_filter_existing_dates` drops items whose dates are already in the store before the loader runs,
and it must be keyed the way the store was written. Both sensors load with
`groupby="solar_day"`, so their time axes hold solar days, and an acquisition's UTC date is not
its solar day where the offset crosses midnight. Callers grouping by solar day pass
`mid_longitude` down through `ingest_tile` / `query_stac_items`; matching UTC dates against a
solar-day set would filter only the half of a committed group on the near side of midnight, and
the surviving half would reload, regroup onto the day already present, and be written twice.

## Fusing OPERA's per-burst timestamps

A single MGRS tile bbox query returns ~10 burst granules per date, each with a slightly
different sub-second UTC timestamp (reflecting actual acquisition time). If passed to
`odc.stac.load` as-is, each burst becomes a separate time step instead of being mosaicked
together.

`normalize_opera_timestamps` delegates to `solar_days.normalize_to_solar_day`, grouping bursts
by **solar day** and setting every timestamp in a group to noon UTC of that day. `odc.stac.load`
then treats them as concurrent acquisitions and mosaics them into a single time slice.

## Which day a slice is called, and why it is not an item's timestamp

A mosaic slice represents one **solar day**, and it is labelled with that day — taken from the
grouping key, not from the loaded dataset's own time coordinate.

`odc.stac.load` stamps each group from `group[0]`, tying the
label to whichever item the sort left first — which can disagree with the solar day wherever the
offset crosses UTC midnight, so two consecutive solar days collide on the time axis: the batched
write rejects them as not strictly increasing, the unbatched write rejects the second as a
duplicate slot.

Taking the day from the grouping key instead makes three things true by construction: labels are
unique per slice, monotonic across them (so the batched write needs no sorting), and stable
against the catalogue revising its cloud estimates. This only decides WHICH day — pixels, ordering
and which tile wins are untouched, and at mid longitudes the value is unchanged anyway.
