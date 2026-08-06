# Radar source coverage across the campaign — measured 2026-08-06

What OPERA RTC-S1 actually publishes over the 112 land zones, how much of the product is
optical-only as a result, and why the largest part of that is a polarisation decision rather
than an absence of radar.

The durable artefacts live in S3, not here:
`s3://global-tessera-inputs-dev/source-coverage/radar_coverage.json` and the same key under
`s3://global-tessera-inputs/`, each beside a `RADAR_README.md` stating the method and caveats.
This document records how the measurement was taken and what it changed, so the next person
does not repeat the two ways I got it wrong first.

## What prompted it

Zone 30S filled as optical-only embeddings and every component behaved correctly on the way
there. Both radar legs reported `items_seen=0` for every thirty-day batch across two whole
years; no SAR store was created; `data_loading` resolved `s1_orbit='both'` down to `'none'`;
the assembly recorded 100% of embedded pixels as having no radar. The only thing missing was
anyone knowing in advance which zones behave this way, or what share of the product they are.

## The answer

For **2025**, the deliverable year, weighted by live MGRS tiles:

| verdict | zones | live tiles | share of land |
|---|---|---|---|
| both orbits | 93 | 23,820 | 98.42% |
| one orbit only | 7 | 181 | 0.75% |
| radar exists, wrong polarisation | 2 | 150 | 0.62% |
| no radar at all | 10 | 51 | 0.21% |

For **2022**, where most of the test programme's cells were filled: 96.91% both orbits, 1.59%
single orbit, 1.00% wrong polarisation, 0.50% no radar. The difference between the two years is
the provider expanding — 57S has no radar at all in 2022 and one orbit in 2025.

## Three things this measurement gets right that a naive version would not

**Weight by land, not by zone count.** Nineteen of 112 zones lack full radar in 2025 — 17% by
zone count, and about 1.6% by land. The radar-free zones are almost all ocean: ten of the
thirteen hold under ten live tiles, and 41S holds one. Reporting the zone share would overstate
the product impact by an order of magnitude.

**Ask the way the ingest asks.** Same collection, same provider, and both of the ingest's
server-side filters — orbit direction *and* the VV polarisation requirement. Without the
polarisation filter the numbers describe a catalogue nobody reads; with it, a zero here is the
same zero the ingest reports as `items_seen=0`.

**Probe per live tile, not per zone bounding box.** The zone-level query uses the union of the
zone's live tiles' bounding boxes, which is a *superset* of the land: a zero over it is
decisive, a hit is only a screen. **Zone 57S is the case that proves the distinction matters.**
The union screen found 558 descending granules; the per-tile probe found none over its 35 live
tiles; and the fill's own assembly recorded 100% radar-free. The probe agrees with what
actually happened and the screen does not. Every zone the screen did not settle was re-probed
per tile, which is about 6,000 queries and a few minutes.

**Two consequences of that asymmetry, both found on 2026-08-06 and both worth carrying.**

*The headline rests on screen verdicts.* The probe runs only where the screen fails, so every
`both_orbits` zone is screen-only by construction — and 57S shows a screen `both_orbits` can be
wrong. The exposure concentrates in zones whose land is scattered across a large box, since
contiguous land gives an accurate screen; that argues the tile-weighted error is small, but it is an
argument rather than a measurement. Bounding it needs a probe of a sample of screen-settled
both-orbits zones.

*A table whose cells used different methods is not a comparison.* Extending the survey to all nine
campaign years as nine separate invocations let a zone be screened in one year and probed in
another, and the difference between those years then measures the method. It produced fourteen
zones apparently LOSING coverage, with the reversals clustered at exactly the two year boundaries
where zones crossed from screened to probed — not at anything the provider did. The sweep now probes
a zone in every requested year if it is unsettled in any of them, and prints the per-year method mix
so the asymmetry cannot recur silently.

## The finding that actually changes something

**Three quarters of the "radar-free" land is radar we decline.** Zones 23N and 24N — Greenland
and Arctic Canada, 208 live tiles — publish effectively no VV+VH and tens of thousands of HH/HV
granules. 24N's 2025 total is *one* ascending VV granule, which is why the stored table carries
a caveat marking any orbit under twelve granules as too thin to build a year.

`opera_query._granule_to_item` already rejects those granules and its comment already names
Greenland, so the behaviour was known. What was not known is the price: the radar half of the
embedding, across all of Greenland and Arctic Canada, for a reason no operator would guess from
the symptom.

Declining them is probably correct — the model was trained on VV+VH, and feeding HH into a VV
channel hands it a different distribution — but that should be a decision with a stated cost
rather than a surprise. **If it is ever revisited, the question is a model question, not an
ingest question.**

Two claims not to make about it, both of which I would have made without checking:

- **It is not EW mode.** Those HH/HV granules report `BEAM_MODE=IW`, and an EW-mode query over
  the same region returns nothing. The code comment says this and says it was speculated
  wrongly once before; it is worth not making a third time.
- **A count is not coverage.** A count says granules exist whose footprint intersects a live
  tile's bounding box. Whether the footprint reaches the live pixels, and whether the year is
  evenly represented, is answered only by `s1_asc_obs_count` / `s1_desc_obs_count` on a filled
  zone group.

## What it implies for the campaign

**Fleet count.** A cell with both orbits is three concurrent fleets; single orbit is two; no
radar is one. For 2025 that is 93×3 + 7×2 + 12×1 = 305 fleets against 336 if every cell needed
both — about 9% fewer. Real, but not a planning lever.

**Product quality is where it lands instead.** Roughly 1.2% of the 2025 land will ship
optical-only embeddings, and it is not scattered: it is Greenland, Arctic Canada, and a set of
small ocean zones. The per-year radar coverage now recorded on each zone group
(`radar_coverage` in the run provenance) makes that legible per cell without consulting this
table, which is the right place for it.

**The radar-free path is not an edge case to be tolerated — it is the correct behaviour for
about a fifth of the zone list.** The per-cell orbit resolution and the `allow_s2_only` path are
load-bearing for the campaign, not defensive.

## Regenerating

The method is in the S3 `RADAR_README.md`. One operational note that cost time: CMR returns an
occasional **400** under twelve-way concurrency, and eight zones came back as errors on the
first pass. Every one resolved on a serial retry. **Do not accept a CMR error as a zero** — that
is a false radar-free verdict, and it is indistinguishable from a real one in the output.
