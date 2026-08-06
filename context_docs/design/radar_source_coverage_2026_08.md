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
| one orbit only | 6 | 89 | 0.37% |
| radar exists, wrong polarisation | 3 | 242 | 1.00% |
| no radar at all | 10 | 51 | 0.21% |

Counted with the **twelve-granule floor applied to the verdict** rather than as a footnote: an
orbit under roughly one granule a month over a zone's live tiles cannot support a year, and
labelling it an orbit invites a reader to explain provider behaviour that never happened. 24N's
entire 2025 ascending total is one granule, which is why it sits in the wrong-polarisation row
rather than the single-orbit one.

Across all nine campaign years, `both_orbits` holds between **95.8% and 98.6%** of live tiles, so
this is a property of the whole span and not of the deliverable year.

**A year-on-year comparison is NOT offered here, and the reason is the point.** The first attempt
found fourteen zones losing coverage; almost all of it was the survey's own method changing between
years (see below). A method-consistent re-measurement of the affected zones was in flight when this
was written. Do not read a trend out of this table until that lands.

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

## The nine-year picture, with the method artefact accounted for exactly

`both_orbits` holds **95.8% to 98.6%** of live tiles in every year from 2017 to 2025, so the
headline is a property of the whole span. Radar does not appear partway through the campaign
window: all four verdict classes already exist in 2017.

**How much of the year-on-year comparison the method artefact actually spoils: 8 zones of 112,
holding 770 of 24,202 live tiles — 3.2%.** Those eight had some years screened and others probed.
The other 104 zones were answered by ONE method across all nine years, so their comparison is
valid as it stands. The eight are 22S, 23S, 24S, 02N, 57S, 25S, 26S and 31S, and a
method-consistent re-probe of them was in flight when this was written.

### The real reversals — measured per live tile across all nine years

**Correction, and this is the second one on this question.** An earlier version of this section said
the reversals were "mostly the method" and named 05S and 06S as the only robust case. **That was
wrong.** Re-probing the ten disputed zones per live tile for all nine years — one method in every
cell — shows the reversals are real, and the two largest are far bigger than the pair I highlighted.

| zone | orbit | granules per year, 2017 → 2025 |
|---|---|---|
| **57S** (35 tiles) | ascending | 11544, 10846, 11563, 10390, 10441, **0, 0, 0**, 6885 |
| **02N** (44 tiles) | ascending | 9019, 6997, 8905, 9987, 10388, **0, 0, 0**, 11025 |
| 02N | descending | 25692, 26937, 27676, 27511, 27358, 24433, 26893, 25620, 25512 |
| 05S (10 tiles) | descending | 252, 270, 279, 270, 261, **0**, 786, 1552, 1799 |
| 06S (24 tiles) | descending | 1176, 1260, 1302, 1260, 1218, **0**, 723, 1546, 2403 |
| 31S (3 tiles) | descending | 546, 1131, 503, **0 from 2020 onward** |

**Two unrelated zones lose an entire orbit for exactly 2022, 2023 and 2024 and regain it in 2025.**
57S is the south-west Pacific and 02N the mid-Pacific; 02N's descending orbit continues unbroken at
about 26,000 granules a year straight through the gap, so this is not an outage of the sensor or of
the zone. The extent is identical in both.

**What the method artefact actually corrupted was the LABELS, not the reversals.** The screen called
57S `both_orbits` for 2017–2021; the probe calls it `single_orbit`. Both agree radar was present, and
both agree 2022–2024 is zero. So the collapse was always there — my first pass over-stated its
before-state, and my correction then over-corrected into denying it. Stating the distinction because
it is the reusable part: **a method inconsistency can misdescribe a change without inventing it.**

**The gap is per ORBIT and not specific to VV, which is what makes it look like a production
backlog rather than anything about the polarisation we require.** Zone 43S's *cross-pol* descending
granules run 463, 3921, 2174, 2012, 1087, **0, 0, 0**, 1297 — the same three-year hole, in HH, while
its ascending HH continues at about 3,700 a year throughout. So in every case one orbit of one
polarisation over one region goes missing for a span while the other orbit carries on.

A small methodological note worth keeping: a first check for this summed ascending and descending
before comparing, and found nothing. **Aggregating over orbits hides a per-orbit gap** — the zone
total never reaches zero because the surviving orbit holds it up. Compare per orbit.

The remaining movement is orbit SWITCHING rather than loss. 42S and 43S publish ascending early (56
and 50 granules in 2019) and descending later (2,913 and 372 by 2025), never overlapping. And 31S is
the one degradation that has NOT recovered: both orbits for 2017–2019, then descending at zero for
six straight years to 2025.

**Operationally, this is the part to carry:** a cell in 57S or 02N for **2022, 2023 or 2024** gets
materially less radar than the same zone in 2021 or 2025. That is not a defect in our pipeline to
chase — it is what the archive holds — and it explains why the 57S-2022 fill resolved to radar-free
and recorded 100% of its pixels as having no radar. It behaved correctly on data that genuinely was
not there.

Three zones — 22S, 23S and 24S, 661 live tiles — still hold a mix of screened and probed years and
were not re-probed. Their year-on-year behaviour is unmeasured; each of their cells says so.

### A limitation of the twelve-granule floor, worth stating rather than fixing blind

The floor is per ZONE, and a zone's tile count varies from 1 to 556. Zone 23N reads `single_orbit`
for 2019 alone on **48** granules spread over **6 of its 116 live tiles** — above the floor, and
still nothing like an orbit. The defensible unit is granules per live tile, and the sweep already
records how many live tiles each orbit reaches, so the fix is available. It is not applied here
because changing the floor changes every verdict, and that wants doing once, deliberately, rather
than as a side effect of writing this section.

## The finding that actually changes something

**Three quarters of the "radar-free" land is radar we decline.** Zones 23N and 24N — Greenland
and Arctic Canada, 208 live tiles — publish effectively no VV+VH and tens of thousands of HH/HV
granules. 24N's 2025 total is *one* ascending VV granule, which is why the twelve-granule floor
now sits in the verdict itself and places 24N with Greenland rather than with the single-orbit
zones.

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
usable radar is one. For 2025 that is 93×3 + 6×2 + 13×1 = 304 fleets against 336 if every cell
needed both — about 9% fewer. Real, but not a planning lever.

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
