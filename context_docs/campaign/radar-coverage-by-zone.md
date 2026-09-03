# Which zones have no usable radar, and why

**What OPERA RTC-S1 publishes over the 112 land zones, measured 2026-08-06.** This survey answers a
narrower question than its first headline claimed, and the narrowing is the point of reading it.

> ## ⚠ READ THIS FIRST — what this instrument can and cannot see
>
> **Withdrawn: "both orbits cover 95.8–98.6% of campaign land in every year."** That figure is **per
> ZONE**, not per pixel: a zone counts as dual-orbit if each orbit has granules *anywhere* over its
> live tiles. The 2022–2024 radar loss is **sub-zonal** — interior Australia and much of Siberia,
> inside zones whose coastal tiles keep their radar — so this instrument cannot see it and reported
> those zones as fully covered.
>
> **The authoritative measurement already existed** and is area-weighted per pixel, in
> [`campaign-cost-model.md`](campaign-cost-model.md) §"The 2022–2024 radar gap": 100% of land covered
> in 2017–2021, **81% in 2022–2024** with S1A alone, 96% in 2025. **The cause was already known too:
> Sentinel-1B failed in December 2021**, and Sentinel-1C restored coverage during 2025. My "looks
> like a production backlog" is withdrawn — it is a satellite failure. Across the nine years **6.8%
> of pixel-years are optical-only**, and `allow_s2_only` is ON precisely so those pixels produce
> data.
>
> **What this survey IS still good for**, and why it is kept:
> - **A zone-level ZERO is decisive.** A zone with no granules over any live tile genuinely has none,
>   so the `radar_free_no_radar` and `radar_free_wrong_polarisation` verdicts stand.
> - **The polarisation finding is new and is not in the census**: zones 23N and 24N publish tens of
>   thousands of HH/HV granules and effectively no VV+VH, so they are radar-free because the ingest
>   declines cross-pol, not because radar is absent.
> - The per-orbit, per-year granule counts are sound as counts.
>
> **What it must never be used for again: any statement of the form "N% of campaign land has both
> orbits."** Use the per-pixel census. This is the same error as *a granule count is not coverage*,
> committed at a coarser granularity — a count over a zone is not coverage of that zone's land.

What OPERA RTC-S1 actually publishes over the 112 land zones, how much of the product is
optical-only as a result, and why the largest part of that is a polarisation decision rather
than an absence of radar.

The durable artefacts live in S3, not here, under `source-coverage/` in **both**
`global-tessera-inputs-dev` and `global-tessera-inputs`, which now hold an identical set:

| key | what it is |
|---|---|
| `README.md` | indexes both sensors' surveys and states the one way they differ |
| `radar_coverage.json` + `RADAR_README.md` | this survey, all nine campaign years |
| `optical_gaps.json` + `OPTICAL_README.md` | the Sentinel-2 equivalent |
| `zone_live_mgrs_tiles.json` | which MGRS tiles carry each zone's live land — the input both surveys key off |
| `mgrs_tile_years.json` | the raw optical tile-year census |

**Files at that level are the current authority; dated directories are snapshots.** The optical
artefacts previously lived only under a dated prefix, so a cross-reference to them from the radar
README was broken and a consumer following it would have found nothing.
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

## Why a naive version of this measurement is wrong

Three things, each of which reverses a zone's answer:

**Ask with the ingest's own filters, including VV+VH.** A zone that publishes only VV, or EW-mode
HH/HV, has granules and no *usable* radar — so a query without the polarisation filter reports
coverage the ingest will refuse. A zero here has to mean the ingest's `items_seen=0`, or it means
nothing.

**A zone's union bounding box is not its land.** Zone 57S's box holds hundreds of granules its live
tiles do not intersect. So a per-zone screen's ZERO is decisive and its HIT is only a screen; the
per-live-tile probe is what settles anything the screen leaves open.

**"No radar" and "radar in a polarisation we decline" are different findings** with different
remedies — one is an archive gap, the other a choice we made — and a measurement that conflates them
cannot inform either.

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

**Product quality is where it lands instead.** Roughly 1.2% of the 2025 land is optical-only *for
this survey's zone-level reason* — the per-pixel figure across all nine years is 6.8% (cost model §6)
— and it ships as optical-only embeddings, and it is not scattered: it is Greenland, Arctic Canada, and a set of
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
