# Tessera v2 on AWS-sourced input: how much does the MPC/AWS split cost us?

Companion to [`v2_large_rollout_2026_07.md`](v2_large_rollout_2026_07.md) (what the
v2 Large run measured) and
[`v2_large_readback_gate_2026_07.md`](v2_large_readback_gate_2026_07.md) (how a
model swap is validated). This doc answers a different question: our ingest reads
AWS sources, and Cambridge confirms v2 was normalised on Microsoft Planetary
Computer output. **How much does that mismatch cost, and what should we do about
it?**

**Headline: the provider gap is real but it is roughly a fifth to an eighth of the
version gap we have already absorbed correctly. It is not a merge blocker. The
expensive risk is not accuracy — it is being asked to re-run a global campaign if
an AWS-normalised v2 lands later.**

---

## 1. What we feed the model

Production ingest reads two AWS Open Data sources, both hardcoded:

| stream | provider | collection | native res | our grid |
|---|---|---|---|---|
| Sentinel-2 | Earth Search (Element 84) | `sentinel-2-l2a` | 10 m / 20 m | 10 m |
| Sentinel-1 | NASA CMR / ASF | `OPERA_L2_RTC-S1_V1_1` | **30 m** | **10 m** |

The MPC alternative would be `sentinel-2-l2a` (same ESA Sen2Cor granules) and
`sentinel-1-rtc` (Catalyst RTC from GRD, PlanetDEM, **10 m native**).

Our ingest is a port of upstream's own `--data_source aws` mode in
`tessera_preprocessing/s1_fast_processor.py` and `s2_fast_processor.py`: identical
`(20·log10(amplitude) + 50) × 200` SAR scaling, identical invalid-SCL set
`{0,1,2,3,8,9}`, identical non-wavelength band order, bilinear for reflectance and
nearest for SCL. We are not off-piste — we are running upstream's other supported
mode.

## 2. Where upstream actually stands

Cambridge's private note — *"it's only the v1.1 model that's actually fine-tuned
for AWS; v2 is currently MPC-only"* — is accurate and worth having. It is also
narrower than it sounds:

- The [v2 preprint](https://arxiv.org/abs/2607.03949) documents **no** data source
  and has no preprocessing appendix. The v2 Large model card names products
  ("Sentinel-2 L2A", "Sentinel-1 RTC") but no catalog.
- The v2 inference code contains **zero** occurrences of `data_source`, `mpc`,
  `aws`, `opera` or `planetary`. One hard-coded stat set. By contrast v1.1 ships a
  `data_source` selector that *raises* on a mismatch, plus two checkpoints
  (`tessera_v1_1_mpc_encoder.pt`, `tessera_v1_1_aws_encoder.pt`). v2 deliberately
  dropped that split.
- Upstream's README states: *"The new generation of embeddings will use OPERA from
  ASF DAAC"*, naming the two AWS registries we read. Their own intended v2
  deployment runs on our sources.

So "MPC-only" describes the **absence of an AWS fine-tune**, not a finding that
AWS input degrades v2. Nobody has published a measurement either way.

## 3. The provider gap versus the version gap — the main point

v1.1 measured the provider gap for us, by accident: it published band statistics
for *both* providers over the same preprocessing. That difference is a property of
the **data products**, not of the model, so it carries over to v2 unchanged.

Everything below is in units of the standard deviation the model divides by, which
is the only scale on which these are comparable.

```
             mean offset in sigma (log scale)
  0.01            0.1                          1.0
    |--------------|----------------------------|
    .   S2 provider gap (MPC->AWS)   0.028-0.054
    .    S1 provider gap (MPC->AWS)   0.049-0.122
    .                                      S2 VERSION gap (v1.1->v2)  0.139-0.413
    .                              S1 VERSION gap (v1.1->v2)  0.036-0.317
        `----- what "MPC-only" costs -----'
                                     `--- what we already handled ---'
```

| | Sentinel-2 | Sentinel-1 |
|---|---|---|
| **Provider gap**, MPC vs AWS — mean | 0.028–0.054 σ | 0.049–0.122 σ (0.20–0.52 dB) |
| **Provider gap** — scale ratio | 1.024–1.060 | 0.939–1.057 |
| **Version gap**, v1.1 vs v2 — mean | 0.139–0.413 σ | 0.036–0.317 σ |
| **Version gap** — scale ratio | 0.687–0.769 | 0.856–0.982 |

The version change is **five to eight times larger** than the provider change. PR
#98 wires v2's own constants (`_V2_NORM_STATS`, selected through
`config.inference.band_stats`), which is the part that would have been
catastrophic to get wrong — and is exactly the failure the readback gate's
structural checks exist to catch, since v2's non-affine output LayerNorm hides it
from every numeric check.

Two further reasons not to over-weight the v1.1 two-checkpoint precedent:

1. **The original split was partly a bug.** Upstream's `v1_1_norm_stats.py` records
   that the AWS row was recomputed on 2026-05-03 after a double-applied Sentinel-2
   `BOA_ADD_OFFSET` was fixed *and* the AWS checkpoint retrained on the corrected
   output. The figures above are the post-fix residual; the gap that originally
   justified two checkpoints was inflated by roughly 1000 DN.
2. **v2 is trained to tolerate input variation that v1.1 was not.** The teacher saw
   ~4.2 B d-pixels (students distilled over ~200 M) with random view lengths
   `L ~ U{8,16}` and whole-modality dropout — the latter explicitly so a pixel with
   no valid observations in one modality still works. v1.1 had far less of both.

## 4. Measured against our real ingested data

Sampled from `s3://arbol-tessera-inputs-dev/mosaics/iowa_epsg5070`, one 4000×4000
chunk, 16 dates across a 12-month window: **107 M valid Sentinel-2 pixels** and
**122 M Sentinel-1 pixels**. Comparing observed band centres against each
published stat set:

| stat set the model would assume | aggregate S2 offset | RMS over 10 bands |
|---|---|---|
| v1.1 MPC | −0.343 σ | 0.486 σ |
| v1.1 AWS | −0.379 σ | 0.509 σ |
| **v2 (MPC-derived)** | **−0.088 σ** | **0.404 σ** |

Sentinel-1 ascending VV lands 0.05 σ from v2's assumed mean; VH is the worst at
−0.26 σ. So our AWS-sourced input sits *closer* to v2's assumed centre than to
either v1.1 set — including v1.1's AWS set, which is what we ship today.

**Caveat, and it is a real one:** this is one landscape, one year, seasonally
weighted toward bare soil, so the SWIR bands read high and the visible bands read
low for reasons that have nothing to do with the provider. It cannot discriminate
provenance. What it does establish is that there is no offset error, no unit error
and no dB-scale error — the failure modes that would actually be catastrophic.

## 5. The biases that deserve attention — none of them is normalisation

Ranked by expected impact on a global campaign:

1. **Sentinel-1 is 30 m, not 10 m.** OPERA RTC-S1 is 30 m native, bilinearly
   resampled onto our 10 m grid; MPC's Catalyst product is 10 m native. Pooled band
   statistics are *completely blind* to this — every number in §3 and §4 misses it —
   yet it is the largest genuine difference. The fused embedding is 10 m optical over
   3×-smoothed radar. Tasks leaning on SAR texture (field boundaries, narrow
   features, structure) lose real information.
2. **Different terrain-correction lineage.** OPERA derives gamma-naught from SLC via
   ISCE3 against Copernicus GLO-30; MPC derives it from GRD via Catalyst against
   PlanetDEM. Residual flattening error correlates with **slope and aspect**, making
   this a geographically structured bias concentrated in high relief — invisible in
   Iowa, potentially material across mountainous UTM zones.
3. **These two compound in the worst place.** The v2 paper's own stated limitation is
   that its benchmarks come from well-studied regions, with label density
   concentrated in Europe and North America. In the persistently cloudy tropics the
   SAR stream carries most of the signal — which is exactly where our SAR is 3×
   coarser *and* where the model is least validated. That interaction threatens
   global results more than anything in §3.
4. **The cross-polarisation ratio shifts.** VV−VH, which carries vegetation
   structure, moves 0.5–0.9 dB between products because they disagree more in VH's
   dark tail (thermal-noise-floor handling) than in VV. Small, but it is the
   informative axis.

## 6. Decisions

**Merge PR #98.** The provider gap is second-order to the version change already
handled correctly, and upstream's own stated v2 deployment plan uses our sources.

**Switching to MPC is off the table** and was never the cheap fix it sounds like.
It would address only the Sentinel-1 resolution point, and it costs: rewiring the
ingest away from the OPERA/CMR path, an account-and-SAS-token access gate, hard
throttling on MPC's side, an open report of RTC ingestion gaps as recently as
February 2026, and cross-cloud egress from Azure to AWS compute for a global 10 m
SAR pull. OPERA and the Sentinel-2 COGs are AWS Open Data sitting next to the
cluster.

**Ask Cambridge one question: is an AWS-normalised v2 planned, and when?** The
operative word in their note is *"currently"*. This is a sequencing question, not
an accuracy question — a 0.05 σ input bias is cheap, and re-running a global
campaign after the fact is not (inference dominates campaign cost; see
[`campaign-cost-model.md`](campaign-cost-model.md)). If an AWS fine-tune is near,
that changes when we run, not whether we merge. Worth noting there may be a short
path: co-author Clement Atzberger is affiliated with dClimate Labs.

**Validate outside Iowa before committing the campaign.** Iowa is flat, temperate,
lightly clouded and label-rich — it cannot see risks 1, 2 or 3. Two ROIs are
prepared under `temp/rois/` for this, each ~155 km tall, inside a single UTM zone,
and each deliberately containing flat ground as an in-scene control:

| ROI | extent | area | 10 m pixels | zone | probes |
|---|---|---|---|---|---|
| `alps_highrelief` | 7.20–9.80 E, 45.60–47.00 N | 31,151 km² | 311 M | 32 N | residual topographic signal vs OPERA's local-incidence-angle layer, and whether v2 carries less of it than v1.1; 200 m to 4634 m, all aspects, Po-plain control |
| `rondonia_humidtropical` | 63.40–61.30 W, 11.60–10.20 S | 35,532 km² | 355 M | 20 S | SAR-dominant regime under persistent cloud; forest / logged / pasture / soy gradient; MapBiomas, PRODES, TerraClass labels |

Each is 21–24% of the state of Iowa. **Note the scope limit on the Alps ROI:** the
OPERA-versus-MPC terrain-flattening difference cannot be measured from one product,
so without an MPC ingest — ruled out above — that ROI tests whether topographic
residual survives into the embedding at all, not how the two providers differ.

Sizing is adequate for every test except the label probe, where the effective sample
size is the number of spatially blocked folds (70–77 at 20 km) rather than the pixel
count, and which resolves differences of roughly 0.6 to 1.3 balanced-accuracy points
— comfortably below v2's claimed 3.5-point margin over v1.1. Power analysis and
usage in [`temp/rois/README.md`](../../temp/rois/README.md).

Also note the existing Iowa validation stores are **ascending-orbit only**, while
the global campaign defaults to `s1_orbit="both"` — so they under-represent the SAR
input the campaign will actually produce.

## 7. Sources

- TESSERA v2 preprint — https://arxiv.org/abs/2607.03949
- v2 Large model card — https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L
- Upstream code and README — https://github.com/ucam-eo/tessera (branch `master`;
  `tessera_preprocessing/`, `tessera_infer_v2/`,
  `tessera_infer_QAT/src/datasets/v1_1_norm_stats.py`)
- OPERA RTC-S1 — https://registry.opendata.aws/nasa-operal2rtc-s1v1/
- MPC Sentinel-1 RTC — https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
- MPC Sentinel-1 data-gap report — https://github.com/microsoft/PlanetaryComputer/issues/472
