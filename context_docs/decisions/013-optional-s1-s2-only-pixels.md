# 013 — Optional S1: S2-only pixels behind `allow_s2_only`

Status: accepted (2026-07-23). Follows a reviewer finding on PR #82's lineage;
builds on ADR-008 (global store), ADR-011 (campaign ingestion), ADR-012
(validated equivalence).

## Context

An external review found two Sentinel-1 availability gaps:

1. **The per-pixel S1 requirement.** `inference/dataset.py` gated every pixel
   on `s1_total_valid > 0`: a pixel with valid S2 but zero S1 observations was
   silently dropped — no embedding, store fill values. S1 coverage gaps are
   **sub-zone** (swath edges and holes in nearly every UTM zone; no zone is
   entirely S1-free), so real S2-covered pixels were lost everywhere, worst at
   high latitudes.
2. **VV-only granules** (VH missing — observed in 2017 Tanzania and SW Brazil
   OPERA data) were dropped whole at a single gate in
   `ingest/opera_query.py::_granule_to_item`, with one undifferentiated
   warning. Polar S1 is a separate case: EW-mode acquisitions carry **HH/HV**,
   which the band regex did not recognize at all — polar is the *zero*-S1
   scenario, not the partial one.

## What the model actually supports (verified)

- The encoder (`inference/models/ssl_model.py`, `modules.py`) has **no
  attention/padding mask and no count division** — it consumes any
  `(B, T≥1, 3)` S1 tensor. A zero-S1 pixel already flows through
  `sampling.resample_s1_bucket` as an **all-zeros slice in normalized space**
  (= per-band mean, a neutral constant) into the smallest bucket via
  `compute_bin_keys`' `np.clip(s1_valid, 1, None)`.
- **Upstream check (ucam-eo/tessera, tag v1.1,
  `tessera_infer_QAT/src/datasets/ssl_dataset_v1_1.py`):** upstream has **no
  valid-pixel gate at all** — it embeds every pixel, and its
  `_sample_s1_merged` returns `np.zeros((target, 3))` when a pixel has no S1.
  Our zero-count resample output is **bit-identical to that convention**. The
  per-pixel S1 requirement was a local addition of this pipeline; relaxing it
  restores upstream parity. (Upstream also exposes a whole-tile `use_s1`
  toggle we do not need.)

## Decision

**`allow_s2_only: bool = False`** on `InferenceConfig`, threaded from
`run_global_campaign` / `fill_zone_year` / `tessera_embeddings` / the plain
runner through `build_inference_config` to the dataset's valid-pixel gate:

- **Off (default):** exactly the historical behavior — S2-valid/S1-empty
  pixels are skipped (pinned by tests).
- **On:** the S1 term of the gate is dropped; S2-valid/S1-empty pixels embed
  with the upstream missing-S1 input (all-zeros normalized S1, smallest
  bucket). S1-informed pixels are **bit-identical** with the flag on or off
  (pinned by an end-to-end test through the real model).

**Explicit opt-in, not automatic:** zero S1 where S1 is expected can be a
transient data bug (ASF outage, auth failure, query regression), and
zone-year tags are write-once forever — an accidental silent downgrade would
permanently publish degraded embeddings. The operator states the intent.

**The flag is part of the staging fingerprint** (`_staging_run_id`): it
changes which pixels get embeddings, so a retry across a flipped flag starts a
fresh staging prefix rather than resuming mixed tiles. It is NOT part of the
ingest marker — ingestion is unaffected by it.

**The single-ROI path (`tessera_embeddings`) does not derive its run_id from a
fingerprint** — a fresh run gets a random uuid and a resume reuses an
operator-supplied `previous_run_id`. Since `run_inference()` skips already-
staged `.zarr`/`.skipped` artifacts by run_id alone, flipping `allow_s2_only`
on a resume would keep old skip markers (the very S2-only pixels the flag now
embeds) for staged chunks while recomputing the rest under the new gate, and
assembly would publish a mix. Fix: a fresh S2-only run's run_id carries an
`s2only-` prefix (`S2_ONLY_RUN_PREFIX`), so the mode that produced a run's
staged chunks is recorded in the staging namespace itself; a resume whose
prefix disagrees with the requested flag is refused (`_resolve_run_id`).
Default-mode runs keep the historical bare-uuid run_id, so the single-ROI path
is unchanged when the flag is off. Assembly-only resumes are exempt (they
re-publish staged tiles and never run the per-pixel gate).

**Public-API + reachability wiring.** `InferenceConfig` is a positional
dataclass documented as public API, so `allow_s2_only` is appended as the
*last* field — a mid-list insertion would silently rebind downstream
positional args. The flag is also forwarded by the master
`tessera_full_pipeline` to the embeddings deployment, so it is reachable from
the documented end-to-end path, not only the child flow. (Both surfaced in
review of PR #95.)

**Zone-level gates stay loud.** `ingest_zone_year`'s
`InsufficientCoverageError` (zero SAR stores produced), `resolve_s1_orbit`
(no SAR store at fill), the temporal coverage gate, and the SAR grid
validation are all unchanged. Since every zone has *some* S1, a zone with
none signals an ingest bug — the campaign halts instead of tagging an S2-only
year. (There is deliberately no `s1_orbit="none"` mode.)

**VV-only granules stay dropped** — with quantifiable logs. Analysis of the
encoding and training: a nodata VH (DN 0) normalizes to ≈ **−1.6σ**, inside
the real VH tail — undetectable as "missing" by a network with no mask
mechanism — and physically reads as *smooth surface/water* (≤ −50 dB
cross-pol). The VV/VH ratio is the vegetation-structure signal in dual-pol
SAR, so accepting VV-only dates would bias affected regions' embeddings
toward bare/water-like signatures *invisibly*: the any-band validity
convention would also count those dates in `s1_obs_count`, poisoning
provenance. With the per-pixel flag, dropping degrades **honestly** instead —
a pixel whose only S1 was VV-only becomes an S2-only pixel with
`s1_obs_count == 0`. The skip path now logs which polarizations each rejected
granule carried (`VV-only`, `HH/HV (EW-mode?)`, `no data links`) plus a
per-query summary count, so regional loss is measurable from logs.

## Provenance

No new store fields. The per-pixel record already exists and is exact:

> An embedded pixel (finite `scales`) with
> `s1_asc_obs_count + s1_desc_obs_count == 0` is an S2-only embedding.

Both obs-count arrays are written for every pixel in the GLOBAL and SINGLE
layouts, flag on or off.

## Quality caveat (REQUIRED follow-up before production use)

There is no evidence the v1.1 checkpoint was trained with S1
dropout/modality masking — the all-zeros input, while upstream-sanctioned, is
out-of-distribution as a *sequence*. S2-only embeddings are therefore **not
validated as comparable** to S1-informed ones. Before enabling
`allow_s2_only` in a production campaign, run a mask-S1 comparison study on a
dual-modality region (embed with and without S1; quantify drift and any
downstream-task impact). Until then the flag is for coverage-critical /
experimental work.

## Alternatives considered

- **Accept VV-only granules (VH=nodata), optionally behind a flag** —
  rejected: invisible directional corruption + provenance poisoning (above).
  Imputing VH at the per-orbit mean halves the bias but still fabricates the
  cross-pol ratio; not worth a third data mode.
- **Zone-level S2-only mode (`s1_orbit="none"`)** — rejected: geography makes
  it moot (no S1-free zones) and the loud zone gates are valuable bug
  detectors.
- **Automatic downgrade (mirror the both→one orbit fallback)** — rejected:
  write-once tags make accidental permanent degradation too costly.
- **HH/HV (EW-mode polar) ingestion** — out of scope; recognized in logs only.
