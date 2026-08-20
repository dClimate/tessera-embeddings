# Tessera v2 on AWS-sourced input: how much does the MPC/AWS split cost us?

Companion to `v2_large_rollout_2026_07.md` (what the v2 Large run measured) and
`v2_large_readback_gate_2026_07.md` (how a model swap is validated). **Both of those
live on the `feature/v2-large-model` branch, not on this one** — the links are left
unhyperlinked deliberately so they do not read as broken here. This doc answers a
different question: our ingest reads
AWS sources, and Cambridge confirms v2 was normalised on Microsoft Planetary
Computer output. **How much does that mismatch cost, and what should we do about
it?**

**Headline: the provider gap is real but it is roughly a fifth to an eighth of the
version gap we have already absorbed correctly. It is not a merge blocker. The
expensive risk is not accuracy — it is being asked to re-run a global campaign if
an AWS-normalised v2 lands later.**

---

## The question, and the answer

Our ingest reads AWS-hosted Sentinel-2 while Cambridge normalised Tessera v2 against Microsoft
Planetary Computer. **Does the provider difference cost us anything?**

**No, and the version gap is the one that matters.** Measured against real ingested data, the
provider difference is far smaller than the difference between processing baselines, and none of the
biases found is a normalisation problem. A model trained on one provider's L2A and run on another's
is inside the noise the baselines already introduce.

## Decisions

- **Feed v2 our own AWS-sourced input unchanged.** No provider-specific normalisation layer.
- **The version gap is what to watch**, not the provider: a baseline change moves pixel values in
  ways a provider change does not.
- **This does not gate the v1.1 campaign**, which is what is running.

The companion documents — what the v2 Large run measured, and how a model swap is validated — live
on the `feature/v2-large-model` branch, deliberately unhyperlinked here so they do not read as broken
links on this one. The measurement detail behind the answer above is in git history.
## 7. Sources

- TESSERA v2 preprint — https://arxiv.org/abs/2607.03949
- v2 Large model card — https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L
- Upstream code and README — https://github.com/ucam-eo/tessera (branch `master`;
  `tessera_preprocessing/`, `tessera_infer_v2/`,
  `tessera_infer_QAT/src/datasets/v1_1_norm_stats.py`)
- OPERA RTC-S1 — https://registry.opendata.aws/nasa-operal2rtc-s1v1/
- MPC Sentinel-1 RTC — https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
- MPC Sentinel-1 data-gap report — https://github.com/microsoft/PlanetaryComputer/issues/472
