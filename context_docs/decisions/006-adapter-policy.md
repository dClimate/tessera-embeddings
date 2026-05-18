# 006 — Community adapter policy

**Status:** Accepted (v0.1.0)

## Context

Given [001](001-thin-prefect-wrapping.md) (Prefect is the
reference, not the requirement) and
[004](004-duck-typed-providers.md) (providers are sibling
directories, not abstractions), the project is open to
community-contributed orchestrator adapters (Dagster, Airflow,
Flyte, Argo) and provider implementations (GCP, Azure, k8s, on-prem).

But every OSS library that accepted such contributions liberally
ended up with an "adapter graveyard" — community plugins that
worked at submission, broke 18 months later, and stuck around
because deletion is contentious.

## Decision

Accept community adapters with **firm submission requirements** and
a **named-maintainer commitment**. Unmaintained adapters move to
`archived/` rather than being deleted.

Submission requirements:

1. **Maintainer named** in the adapter's own README.
2. **Parity test** in CI against `runners/plain.py` for the bundled
   quickstart ROI. The adapter's flow must produce
   byte-equivalent output (modulo timestamps + run IDs).
3. **Parity doc** listing which features map cleanly, which have
   idiomatic equivalents, and which have no analog.
4. **Explicit "community-maintained, not core-supported" labelling**
   in the adapter's README and module docstring.

Lifecycle:

* Core reviews submissions for fit and correctness, not
  feature-by-feature design choices within the adapter.
* If the named maintainer goes silent and CI breaks, an issue is
  opened. After a reasonable period (60 days, no response), the
  adapter is moved to `archived/<adapter_name>/` with a deprecation
  notice.
* `archived/` adapters are clearly labelled but not deleted —
  preserves knowledge for future contributors.

## Rejected alternatives

**No community adapters:** Closed-doors policy. Forces every
non-Prefect user to maintain a fork forever. Prevents the
ecosystem benefit OSS is supposed to provide.

**Open submissions, no maintenance contract:** Adapter graveyard.
Within ~2 years, half the adapters break, downstream users file
issues against core, core can't fix them.

**Core-maintained adapters:** We'd own ~5x more code. Each new
orchestrator we accept means we have to track its API changes, its
idiom shifts, its bugs. Not feasible at our team size; would also
slow down core development.

## Consequences

- **Pro:** community adapters are real (parity tests prove
  correctness) and aren't a maintenance liability for core.
- **Pro:** the `archived/` lifecycle is clear and respectful —
  adapter contributions don't disappear, they just get correctly
  labelled.
- **Pro:** the parity test is a meaningful contract; downstream
  users of an adapter can trust it produces the same outputs as the
  reference.
- **Con:** the bar to submission is non-trivial — writing a parity
  test requires a working LocalCluster fixture and understanding
  the parity helpers. Documented in
  [`tests/parity/adapter_template/`](../../tests/parity/adapter_template/);
  still a barrier for casual contributions.
- **Con:** if a previously-accepted adapter is moved to
  `archived/`, downstream users have a discoverability hit. We'll
  publish a release note when this happens.

## Related

- README.md "Contributing" section — user-facing summary.
- [`tests/parity/adapter_template/`](../../tests/parity/adapter_template/) —
  the contract in code form.
- [`design/orchestration_infra_leakage_audit.md`](../design/orchestration_infra_leakage_audit.md) —
  the audit informing what counts as "leakage" vs "expected" in an
  adapter.
