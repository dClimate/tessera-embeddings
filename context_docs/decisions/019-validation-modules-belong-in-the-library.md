# 019 — The validation instrument belongs in this library, and moves after the campaign

**Status:** Accepted, not yet executed. Ruled 2026-08-13 by the repo owner; deferred until after
the campaign, also by ruling.

## Context

`yield_embeddings.domain.embedding_validation` and its `_rules` sibling audit whether a published
zone-year of the global store is correct — the coverage cross-tabulation, the sampled overview, the
shard-boundary seam test, and the per-dimension distributions. Roughly 4,600 lines including tests.

They live in Arbol's private repository, and that is placement by adjacency rather than by decision:
they were born 2026-08-11 in the same commit as the flow that calls them, which legitimately lives
there.

## Decision

**They move to this repository.** The written boundary says so from both sides. The private repo's
README defines it as *"a thin production layer over the OSS `tessera_embeddings` library"* — CDK
infrastructure, the closed coarsening flow, Prefect deployments, the Packer AMI, customer ROI
resolution, operational tooling — with `domain/` for *"closed-source domain functions (coarsen, …)"*.
This repo's `architecture_tests/allowlist.py` says the same from the other side, framing closed-source
repos as downstream consumers that extend the OSS rules.

Validation is none of those things. It audits the output of the library's own assembly step, which
the README puts in the library's column, and the imports agree: four *tessera* modules and, from the
private package, nothing but its own rules file. **The methodology is not being treated as
commercial know-how.**

**The move happens AFTER the campaign.** An earlier reading wanted it first, so that every consumer
of the depth constant would sit under one CI. That benefit is real and available far more cheaply;
the cost of the move is not — 4,600 lines across a repository boundary, a new optional Pillow
dependency, and the modules the campaign's own monitoring depends on, weeks before a deadline.

## What moves, and what it costs

| file | lines | coupling |
|---|---:|---|
| `embedding_validation_rules.py` | 1,201 | **none** — `dataclasses`, `math`, `numpy` only |
| `embedding_validation.py` | 2,075 | four *tessera* modules; privately, only its own rules file |
| `yield-embeddings/tests/unit/embedding_validation/` | 1,303 | paired |

**Essentially nothing to untangle.** The rules module is pure logic and the audit module's sole
private dependency travels with it. It violates none of the enforced architecture rules — the
forbidden set is prefect, boto3, botocore, `tessera_embeddings.profiling` and three call names, while
this uses PIL, fsspec, zarr, numpy and asyncio.

**Pillow becomes a new optional extra** (`validation`), for 29 references in a bounded ~300-line
rendering block. Add the extra rather than splitting the module; the repo already carries four and
splitting to dodge one optional dependency is the worse shape.

**Stays behind:** the CLI drivers (`validate_cell.py`, `campaign_health.py`) and
`test_campaign_health_detectors.py`, which need `config.buckets` — deployment identity, which by the
boundary rule does not move.

## The rules are defaults, not laws — already done, carry it through

Someone running a different campaign — different sensor mix, region, or quality bar — must be able to
adjust thresholds without forking, and that was almost already true: every numeric threshold is a
keyword parameter defaulting to the module constant.

The exception was `blocking()`, which took *which findings fail a cell* from the module constant, so
it was the one policy an alternative campaign could not change without editing the file. **Fixed
2026-08-13 in the private repo**: it takes `blocking_slugs` like its five neighbours, the policy is
carried on `CellAudit` and accepted by `audit_cell()`, so the verdict, the report and the campaign's
failure raise all read one policy rather than the leaf offering a knob nobody could reach. **Carry
that through the move; do not redo it.**

It deliberately did *not* go on `CheckOptions`: that object is the sampling budget — how much of the
cell to read — and what counts as failure is a different question.

**`OPTICAL_MIN_OBS` is deliberately NOT tunable in that sense.** It is campaign-wide, stamped into
the store root as write-once identity, and enforced there (ADR-018). A downstream campaign sets it
for its own store; it is not a per-validation-run argument.

## Consequences

- Two *prose* statements in the private validator are wrong under the current rule and the rename
  surfaces them as import errors rather than silent drift: one defines a metric as embedded pixels
  below the line, which is **empty by construction** once nothing below the line is embedded; the
  other prints that being under the line is "a property of the input, not a fault", which is now
  vacuous and contradicts the policy. Both are in the reconciliation list.
- Line numbers in a handoff document are hints, not addresses — one of those two had already moved by
  the time it was read.

## Related

- [ADR 018](018-refuse-pixels-below-minimum-optical-depth.md) — the rule these two statements
  contradict
- [`../inference/minimum-optical-depth.md`](../inference/minimum-optical-depth.md)
