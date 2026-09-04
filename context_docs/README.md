# context_docs/

**The rationale behind the code, not documentation of it.** Reference documentation lives in
`docs/`. Read this if you are trying to understand *why* the code looks the way it does, or if you
are proposing a change that conflicts with one of these decisions.

Two kinds of thing live here. **Decision records** are one page each and answer "why is X the way it
is?". **Stage records** are long-form and answer "what did we measure, what did it cost, what did we
try that failed?" — one per pipeline stage, so a question about ingest has exactly one place to go.

## Layout

```
context_docs/
├── corrections-register.md        EVERY withdrawn figure, grouped by how it went wrong
├── test-suite-streamlining.md     this repo's own test suite: what has been done to it, and
│                                  what is left
│
├── decisions/                     Architecture Decision Records — 1 page each, append-only
│   ├── 001-thin-prefect-wrapping.md
│   ├── 002-shape-c-task-shells.md
│   ├── 003-tenacity-retries-not-prefect-native.md
│   ├── 004-duck-typed-providers.md
│   ├── 005-multi-lock-file-strategy.md
│   ├── 006-adapter-policy.md
│   ├── 007-icechunk-chunk-cache-disabled.md
│   ├── 008-global-store-architecture.md
│   ├── 009-native-cmr-granule-query.md
│   ├── 010-landmask-registry-coverage.md
│   ├── 011-campaign-zone-ingestion.md
│   ├── 012-validated-equivalence-for-inference-outputs.md
│   ├── 013-optional-s1-s2-only-pixels.md
│   ├── 014-architecture-rules-scan-the-package.md
│   ├── 015-regrouping-the-unit-tests.md
│   ├── 016-scripts-layout-and-shared-modules.md
│   ├── 017-no-antarctic-coverage.md
│   ├── 018-refuse-pixels-below-minimum-optical-depth.md
│   ├── 019-validation-modules-belong-in-the-library.md
│   ├── 020-boa-offset-applies-to-every-valid-dn.md
│   ├── 021-correct-the-boa-offset-per-image.md
│   ├── 022-resolve-the-roi-mask-credential-at-read-time.md
│   └── 023-the-single-path-end-to-end-is-the-quickstart-run.md
│
├── campaign/                      THE GLOBAL CAMPAIGN — start here
│   ├── campaign-plan.md                        <- START HERE. What runs, with what settings, in
│   │                                              what order, and what to do when it breaks.
│   │                                              §11 is the authority map over everything else
│   ├── campaign-cost-model.md                  every cost, rate, fleet size and GPU-hour, plus
│   │                                              how work balances across N clusters (§5b)
│   ├── campaign-validation-and-monitoring.md   how each published cell is checked, and how a
│   │                                              finding reaches a person
│   └── radar-coverage-by-zone.md               which zones publish no usable radar, and why
│                                                  three quarters of that is a polarisation choice
│
├── ingest/                        BUILDING THE MOSAICS
│   ├── ingest-performance.md                   what ingest costs, what made it faster, and what
│   │                                              limits it now — the graph budget, the catalogue
│   │                                              budget, live-tile cropping, region writes
│   ├── source-read-failures.md                 twelve ways a source read fails, the guard each
│   │                                              earned, and the retry budget they share
│   └── solar-day-fusion-order.md               which scene wins a contested pixel, and why the
│                                                  answer was inverted for the life of the code
│
├── inference/                     RUNNING THE ENCODER
│   ├── inference-on-gpus.md                    throughput, which cards may be rented, why the
│   │                                              batch is sized to the card, and what the
│   │                                              campaign path itself measured
│   ├── minimum-optical-depth.md                the one data-quality rule: refuse a pixel below
│   │                                              15 observations. Four measurement campaigns
│   └── gpu-fleet-launch-throttling.md          why several clusters asking EC2 for GPUs at once
│                                                  throttle each other, and the two bounds
│
└── storage/                       WRITING AND KEEPING THE RESULT
    ├── writing-to-the-global-store.md          assembly's fork pool, the session catch-up, the
    │                                              credential incident, why commits are ungated,
    │                                              and the registry published beside the store
    ├── staging-identity-and-resume.md          what a run id identifies, and what resumes
    └── icechunk-api-ledger.md                  signatures and gotchas the scale tests earned
```

## How to read these

**Decision records** have a fixed shape — Context, Decision, Rejected alternatives, Consequences —
so you can scan to the part you need. They are **append-only**: supersede one with a new record
rather than editing it in place.

**Stage records** are grouped by the pipeline stage they belong to, which is also how the code is
grouped. Each one is the single place for its subject: what was measured, what each change bought,
what was tried and abandoned, and what is still open. They carry their own withdrawn claims beside
the corrected ones, on purpose — a reader who sees only the final number learns nothing about how it
went wrong.

**The corrections register** is the index over everything this programme has published and then
withdrawn, grouped by the eight mechanisms that produced them rather than by which file they sit in.
It does not replace the withdrawals themselves, which stay beside the claims they correct; it exists
because the same mistake has recurred across documents that never cite each other, and no single
document can see that. **Read it before publishing a figure or quoting one.**

**For the global campaign specifically**, [`campaign/campaign-plan.md`](campaign/campaign-plan.md)
is the entry point: what will run, with what settings, at what cost, and what is still open. It
links onward to the sizing, cost and ingest evidence rather than restating it, and its §11 states
which document is authoritative for what.

## Conventions

**One subject, one file.** If a finding belongs to ingest, it goes in the ingest record — not in a
new document that the ingest record then has to cite. The corpus reached 29 topical files once, and
the cost was not the reading: it was that a superseded number could sit uncorrected in a neighbouring
file for weeks. Add a section, not a file.

**Name a file for its subject, not for when it was written.** Dates belong inside the document, on
the measurement they qualify.

**Keep the layout block above current** — add a row in the same commit that adds a file, and remove
one in the same commit that removes a file. There is no test enforcing this, deliberately: the
documentation tree is not a subject for the test suite.

## When to add a new ADR

- A change PR conflicts with an existing ADR.
- A change adds a new architectural constraint future contributors need to know about.
- Multiple PRs have hit the same surprise; the ADR makes the reasoning durable.

We don't write ADRs for trivial choices (variable names, file layout) — only for decisions that
shape multiple components or constrain future work.
