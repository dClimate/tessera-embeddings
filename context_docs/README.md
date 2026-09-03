# context_docs/

Design records and decision rationale for `tessera_embeddings`.
This isn't reference documentation — that lives in `docs/`. This is
the **rationale** behind the choices. Read it if you're trying to
understand why the code looks the way it does, or if you're
proposing a change that conflicts with one of these decisions.

## Layout

```
context_docs/
├── corrections-register.md   EVERY withdrawn figure, grouped by how it went wrong
├── decisions/        Architecture Decision Records (1 page each)
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
│   └── 022-resolve-the-roi-mask-credential-at-read-time.md
└── design/           Long-form framing docs, grouped by what they answer
    │
    │  THE CAMPAIGN — start here
    ├── campaign-plan.md                          <- START HERE; §10 is the authority map
    ├── campaign-cost-model.md                    every cost, rate, fleet size, GPU-hour
    ├── campaign-cluster-sizing.md                how work balances across N clusters
    ├── campaign_inference_profile_2026_08.md     measured per-cell inference behaviour
    ├── ec2_launch_throttling_2026_08.md          why fleets throttle themselves on RunInstances, and the two bounds
    ├── radar_source_coverage_2026_08.md          which zones publish no usable radar
    ├── optical_depth_census_2026_08.md           how much of the product is optically thin, and where
    ├── minimum-optical-depth-plan.md             BUILT at 15; the evidence behind ADR-018
    ├── window_legibility_vs_depth_2026_08.md     the elbow at 25 — which the line at 15 knowingly crosses
    ├── optical_retention_per_pixel_2026_08.md    what each candidate line costs, counted per pixel
    ├── inference-perf-run-ledger.md              raw per-run measurements
    ├── campaign-monitoring-plan.md               what is watched while it runs, and how
    ├── immediate-refill-of-a-settled-fill.md      how a dead fill's roster is recovered without waiting for the round
    ├── commit-gate-removal-2026_08.md            why the fleet-wide committer limit is gone, and when to put it back
    ├── final-data-validation-plan.md             the closing gate over every published cell
    ├── optical-registry-2026-08-19.md            the published per-tile index: schema, and what the first live run corrected
    │
    │  INGEST
    ├── ingest_optimization_campaign_2026_07.md   every ingest measurement
    ├── ingest_concurrency_investigation_2026_08.md
    ├── gdal-read-config-2026_08.md               odc shadows three GDAL options on the imagery read path; no knob overrides them
    ├── ingest_read_failure_causes_2026_08.md     five source-read failure causes, and the retry budget they share
    ├── solar_day_fusion_order_2026_08.md         which scene wins a contested pixel, and why it was inverted
    ├── ingest-live-tile-cropping.md              + appendix A, the multi-write-per-commit test
    ├── ingest-graph-and-stac-budget.md
    ├── single-path-audit-2026-09.md          the SINGLE-ROI path after the campaign: what it inherited, what it did not
    ├── region-writes.md
    │
    │  INFERENCE + STORE
    ├── inference_gpu_saturation_profile_2026_07.md
    ├── gpu-card-choice-2026_08.md                which GPU rungs the campaign may open, and why
    ├── a10g_batch_size_2026_08.md                why the inference batch is sized to the card (written for a general reader)
    ├── stage_decoupling_2026_08.md              why ingest/inference/assembly are ungated
    ├── icechunk-credential-stampede-2026_08.md   the 28 Aug storage-credential incident (written for a general reader)
    ├── assembly-worker-clamp-2026_08.md          why assembly ran 5 of its 16 forks
    ├── keeping-the-assembly-session-current-2026_08.md  why an assembly catches up while its forks write
    ├── single-global-alignment.md                why single-ROI was aligned to the campaign
    ├── staging-identity-and-resume.md            what a run_id identifies, and what resumes
    ├── d3-sharding-plan.md                       settled ADR-008 D3; spec for scale_tests/t8
    ├── global-store-test-plan.md                 the T0-T8 scale tests; §8 is the icechunk API ledger
    ├── test-suite-streamlining-plan.md           where the unit suite's time goes, and what is safe to cut
    │
    │  MODEL
    └── v2_data_source_alignment_2026_07.md       AWS-vs-MPC input for Tessera v2
```

## How to read these

**The corrections register** is the index over everything this programme has published and
then withdrawn — **83 marked retractions across the 14 documents that carried them, audited
2026-08-11** — grouped by the eight mechanisms that produced them rather than by which file
they sit in. It does not replace the withdrawals
themselves, which stay beside the claims they correct; it exists because the same mistake
has recurred across documents that never cite each other, and no single document can see
that. Read it before publishing a figure or quoting one.



**Decision records** answer "why is X the way it is?" They have a
fixed shape — Context, Decision, Rejected alternatives, Consequences
— so you can scan to the part you need. Append-only: supersede with
a new record rather than editing in place.

**Design docs** are the longer-form framing material that informed
the decisions. Read these if a single ADR doesn't satisfy your
question and you need the broader thinking.

**For the global campaign specifically**, `design/campaign-plan.md` is
the entry point: what will run, with what settings, at what cost, and
what is still open. It links onward to the sizing, cost and ingest
evidence rather than restating it.

## When to add a new ADR

- A change PR conflicts with an existing ADR.
- A change adds a new architectural constraint future contributors
  need to know about.
- Multiple PRs have hit the same surprise; the ADR makes the
  reasoning durable.

We don't write ADRs for trivial choices (variable names, file
layout) — only for decisions that shape multiple components or
constrain future work.
