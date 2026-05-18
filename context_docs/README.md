# context_docs/

Design records and decision rationale for `tessera_embeddings`.
This isn't reference documentation — that lives in `docs/`. This is
the **rationale** behind the choices. Read it if you're trying to
understand why the code looks the way it does, or if you're
proposing a change that conflicts with one of these decisions.

## Layout

```
context_docs/
├── decisions/        Architecture Decision Records (1 page each)
│   ├── 001-thin-prefect-wrapping.md
│   ├── 002-shape-c-task-shells.md
│   ├── 003-tenacity-retries-not-prefect-native.md
│   ├── 004-duck-typed-providers.md
│   ├── 005-multi-lock-file-strategy.md
│   └── 006-adapter-policy.md
└── design/           Long-form framing docs (ported from planning)
    ├── open_sourcing_conceptual_background.md
    └── orchestration_infra_leakage_audit.md
```

## How to read these

**Decision records** answer "why is X the way it is?" They have a
fixed shape — Context, Decision, Rejected alternatives, Consequences
— so you can scan to the part you need. Append-only: supersede with
a new record rather than editing in place.

**Design docs** are the longer-form framing material that informed
the decisions. Read these if a single ADR doesn't satisfy your
question and you need the broader thinking.

## When to add a new ADR

- A change PR conflicts with an existing ADR.
- A change adds a new architectural constraint future contributors
  need to know about.
- Multiple PRs have hit the same surprise; the ADR makes the
  reasoning durable.

We don't write ADRs for trivial choices (variable names, file
layout) — only for decisions that shape multiple components or
constrain future work.
