# 003 — Tenacity retries (not Prefect's `@task(retries=...)`)

**Status:** Accepted (v0.1.0)

## Context

The ingest pipeline has a known transient-failure mode: GDAL hits
intermittent COG-read errors when fetching ~50–100 tiles
concurrently from S3 / CloudFront. The error rate is < 1%, but at
scale it surfaces on every run and needs retry logic.

Two retry strategies were on the table:

* **Prefect-native:** `@task(retries=3, retry_delay_seconds=...)`.
  The orchestrator catches the exception, re-runs the entire task.
* **Tenacity, in-domain:** `tenacity.Retrying(...)` wraps just the
  flaky call (`write_dataset`).

## Decision

**Tenacity, in-domain, narrowly scoped.** Each domain function
wraps its single retry-eligible operation:

```python
for attempt in Retrying(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
):
    with attempt:
        write_dataset(...)
```

The retry happens inside the domain function. The `@task` doesn't
set `retries=`.

## Rejected alternatives

**`@task(retries=...)`:** The retry would re-run the entire task,
including the multi-day STAC query, the SCL coverage check, all the
expensive setup, and the tenacity-eligible last-step write. A
retried task does ~5–10 minutes of redundant work to get back to
the actual flaky call.

It also couples retry behaviour to the orchestrator. The same
flaky call from `runners/plain.py` would have no retry at all,
since the plain runner doesn't have Prefect's retry machinery.
Either we re-implement it there, or we accept that the plain
runner is less robust than the Prefect path — both bad outcomes.

## Consequences

- **Pro:** retry scope matches the failure scope. Cheap.
- **Pro:** retry logic is identical across Prefect, Dagster,
  Airflow, the plain runner, anything — it's a domain-layer
  concern.
- **Pro:** `runners/plain.py` is just as robust as the production
  Prefect flow.
- **Con:** Prefect's UI doesn't visualise tenacity retries the way
  it visualises task-level retries. We accept this — the relevant
  observability is in the structured log line tenacity emits via
  `before_sleep_log`.
- **Con:** if the failure mode shifts (e.g. a new flaky operation
  surfaces upstream of `write_dataset`), the tenacity wrap won't
  catch it; we'll need to add another wrap. Cheap to do; the
  scope-matching trade-off is the right one.

## Related

- `appendix/prefect_flow_handling.md` §3.3 — the longer-form analysis. **It is in the
  open-sourcing planning archive, which is gitignored and therefore not in the repository**, so
  this is a pointer for whoever holds that archive rather than a link a reader can follow.
- [`docs/prefect-setup.md`](../../docs/prefect-setup.md) §"Common
  gotchas" — what users should know.
