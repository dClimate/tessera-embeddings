# 029 — One flow per ingest

**Status:** Proposed 2026-10-19. Replaces the "two-flow pattern" guidance in the Prefect README and
the ingest flows' docstrings, which no ADR recorded.

## Context

Each ingest deployment (`ingest_s2_roi_reflectance`, `ingest_s1_roi_sar`) was two flows. The outer
flow provisioned the Dask cluster; an inner `ingest_s*_roi_impl` flow, given a `DaskTaskRunner`
through `.with_options(task_runner=...)`, submitted the task shell to it. The split existed because a
task runner binds when a flow is defined, and the scheduler address exists only once the flow body
has provisioned the cluster.

It cost three things. Every leg was two flow runs, the inner one unnamed. The task's progress lines
(per date for S2, per batch for S1) landed on that inner run, not on the deployment run an operator
opens. A failed leg showed up as two failed runs in any monitoring that counts runs. The inner
flow's parameters, the S1 credential callables among them, also went through Prefect's flow-run
parameter handling for no benefit.

## Decision

**One flow per ingest.** The flow provisions its cluster as before, then runs its task shell with
`flows/_dask_lifecycle.run_on_cluster`, which submits it through a `prefect_dask.PrefectDaskClient`
on the new scheduler. The task executes on a Dask worker, as it did before, and Prefect records it as
a task run of the calling flow run, so its logs land on the deployment's own run.

Nothing else moves: the driver still runs on a worker (the worker sizing in `providers/aws/dask.py`
still applies), and the task shells, domain functions, flow names, parameters, entrypoints, teardown
hooks, resource tags, performance report and `use_local` path are all unchanged. The change sits
outside both code identities. A local run of the S2 flow on main and on this change wrote mosaics
that compare identical.

`PrefectDaskClient.submit(task, parameters=...)` is the call `DaskTaskRunner.submit` makes itself, so
the task engine path on the worker is the same one the inner flow used.

## Rejected alternatives

- **Call the domain function in the flow body with a `distributed.Client`** (driver relocation).
  This also removes the split and puts logs on the flow run, but it moves the driver off the worker
  and onto the flow runner. Its memory and graph-build CPU move with it, gathered results route
  through the scheduler, and driver log lines move from the Dask worker log streams that the
  profiling queries read to the flow runner's. S1's `set_s3_credentials` would also rewrite the
  `AWS_*` variables of the process that manages the ECS cluster, and nobody has verified that
  scaling and teardown keep their own identity after that. It is a behaviour change that needs
  measuring on AWS, so it is its own decision. This change does not block it: it would replace the
  `run_on_cluster` call with a direct one.
- **`DaskTaskRunner(cluster_class=..., cluster_kwargs=...)`.** The runner would build the cluster
  itself, but its kwargs are fixed when the flow is defined. Per-run sizing (`min_workers`,
  `max_workers`), `worker_env_overrides`, `ec2_scheduler`, the flow-run resource tag and the
  provisioning retries all live in `ecs_cluster` and take run-time values. Applying them would need
  `.with_options` again, which brings back the wrapper flow.
- **Keep the split.** It keeps the costs above and buys nothing the single flow lacks.

## Consequences

- One flow run per leg, carrying its own progress lines. Tooling that walked into an inner run to
  read a leg's logs reads the leg itself. Tooling that counted the inner run as a second failure
  record stops seeing it.
- Dask workers remain Prefect API clients for the task's logs, as before.
- `export_zone_rois` is the one flow left that calls an inner flow, to size its
  `ThreadPoolTaskRunner` per run. The same approach would apply there.
