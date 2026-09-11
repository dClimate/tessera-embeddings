# providers/

Cloud-substrate provisioning for the reference orchestration layer.

Each provider owns a concrete cloud (or local) implementation of the
two substrates the pipeline uses:

* **Ray clusters** — for distributed GPU inference.
* **Dask clusters** — for ingest fan-out and assembly.

There is no abstract `Provider` base class — providers are
**duck-typed context managers** that yield substrate handles. The
contract is "what the orchestration layer calls", not "what an
interface declares":

```
with ecs_cluster(log, min_workers=..., max_workers=...) as cluster:
    runner = DaskTaskRunner(cluster.scheduler_address)
    inner_flow.with_options(task_runner=runner)(...)

with ray_cluster(log, ami_ssm_name=...) as resolved_yaml:
    # `ray.init()` already called by the context manager
    actors = [InferenceActor.options(num_gpus=1).remote(cfg, ckpt)
              for _ in range(num_actors)]
    ...
```

If your context manager has those shapes, the orchestration layer
doesn't care what's underneath.

## Layout

```
providers/
├── aws/                      Fully maintained AWS reference
│   ├── ray.py                ray_cluster + SSM-driven YAML resolution
│   ├── dask.py               ecs_cluster (Fargate or hybrid EC2 scheduler)
│   ├── diagnostics.py        CloudWatch fetch_events for inference.diagnostics
│   ├── cluster.yaml.template Ray autoscaler template with INJECT comments
│   ├── cloudwatch-agent.json.tpl
│   └── gotchas.md            operational knowledge: AMI bake, teardown, etc.
└── local/                    Single-machine fallbacks
    ├── ray.py                ray.init / ray.shutdown ctx manager
    └── dask.py               LocalCluster ctx manager
```

## Scheduler health logging (Dask)

`ecs_cluster` registers a `SchedulerResourceLogger` (a `distributed`
`SchedulerPlugin`) on the scheduler at startup. The cluster dashboard
shows aggregate *worker* load, but the scheduler is a single
event-loop process that builds every graph and routes every task —
its own pressure is invisible there until it stalls (the built-in
`Event loop was unresponsive for Ns` warning) or the ECS task is
silently OOM-killed.

The plugin runs a `PeriodicCallback` on the scheduler's event loop and
emits one line every `DEFAULT_SCHEDULER_PROFILE_INTERVAL_S` (30s) to
the `dask-scheduler` CloudWatch stream:

```
scheduler health: cpu=83% rss=6.41GiB mem=80% lag=0.2s fds=412 threads=22 workers=148 tasks=51234 processing=296 no-worker=0
```

| Field | Meaning / why it matters |
|---|---|
| `cpu` | Scheduler process CPU %, interval-averaged (can exceed 100% across threads). Sustained ~100% precedes event-loop stalls. |
| `rss` / `mem` | Process resident memory, absolute and as % of the container limit. The OOM predictor. |
| `lag` | Event-loop lag — how late the callback fired vs. schedule. The leading measure behind the "unresponsive" warning. |
| `fds` / `threads` | Open file descriptors and thread count — catches connection/fd leaks. |
| `workers` / `tasks` | Cluster size and total tracked tasks, with `processing` (in flight) and `no-worker` (stuck waiting). A rising backlog means the scheduler is falling behind. |

Registration is best-effort: a failure logs a warning and the run
continues, since this is diagnostics only.

## When a provider is needed

```
flow runner (or plain.py)
        │
        │   with ecs_cluster(...) as cluster:        ← provider call
        │       …                                     ← your work runs here
        │   # cluster torn down on exit
        │
        ▼
production: AWS Fargate (provider/aws)
local dev:  dask.distributed.LocalCluster (provider/local)
your cloud: fork a sibling directory; same context-manager shape
```

Providers are imported **explicitly** by the flow file —
`from tessera_embeddings.providers.aws.dask import ecs_cluster`.
There's no plugin discovery, no entry-point magic, no `Provider`
factory. To swap clouds: edit the import.

## Adding a new cloud

Step-by-step in
[`docs/providers/adding-your-own.md`](../../../docs/providers/adding-your-own.md).
The short version:

1. Create `providers/<your_cloud>/`.
2. Implement `ray_cluster(...)` and `dask_cluster(...)` (or whatever
   names your cloud's substrate calls them) as context managers
   yielding handles compatible with the consumers in
   `orchestration/prefect/flows/`.
3. If your cluster YAML / DAG-runner config differs from AWS's, ship
   templates next to the Python code (as we do with
   `cluster.yaml.template`).
4. Update or fork the flow file to import your provider instead of
   the AWS one.
5. Add a parity test under `tests/parity/<your_cloud>/` proving your
   provider produces the same outputs as `runners/plain.py` on the
   bundled quickstart ROI. See
   [`tests/parity/adapter_template/`](../../../tests/parity/adapter_template/).

The AWS provider's `gotchas.md` documents what we wished we'd known —
worth skimming before you start.

## Why no abstraction?

Cloud APIs are too different to share a useful base class — AWS uses
SSM + EC2 + ECS Fargate where GCP would use Secret Manager + GCE and
on-prem might use Slurm — so anything fitting all three is too thin to
help or too thick to evolve. Parameter surfaces make it worse: an
`ami_ssm_name` kwarg is meaningless on GCP, and a shared signature
would carry dead parameters and surprise `NotImplementedError`s.
Forking is cheap by comparison: a provider is ~500 LOC and shares
essentially nothing cloud-specific with its siblings.

Full argument in
[`context_docs/decisions/004-duck-typed-providers.md`](../../../context_docs/decisions/004-duck-typed-providers.md).
