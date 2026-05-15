# AWS Ray provider — gotchas, ops, and how to add a new cloud

The `tessera_embeddings.providers.aws.ray` module ships a reference
implementation of the Ray cluster provisioning contract on AWS. This
doc covers the concepts and operational quirks that aren't obvious
from reading the code.

The two-line summary: **the cluster YAML is a template, and SSM is
the source of truth for environment-specific resource IDs**. There
are no hardcoded AWS account IDs in source control.

---

## The provisioning contract

A "Ray provider" is duck-typed: a `ray_cluster` context manager that
yields *something* (a head address, a YAML path, `None`, …) and tears
the cluster down on exit. There is intentionally no abstract base
class — providers vary too much across clouds for a useful interface.
Adding a new cloud means writing a new sibling directory whose
`ray_cluster` accepts whatever inputs that cloud needs.

The AWS implementation owns five concerns:

1. **Resolve the cluster YAML.** Read SSM for environment-specific
   IDs, pick a subnet, materialise the SSH key, inject the CloudWatch
   agent config, write a resolved tempfile.
2. **Optional code-sync.** When `sync_source_path` is set, tar the
   source tree and upload it to S3 so workers can pull it on boot.
3. **`ray up`.** Launch the cluster from the resolved YAML.
4. **`ray.init`.** Connect the orchestration runtime (Prefect flow
   runner, plain runner, local script) to the head node.
5. **Teardown.** `ray.shutdown` + `ray down` + clean up tempfiles.

---

## The cluster YAML template

`providers/aws/cluster.yaml.template` ships a working default for an
A10G GPU pool on `g5.2xlarge`. Account-bound IDs are absent; comments
flag every injection point with `# INJECT: <what>`.

**Why runtime injection from SSM?** Account-bound IDs (security
group, subnets, IAM role, key pair) are not known at template-write
time and can't be committed to source. The SSH private key in
particular must never be in source control. SSM is also where AWS
stores the key material when EC2 creates a key pair, so reading from
SSM at runtime is the natural retrieval point. The same template
works across environments — dev branches just point at a different
AMI parameter (e.g. `/tessera/ray/ami-id-mybranch`).

Required SSM keys under `ssm_prefix` (default `/tessera/ray/`):

| Key | Type | Purpose |
|---|---|---|
| `security-group-id` | String | Cluster security group |
| `instance-profile-arn` | String | IAM instance profile ARN |
| `private-subnet-ids` | StringList (comma-sep) | Private subnets to choose from |
| `key-pair-name` | String | EC2 key pair name (for SSH) |
| `key-pair-id` | String | EC2 key pair ID (for SSH key lookup) |

The AMI ID lives in a separate SSM parameter, named via the
`ami_ssm_name` argument to `ray_cluster`.

---

## AMI baking pattern

Strongly recommended for production. The default `cluster.yaml.template`
expects the worker AMI to have:

- NVIDIA drivers + CUDA (use the AWS Ubuntu 22.04 GPU DLAMI as base)
- Python 3.12 venv at `/opt/tessera/venv` with all inference deps
  (`torch`, `ray`, `tessera_embeddings`)
- `aws` CLI for the optional source-tarball pull

The reference repo uses Packer to build this AMI in CI, then publishes
the AMI ID to SSM so the cluster YAML resolves it at runtime. Any
other AMI-baking tool works — what matters is that workers boot ready
to run, with no `pip install` step at provisioning time.

Why bake instead of `pip install` on every boot? At 100–500 workers,
pip install of `torch` + `ray` + their transitives is the dominant
cost (~5 min per worker). Baking moves it to a single CI job; nodes
boot ready in ~1 minute.

If you ship a different layout (different venv path, different
package install location), edit the `head_start_ray_commands` /
`worker_start_ray_commands` and the `setup_commands` accordingly.

---

## Code sync (`sync_source_path`)

Two supported paths for getting application code onto workers:

1. **AMI-baked (default, recommended for production).** Skip
   `sync_source_path` entirely; the source is already inside the
   AMI's venv. The `aws s3 cp` line in `setup_commands` becomes a
   no-op or you remove it from your customised template.
2. **Runtime sync (dev iteration).** Set
   `sync_source_path=Path("src/tessera_embeddings")` and pass
   `code_bucket="my-dev-bucket"`. The provider tars the directory and
   uploads it to `s3://{code_bucket}/code/src{code_suffix}.tar.gz`
   before `ray up`. Workers pull this tarball during `setup_commands`.

`code_suffix` lets multiple branches coexist in the same bucket
(e.g. `-mybranch` → `src-mybranch.tar.gz`). The reference repo
auto-derives this from the git branch; in the OSS provider you pass
it explicitly so the orchestration layer is in charge of the
convention.

---

## CloudWatch logging

Each Ray node ships `/tmp/ray/session_latest/logs/*` to a CloudWatch
log group (`cloudwatch_log_group`, default `/ec2/tessera/ray`). Log
streams are namespaced by instance ID:

| Stream suffix | Source |
|---|---|
| `{instance-id}/raylet` | `raylet.log` |
| `{instance-id}/gcs_server` | `gcs_server.log` (head only in practice) |
| `{instance-id}/dashboard` | `dashboard.log` |
| `{instance-id}/monitor` | `monitor.log` (autoscaler) |
| `{instance-id}/workers` | `worker-*.log` (actor stdout) |
| `{instance-id}/actors` | `worker-*.err` (actor stderr / tracebacks) |

The agent config lives in `cloudwatch-agent.json.tpl` as readable
JSON with `__LOG_GROUP__` and `__INSTANCE_ID__` placeholders. At
resolve time, `_build_cloudwatch_setup_command` compacts it to a
single line, substitutes the log group, and wraps it in a shell
command that resolves the real instance ID at boot via the EC2 IMDS.

This avoids heredocs in `setup_commands` — which break when Ray
sends them over SSH (indented `EOF` terminators are never matched
by bash).

`providers/aws/diagnostics.py::make_cloudwatch_fetcher` returns a
`fetch_events(instance_id) -> list[dict]` callable that the
domain-pure `inference/diagnostics.py` accepts. Wire it in at flow
boundaries when you want post-mortem memory-ramp tables on actor
failures.

---

## Teardown — three layers of defence

GPU instances are expensive; leaks are felt fast. The provider has
three lines of defence:

1. **Per-actor retirement** (in the inference scheduler, not this
   provider): retire idle actors during inference. Pair with
   `make_instance_terminator` so the underlying EC2 instance is
   terminated immediately rather than waiting for the autoscaler.
2. **`ray down`** at context-manager exit. Terminates everything
   `ray up` provisioned.
3. **Autoscaler idle timeout** (default 2 min in the YAML template).
   Final safety net if `ray down` fails — idle workers self-retire.

For Prefect cancellation specifically, register
`terminate_ray_instances_by_tag` and `cleanup_ray_tempfiles` on the
flow's `on_cancellation` hook. They handle the case where the flow
was cancelled before `ray up` returned (no resolved YAML to feed to
`ray down`).

---

## Connection modes

`ray_cluster` supports two modes:

| Mode | How to invoke | Use case |
|---|---|---|
| **Managed** | default | Production: provider owns full lifecycle |
| **Attach** | `ray_address="ray://..."` | Debug against a running cluster; provider does NOT call `ray up`/`ray down` |

Local single-node mode lives in
`tessera_embeddings.providers.local.ray::ray_cluster` — it's a
separate context manager because the AWS argument surface (SSM
prefix, AMI, tags, …) is irrelevant locally. Code that wants to be
substrate-agnostic should accept a `ray_cluster` callable as a
parameter and let the caller pick the provider.

---

## Adding a new cloud provider

The AWS provider is the reference. To add GCP, Azure, on-prem, or
Kubernetes:

1. **Create a sibling directory** under `providers/`, e.g.
   `providers/gcp/`. Add a `__init__.py` and a `ray.py`.
2. **Write your own `ray_cluster` context manager.** Match the AWS
   signature loosely — same idea (provision, connect, yield, tear
   down), different arguments. Don't try to inherit from the AWS one.
3. **Resolve substrate-specific config however your cloud does it.**
   GCP would read from Secret Manager + GCE metadata; Azure from Key
   Vault. The AWS path through SSM is one example, not a contract.
4. **Provide your own `cluster.yaml.template`** if the Ray autoscaler
   cluster YAML differs in shape from AWS (`provider.type` will
   differ; node config keys will differ). Mark injection points with
   `# INJECT: <what>` comments matching the AWS template's style.
5. **Wire your terminator** (`make_instance_terminator` equivalent)
   into the inference scheduler via the `on_actor_retire` callback so
   idle actors terminate their underlying VMs promptly.
6. **Document the gotchas in a sibling `gotchas.md`.** What surprised
   you, what credential resolution looks like, how teardown is
   defended in depth.

The orchestration layer (Phase 8 onwards) accepts a `ray_cluster`
callable as a parameter — there is no enum of supported clouds. If
you wrote a context manager, it works.

---

## Kubernetes

Not currently shipped; the pattern would be a `providers/k8s/ray.py`
that wraps a `KubeRay` cluster CR. Many of the AWS concerns
(SSM resolution, `ray up`/`ray down`, EC2 termination) collapse on
Kubernetes — the cluster is reconciled by an operator and teardown
is `kubectl delete`. The CloudWatch fetcher would be replaced by a
pod-log fetcher (e.g. via the Kubernetes API). Open to PRs.
