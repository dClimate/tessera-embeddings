# AWS provider

The fully-maintained reference cloud. This doc covers what you need
to provision in AWS for `tessera_embeddings.providers.aws.*` to work
end-to-end. The package itself ships zero IaC — VPCs, IAM, ECR,
ECS, etc. are your responsibility.

For day-to-day operations (Ray cluster behaviour, AMI baking,
teardown defence-in-depth), see
[`src/tessera_embeddings/providers/aws/gotchas.md`](../../src/tessera_embeddings/providers/aws/gotchas.md) —
that's the operational companion to this provisioning doc.

## What lives where

```
                ┌──────────────────────────────────────────┐
                │     Prefect server (your responsibility) │
                │                                          │
                │     Triggers a flow run                  │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │     Flow runner (ECS Fargate task)       │
                │                                          │
                │     reads VPC_ID, PRIVATE_SUBNETS, ...   │
                │     from work-pool job-template env vars │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────┴─────────────────────┐
                │                                          │
                ▼                                          ▼
   ┌──────────────────────────┐              ┌──────────────────────────┐
   │  Ray cluster on EC2      │              │  Dask cluster on Fargate │
   │                          │              │                          │
   │  reads SSM /tessera/ray/ │              │  reads env vars from     │
   │  for SG, subnets, IAM,   │              │  the flow runner's job   │
   │  AMI, SSH key            │              │  template                │
   │                          │              │                          │
   │  GPU inference           │              │  Ingest fan-out,         │
   │                          │              │  embedding assembly      │
   └──────────────────────────┘              └──────────────────────────┘
```

Two distinct config surfaces:

* **SSM Parameter Store** for the **Ray cluster** — runtime YAML
  injection. Account-bound IDs (security group, subnets, IAM, key
  pair, AMI) live in SSM because they're not known until your IaC
  has run, and the SSH key in particular cannot be in source
  control.

* **Env vars on the flow runner's ECS task definition** for the
  **Dask cluster** — read at flow start by `get_fargate_config()`.

The two surfaces don't overlap; the AWS Ray provider doesn't read
env vars, the AWS Dask provider doesn't read SSM.

## Required SSM keys (Ray)

Under the prefix you pass to `ray_cluster(ssm_prefix=...)` (default
`/tessera/ray/`):

| Key | Type | Purpose |
|---|---|---|
| `security-group-id` | String | Ray cluster security group |
| `instance-profile-arn` | String | IAM instance profile ARN for cluster nodes |
| `private-subnet-ids` | StringList (comma-sep) | Private subnets to choose from |
| `key-pair-name` | String | EC2 key pair name (for SSH from flow runner) |
| `key-pair-id` | String | EC2 key pair ID (used to look up the SSH key material at `/ec2/keypair/<id>`) |

Plus an AMI parameter at the path you pass via `ami_ssm_name`,
e.g. `/tessera/ray/ami-id`. The AMI ID points at the GPU AMI you've
baked (see `gotchas.md` for the AMI-bake pattern).

### Optional SSM key: `gpu-worker-ladder`

| Key | Type | Purpose |
|---|---|---|
| `gpu-worker-ladder` | String | Per-instance-type GPU worker ceilings, e.g. `g6e.xlarge:101,g5.2xlarge:105` |

**Absent is the default, and it means the cluster template's node types stand
exactly as shipped.** Set it and `_resolve_ray_config` rewrites the `max_workers`
of each on-demand GPU rung from the value, letting you change which EC2 instance
types a fleet may buy — and how many of each — without a release, a deployment
re-registration, or an AMI re-bake. It does still require provisioning a cluster
after the edit; see the first bullet below. Every `g6e` size is the same L40S on x86_64,
so one AMI serves all of them.

Four properties worth knowing before you set it:

- **It is read ONCE, when a cluster's config is resolved before `ray up`.** The
  running autoscaler never re-reads it, so editing the parameter does nothing to a
  fleet that is already up: it keeps requesting the rung it launched with, however
  long the capacity refusal lasts. **Failover therefore requires a new cluster**,
  not a parameter edit — set the value first, then provision. Treat this as the
  operational cost of the mechanism: it is a pre-launch knob, not a live one.
- **It is authoritative, not additive.** A rung the value does not name is set to
  `0`. So `g6e.2xlarge:150` alone also closes `g6e.xlarge`; name every rung you
  want the fleet to use.
- **It refuses rather than warns.** A malformed entry, a repeated instance type,
  or an instance type no shipped rung offers fails the cluster launch. A ladder
  that part-applied would grow a fleet nobody asked for.
- **`max_workers` is the only mechanism that moves Ray's choice.** Ray's
  autoscaler scores node types purely from their resources; its scorer takes a
  `node_availability_summary` and never reads it, so an `InsufficientInstanceCapacity`
  has *no* influence on which type it asks for next. Two consequences:
  `max_workers: 0` makes a rung genuinely unreachable, and **opening a second rung
  beside an unrestricted first one buys nothing** — you must lower the first rung's
  count to push demand onto the second. See `TestRayNodeTypePreference`.

Spot rungs are outside the ladder's domain and are never touched by it.

Populate these with whatever IaC you use:

```
# Pulumi sketch
ray_sg = aws.ec2.SecurityGroup("tessera-ray-sg", ...)
ssm.Parameter("ray-sg-id",
    name=f"{ssm_prefix}security-group-id",
    type="String", value=ray_sg.id)

ray_role = aws.iam.Role("tessera-ray-instance-role", ...)
ray_profile = aws.iam.InstanceProfile("tessera-ray-profile", role=ray_role.name)
ssm.Parameter("ray-instance-profile-arn",
    name=f"{ssm_prefix}instance-profile-arn",
    type="String", value=ray_profile.arn)

# ... and so on for the other keys
```

## Required env vars (Dask)

Set these on your flow runner's ECS task definition (typically via
your work-pool job template):

| Env var | Purpose |
|---|---|
| `ECS_CLUSTER_ARN` | Target ECS cluster ARN for Fargate workers |
| `DASK_ECR_IMAGE_URI` (or `ECR_IMAGE_URI`) | Container image (must have `dask`, `xarray`, `tessera_embeddings` installed) |
| `VPC_ID` | VPC for Fargate networking |
| `PRIVATE_SUBNETS` | Comma-separated private subnet IDs |
| `SECURITY_GROUP_ID` | Security group for Fargate tasks |
| `ECS_EXECUTION_ROLE_ARN` | ECS execution role (pulls image, writes logs) |
| `DASK_TASK_ROLE_ARN` | Task role used by Dask worker tasks (S3 access, etc.) |
| `CLOUDWATCH_LOG_GROUP` | CloudWatch log group for Dask agent logs (default `/ecs/tessera/dask`) |

For the optional EC2-scheduler mode (`ec2_scheduler=True` on
`ecs_cluster`), also set:

| Env var | Purpose |
|---|---|
| `EC2_SCHEDULER_CAPACITY_PROVIDER` | ECS capacity provider name backed by an EC2 ASG |
| `EC2_SCHEDULER_SUBNET` | Single subnet for both scheduler and workers (avoids cross-AZ transfer) |

## IAM you'll need

The flow runner's task role needs:

```
- ssm:GetParameter, ssm:GetParametersByPath  → for the Ray provider's SSM reads
- ec2:DescribeSubnets, ec2:RunInstances, ec2:TerminateInstances,
  ec2:DescribeInstances, ec2:CreateTags                → for ray up/down
- iam:PassRole (for the Ray instance profile)         → for ec2:RunInstances
- ecs:RunTask, ecs:DescribeTasks, ecs:ListTasks,
  ecs:RegisterTaskDefinition, ecs:DescribeTaskDefinition → for dask-cloudprovider
- iam:PassRole (for the Dask task + execution roles)  → for ECS RunTask
- s3:* on your input/output buckets                    → for fsspec operations
- logs:CreateLogStream, logs:PutLogEvents              → for CloudWatch
- logs:FilterLogEvents                                 → for the diagnostics fetcher
```

The Ray cluster instance profile (the one whose ARN sits in SSM)
needs:

```
- s3:* on your buckets                  → for the inference workers themselves
- ssm:GetParameter                      → for any per-instance config
- ec2:DescribeInstances                 → for the autoscaler
- logs:* on the Ray log group           → for the CloudWatch agent
```

Both roles also need the standard ECS / EC2 trust policies for the
tasks/instances they're attached to.

## The cluster YAML template

Lives at
[`src/tessera_embeddings/providers/aws/cluster.yaml.template`](../../src/tessera_embeddings/providers/aws/cluster.yaml.template).
INJECT comments mark every account-bound field. Don't edit the
template unless you're changing cluster shape (instance type,
worker counts) — the resource-ID injection is automated.

Default node config:

```
head:        m5.2xlarge          (8 vCPU, 32 GB)   GCS + autoscaler
workers:     g6e.xlarge          (4 vCPU, 32 GB,    1 InferenceActor each
                                  1× L40S 45,776 MiB,
                                  250 GB NVMe)
```

The L40S carries **45,776 MiB — 44.7 GiB, or 48.0 GB decimal**. All three numbers
name the same card; quote the unit, because dropping it has already put the figure
in our docs two different ways. 250 GB NVMe is fast enough for `torch.load` of the
checkpoint without EBS stalls, and the NIC is rated "up to 20 Gbps" — a burst
credit, not a sustained floor — for the S3-heavy load phase. ~$1.86/hr on-demand
at us-west-2; spot varies (~$0.5–0.9/hr).

Scaling is horizontal: one GPU, one `InferenceActor`. The 4 vCPUs make host-side
data loading the tight resource per worker, and the template ships two
**capacity-fallback** rungs, both at `max_workers: 0`:

| rung | card | vCPU/GPU | host GiB | throughput vs L40S |
|---|---|---:|---:|---:|
| `g5.2xlarge` | A10G | 8 | 32 | **0.46** |
| `g6.2xlarge` | L4 | 8 | 32 | 0.32 |

**Opening one is a campaign parameter, not a release.** Pass
`gpu_fallback_instance_types=["g5.2xlarge"]` to `run-global-campaign`. That does two things per fill,
and neither works without the other:

1. **Opens the card's rung** at the same ceiling as the production rung, so either card
   can carry the whole fleet.
2. **Installs a capacity-aware autoscaler scorer.** Ray's stock scorer *cannot see
   capacity* — `_resource_based_utilization_scorer` takes a `node_availability_summary`
   and never reads it — so a rung refusing with `InsufficientInstanceCapacity` stays
   top-scored and is re-requested indefinitely while an open fallback sits idle.

With both in place the fleet prefers the L40S whenever AWS will sell one and moves to
the A10G when it will not. Recovery applies to NEW demand, not to nodes already running:
Ray clears the unavailability record on the next successful launch of that type (and in
any case after 30 minutes), after which fresh bundles go back to the L40S — but nothing
evicts a running A10G to make room for one. A fleet whose demand is already satisfied by
fallback nodes stays on them until those nodes idle out and retire. **The fleet never gets bigger**, only differently made: `num_actors` bounds
it exactly as before.

Three things to know before you turn it on.

**It costs double the quota per GPU.** The G-and-VT quota is counted in vCPU. These
sizes are 8 vCPU per GPU against `g6e.xlarge`'s 4, so the applied 10,000 vCPU buys
either 2,500 L40S or 1,250 A10G. A fleet fully on fallback is half the card count
*before* the 0.46 throughput factor.

**A cluster that fails over stays failed over.** Recovery applies to new demand only.
Once the fallback has satisfied every actor bundle there is no unfulfilled demand, so
the scorer is not consulted and nothing evicts a running A10G to make room for an L40S
— the fleet holds its composition until those actors go idle and retire. Under the
chained-cluster strategy actors are long-lived, so a cluster that failed over early can
spend its whole roster on the 0.46-throughput card at twice the quota per GPU.

That is deliberate rather than an oversight: draining healthy workers mid-chunk to trade
cards would discard the work in flight. But it is a real operational cost, and the
practical remedy is that **the next cluster starts fresh** — ceilings and preference are
resolved per provision. If a long-running cluster is stuck on the fallback while L40S
capacity has visibly returned, the lever is to let that fill finish rather than to fight
it. An explicit probe-and-drain would be a feature, and is not here.

**Only genuine capacity refusals move the fleet.** A throttle (`RequestLimitExceeded`)
clears in seconds, and a quota refusal (`InstanceLimitExceeded`) only gets worse on a
rung that spends more quota — so neither demotes.

**The configuration the campaign runs.** As of 2026-08-28, with the L40S in short
supply:

```
gpu_fallback_instance_types = ["g5.2xlarge"]
gpu_fallback_vcpu_budget    = 840
gpu-worker-ladder (SSM)     = g6e.xlarge:101
```

That yields ceilings of **101 L40S and 105 A10G** per cluster. The L40S ceiling sits
well above the ~39 AWS is currently supplying, which costs nothing unclaimed and
converts straight into more actors per vCPU if supply recovers — an L40S actor is half
the quota of an A10G one. The A10G ceiling is what actually decides the bill while the
L40S trickles: `39x4 + 105x8 = 996` vCPU per cluster, **9,960 across ten, inside the
10,000 account quota by design** rather than by AWS refusing the last launches.

Both rungs full would be 1,244 vCPU per cluster and over quota. That is the accepted
limitation — Ray's ceilings count nodes and cannot be jointly weighted — and AWS
enforces the real line by refusing, which the scorer correctly declines to treat as a
reason to fall back further. `TestTheCampaignRestartConfiguration` pins all of it.

**The vCPU-matched sizes are deliberately not offered.** `g5.xlarge` and `g6.xlarge` are
4 vCPU per GPU like the production rung, but carry 16 GiB of host RAM against a measured
~17.7 GB per-actor requirement — the exact shape that OOMed the loader on the earlier
16 GB g5-class workers.

The L4 is half the L40S's VRAM. What makes it arguable at all is the per-chunk
peak-VRAM telemetry on the `CHUNK_SUMMARY` line: `max_memory_allocated` measured
**4.6–7.5 GiB** at optical depths of 54–113 timesteps, against the ~43 GiB the
earlier `nvidia-smi` reading implied. That reading was the caching allocator's
*reserved* pool, which runs ~3× the live requirement and sizes itself to the card
it is given. Read `vram_peak_gib` against `t_kept` before trusting any card-fit
argument: the requirement grows with optical depth.

## Region

Default is `us-west-2`. The Sentinel-1 OPERA RTC archive is hosted
in us-west-2 and **only authenticates from in-region clients** (not
just an egress-cost preference — cross-region requests are
**rejected**). If you operate elsewhere, expect S1 ingest to fail
until you stand up a us-west-2 footprint.

`zarr_store._create_storage` defaults its S3 region to us-west-2 for
the same reason; override per-bucket if you have a multi-region
setup.

## Costs

Order-of-magnitude:

```
Inference, 5 km × 5 km ROI, 1 month of S2 + S1:
─────────────────────────────────────────────────
ROI generation                            $0.001    (Fargate, ~5 s)
S2 ingest    (Fargate, ~10 min × 4 wkrs)  $0.10
S1 ingest    (Fargate, ~10 min × 4 wkrs)  $0.10
Inference    (1 g6e.xlarge spot × 30 min) $0.35
Assembly     (Fargate, ~5 min × 20 wkrs)  $0.30
─────────────────────────────────────────────────
Total                                     ~$0.85

Inference, 100 km × 100 km ROI:
─────────────────────────────────────────────────
~10× the chunks → ~10× the time on each step.
Ingest scales linearly; inference uses ~10–20 spot GPUs (~$2-4/run)
Total                                     ~$8-15
```

These are budgetary; profile your specific AOI before scaling
billing assumptions.

## See also

- [`src/tessera_embeddings/providers/aws/gotchas.md`](../../src/tessera_embeddings/providers/aws/gotchas.md) —
  AMI baking, teardown defence-in-depth, CloudWatch wiring,
  diagnostics shim.
- [`docs/prefect-setup.md`](../prefect-setup.md) — Prefect-server side
  of the deployment.
- [`docs/providers/adding-your-own.md`](adding-your-own.md) —
  applying these patterns to GCP, Azure, or k8s.
