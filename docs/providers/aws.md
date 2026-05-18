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
- s3:* on your input/output/preprocessed buckets       → for fsspec operations
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
workers:     g5.2xlarge          (8 vCPU, 32 GB,    1 InferenceActor each
                                  1× A10G 24 GB,
                                  450 GB NVMe)
```

`g5.2xlarge` was chosen for ~$1.21/hr at us-west-2 spot pricing,
24 GB VRAM (fits the Tessera model + working set), and 450 GB NVMe
(fast enough for `torch.load` of the checkpoint without EBS stalls).
Bigger workers don't help — single-A10G throughput is the
bottleneck.

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
Inference    (1 g5.2xlarge spot × 30 min) $0.20
Assembly     (Fargate, ~5 min × 20 wkrs)  $0.30
─────────────────────────────────────────────────
Total                                     ~$0.70

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
- [`docs/prefect-setup.md`](prefect-setup.md) — Prefect-server side
  of the deployment.
- [`docs/providers/adding-your-own.md`](adding-your-own.md) —
  applying these patterns to GCP, Azure, or k8s.
