"""AWS-backed Ray cluster provisioning.

The :func:`ray_cluster` context manager handles SSM config resolution,
SSH key materialisation, ``ray up`` / ``ray down`` subprocess calls,
CloudWatch agent configuration, and (optionally) code-tarball sync to
S3. No ML logic, no Prefect tasks — orchestration code imports this as a
plain context manager.

Adapted from the reference repo's ``infra/ray/cluster.py``. Notable
differences for the open-source release:

* Account-bound IDs (security groups, AMIs, IAM roles, subnets, key
  pairs, EC2 tags) are read from SSM under a configurable prefix —
  there are no hardcoded ``/yield/ray/`` paths.
* The dev/prod bucket toggle from the reference repo is removed; bucket
  names are caller-supplied.
* Source-code tarball sync is opt-in (``sync_source_path``); the default
  assumes a pre-baked AMI and skips the sync step entirely.
* CloudWatch log-group name is a parameter, not a hardcoded
  ``/ec2/yield/ray``.

# NOTE — cancellation hook dependency:
# :func:`terminate_ray_instances_by_tag` and :func:`cleanup_ray_tempfiles` are
# intended to be wired into the orchestrator's on-cancellation/on-crashed
# hooks. Those hooks are the real last line of defence: the autoscaler idle
# timeout only drains workers above a node type's ``min_workers`` floor and
# NEVER terminates the head node, so an untorn-down cluster runs until
# someone terminates it by tag. See gotchas.md ("Teardown").
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import boto3
import ray
import yaml

DEFAULT_CLUSTER_TEMPLATE = Path(__file__).parent / "cluster.yaml.template"
"""Path to the cluster YAML template shipped with this provider."""

DEFAULT_CLOUDWATCH_TEMPLATE = Path(__file__).parent / "cloudwatch-agent.json.tpl"
"""Path to the CloudWatch agent JSON template shipped with this provider."""

DEFAULT_SSM_PREFIX = "/tessera/ray/"
"""Default SSM Parameter Store prefix for Ray cluster resource IDs.

Production deployments override this — there is no global default that
makes sense across organisations. Configure the prefix to match wherever
you've stored the EC2 resource IDs your Ray nodes need.
"""

DEFAULT_CLOUDWATCH_LOG_GROUP = "/ec2/tessera/ray"
"""Default CloudWatch log group for Ray agent logs."""

RAY_DOWN_TIMEOUT_S = 300
"""Upper bound (seconds) on a cancellation-time ``ray down``.

``ray down`` SSHes into the head to tear the cluster down; if the head is
unreachable or the CLI wedges it can block forever. A cancellation hook that
hangs here never reaches the tag-based EC2 termination fallback, leaving GPU
workers running and billed — so the call is bounded and a timeout falls through
to tag termination. Generous enough for a normal teardown (~1-2 min)."""

PROJECT_TAG_VALUE = "tessera-embeddings"
"""Value of the ``Project`` EC2 tag stamped on every Ray node.

The deployment's runner IAM role conditions ``ec2:TerminateInstances`` on
``aws:ResourceTag/Project`` equal to this value, so that ``ray down`` (which
terminates nodes using the driver's credentials) and the
:func:`terminate_ray_instances_by_tag` fallback are both authorised. Without
this tag every teardown terminate is IAM-denied and the instances leak. Keep
this in lockstep with the deployment's IAM condition (yield CDK:
``ray_inference.py`` ``RayEc2Terminate``)."""

_REQUIRED_SSM_KEYS = frozenset(
    {"security-group-id", "instance-profile-arn", "private-subnet-ids", "key-pair-name", "key-pair-id"}
)


def resolve_ami_id(ami_ssm_name: str, region: str = "us-west-2") -> str:
    """Resolve the worker AMI ID the ``ami_ssm_name`` SSM parameter currently points at.

    The campaign resolves this ONCE up front and pins it into every fill's
    provisioning (``ray_cluster(ami_id=...)``), so a re-bake that repoints the SSM
    parameter mid-campaign can't make a fill boot a different image than the one its
    staging fingerprint recorded — see :func:`resolve_code_artifact_identity`.
    """
    return boto3.client("ssm", region_name=region).get_parameter(Name=ami_ssm_name)["Parameter"]["Value"]


def resolve_code_artifact_identity(
    ami_ssm_name: str,
    code_bucket: str | None = None,
    code_suffix: str = "",
    region: str = "us-west-2",
    ami_id: str | None = None,
) -> str:
    """Immutable identity of the code a Ray fill will run, for the staging fingerprint.

    Returns ``ami=<ami-id>`` and, when a source tarball overlays the AMI, appends
    ``|tarball=<etag>`` — the same two artifacts :func:`provision_ray_cluster` boots
    from (the AMI behind ``ami_ssm_name`` and ``s3://{code_bucket}/code/src{suffix}.tar.gz``).

    The global campaign folds this into each cell's staging ``run_id`` because
    ``code_suffix`` alone is NOT immutable: it is empty for a baked production AMI and
    only a filename/branch stem for a tarball, so re-baking the AMI under the same SSM
    name, or overwriting the tarball, leaves it unchanged. A retry would then resume
    tiles staged by the OLD code while remaining tiles run the NEW code, permanently
    publishing a mixed-version year. Resolving the real AMI ID and tarball ETag makes
    any code change flip the fingerprint, so a fresh staging prefix is used.

    KNOWN RESIDUAL WINDOW (dev-overlay path only). The ETag is read here, once, while
    workers later download the mutable key ``code/src{code_suffix}.tar.gz``. Overwrite
    that object mid-campaign and workers boot code the fingerprint does not describe.
    Re-reading the ETag just before launch would narrow the window, not close it — the
    overwrite can land between that HEAD and the worker's GET — so this is left as a
    constraint rather than a partial mitigation: DO NOT overwrite a tarball a campaign
    is running against. Production is unaffected (a baked AMI passes ``code_bucket=None``
    and has no tarball term at all). To close it properly, upload content-addressed keys
    (``code/src-<sha>.tar.gz``) so the object is immutable by construction, rather than
    threading an S3 versionId through provisioning. Same reasoning as the model
    checkpoint's filename-not-bytes identity in ``_staging_run_id``.

    Args:
        ami_ssm_name: SSM parameter holding the worker AMI ID.
        code_bucket: S3 bucket of the source tarball; ``None`` for a pure-AMI deploy.
        code_suffix: Tarball filename suffix (``code/src{code_suffix}.tar.gz``).
        region: AWS region for the SSM/S3 clients (the store's region; us-west-2 default).
        ami_id: A pre-resolved AMI ID to fingerprint instead of reading ``ami_ssm_name``.
            Pass the SAME id that provisioning is pinned to (``ray_cluster(ami_id=...)``)
            so the fingerprint and the booted image are guaranteed identical — a caller
            that pins provisioning must pin the fingerprint too, or the two could resolve
            the SSM pointer at different instants and disagree.
    """
    parts = [f"ami={ami_id if ami_id is not None else resolve_ami_id(ami_ssm_name, region)}"]
    if code_bucket:
        s3 = boto3.client("s3", region_name=region)
        etag = s3.head_object(Bucket=code_bucket, Key=f"code/src{code_suffix}.tar.gz")["ETag"].strip('"')
        parts.append(f"tarball={etag}")
    return "|".join(parts)


def cluster_name_for_flow_run(flow_run_id: object, cluster_yaml: Path = DEFAULT_CLUSTER_TEMPLATE) -> str | None:
    """Deterministic Ray cluster name for a flow run, or ``None`` if no id is known.

    The name must be recomputable from nothing but the flow-run id: Prefect runs
    cancellation/crash hooks in a freshly imported module after the flow's child
    process is killed, so a hook can re-derive the cluster tag (and terminate the
    fleet) even with the flow's module globals unset. Both the ``tessera_embeddings``
    and ``fill-zone-year`` flows pass this as ``ray_cluster(cluster_name=...)`` so the
    provisioned name and the hook's re-derived name always match. The base comes from
    the shipped cluster template so it stays in sync with what ``ray up`` uses.
    """
    if not flow_run_id:
        return None
    with cluster_yaml.open() as f:
        base = yaml.safe_load(f).get("cluster_name", "tessera-inference")
    return f"{base}-{str(flow_run_id).replace('-', '')[:8]}"


def _build_cloudwatch_setup_command(
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
) -> str:
    """Build a shell command that configures the CloudWatch agent on a Ray node.

    Reads the human-readable JSON template, compacts it to a single line,
    substitutes the EC2 instance ID at boot, and writes the resulting
    config in place. Heredocs in YAML setup_commands break when Ray
    sends them over SSH (indented terminators are never matched), so we
    inline the whole config as a single shell command.

    Args:
        cloudwatch_template: Path to the JSON template. Defaults to the
            template shipped with this provider.
        log_group: CloudWatch log group name to receive Ray logs.
    """
    with cloudwatch_template.open() as f:
        template = json.load(f)
    compact = json.dumps(template, separators=(",", ":"))
    compact = compact.replace("__LOG_GROUP__", log_group)
    # Escape single quotes so embedding in a single-quoted shell string is safe.
    compact = compact.replace("'", "'\\''")
    return (
        "sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc"
        " && INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
        f" && echo '{compact}' | sed \"s/__INSTANCE_ID__/$INSTANCE_ID/g\""
        " | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null"
        " && sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"
        " -a fetch-config -m ec2 -s"
        " -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"
        ' || echo "WARNING: CloudWatch agent not available — logs will not ship to CloudWatch" >&2'
    )


def _resolve_ray_config(
    cluster_yaml: str | Path,
    *,
    region: str = "us-west-2",
    ami_ssm_name: str,
    ami_id: str | None = None,
    ssm_prefix: str = DEFAULT_SSM_PREFIX,
    cluster_name: str | None = None,
    instance_tags: list[dict[str, str]] | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    cloudwatch_log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    idle_timeout_minutes: int | None = None,
) -> str:
    """Inject AWS resource IDs from SSM into a Ray cluster YAML template.

    Reads SSM parameters under ``ssm_prefix``, looks up the AMI ID from
    ``ami_ssm_name``, lists every private subnet on every node type (multi-AZ
    with launch-time capacity failover — see the subnet block below),
    materialises the SSH key from SSM into a tempfile, injects the
    CloudWatch agent setup command, substitutes ``{CODE_BUCKET}`` and
    ``{CODE_SUFFIX}`` in setup_commands, and writes the resolved config to
    a tempfile.

    Args:
        cluster_yaml: Path to the cluster YAML template. Use
            :data:`DEFAULT_CLUSTER_TEMPLATE` for the bundled template.
        region: AWS region for SSM and EC2 clients.
        ami_ssm_name: SSM parameter name holding the worker AMI ID. Used only
            when ``ami_id`` is not given.
        ami_id: A pre-resolved worker AMI ID that PINS the image, bypassing the
            ``ami_ssm_name`` lookup. The campaign resolves the AMI once and threads
            it here so a mid-campaign re-bake can't repoint the SSM parameter and
            boot a different image than a fill's staging fingerprint recorded.
        ssm_prefix: Prefix under which Ray resource IDs are stored.
            Required keys: ``security-group-id``, ``instance-profile-arn``,
            ``private-subnet-ids``, ``key-pair-name``, ``key-pair-id``.
        cluster_name: Override the template's ``cluster_name``. Required
            for running multiple clusters concurrently.
        instance_tags: Extra EC2 tags to apply to every node, on top of
            the always-present ``Project`` tag (see :data:`PROJECT_TAG_VALUE`)
            and Ray's own ``ray-cluster-name`` tag. List of
            ``{"Key": str, "Value": str}`` dicts; a ``Project`` key here
            overrides the default. ``None`` means only the defaults.
        code_bucket: S3 bucket name (without ``s3://``) substituted for
            ``{CODE_BUCKET}`` in setup_commands. ``None`` leaves the
            placeholder; pair with ``sync_source_path=None`` to disable
            tarball sync entirely.
        code_suffix: Substituted for ``{CODE_SUFFIX}``. Empty string for
            production tarballs; ``"-mybranch"`` for dev branches.
        cloudwatch_log_group: CloudWatch log group for Ray agent logs.
        cloudwatch_template: Path to the CloudWatch agent JSON template.
        idle_timeout_minutes: Override the template's autoscaler idle-down
            delay. The template default (2 min) suits single-ROI runs, where
            any idle worker is surplus; a multi-zone sequential fill holds one
            cluster across zones and must survive the inter-zone gap
            (staged-completeness verify + next zone's dispatch), so it passes
            a larger value. ``None`` keeps the template's value.

    Returns:
        Path to the resolved YAML tempfile.

    Raises:
        RuntimeError: If required SSM parameters are missing or SSH key
            material cannot be retrieved.
    """
    cluster_yaml = Path(cluster_yaml)
    with cluster_yaml.open() as f:
        config = yaml.safe_load(f)

    if cluster_name:
        config["cluster_name"] = cluster_name
    if idle_timeout_minutes is not None:
        config["idle_timeout_minutes"] = idle_timeout_minutes

    ssm = boto3.client("ssm", region_name=region)

    # Fetch all params under the prefix in one call
    params: dict[str, str] = {}
    paginator = ssm.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=ssm_prefix, Recursive=True):
        for p in page["Parameters"]:
            key = p["Name"].rsplit("/", 1)[-1]
            params[key] = p["Value"]

    missing = _REQUIRED_SSM_KEYS - params.keys()
    if missing:
        msg = f"Missing required SSM parameters under {ssm_prefix!r}: {sorted(missing)}"
        raise RuntimeError(msg)

    sg_ids = [params["security-group-id"]]
    all_subnet_ids = [s.strip() for s in params["private-subnet-ids"].split(",")]
    iam_profile = {"Arn": params["instance-profile-arn"]}
    key_name = params["key-pair-name"]
    # Always stamp the Project tag so teardown terminates are IAM-authorised
    # (see PROJECT_TAG_VALUE). Caller-supplied tags win on key collision.
    merged_tags = [{"Key": "Project", "Value": PROJECT_TAG_VALUE}]
    caller_keys = {t["Key"] for t in (instance_tags or [])}
    if "Project" in caller_keys:
        merged_tags = []
    merged_tags.extend(instance_tags or [])
    tag_specs: list[dict[str, Any]] = [{"ResourceType": "instance", "Tags": merged_tags}]

    # Every node type gets ALL private subnets, in SSM-param order. Ray's AWS
    # node provider launches in the FIRST listed subnet and rotates to the
    # next on a launch ClientError (e.g. InsufficientInstanceCapacity),
    # trying every subnet before giving up — so the fleet lands mostly in one
    # AZ with automatic capacity spillover to the others (a 2026-07-17 run
    # stalled at 2 workers when its pinned AZ ran out of g6e capacity).
    # Cross-AZ exposure is negligible by construction: inference's bulk data
    # plane is actor↔S3 only (free in-region via the subnets' S3 gateway
    # endpoints — verify routes when subnets change, see gotchas.md) and
    # head↔worker traffic is KB/s control RPCs. INVARIANT that keeps this
    # cheap: Ray actors must never exchange bulk data node-to-node — all bulk
    # I/O goes to S3. (The Dask provider keeps its single-AZ pin; Dask
    # genuinely shuffles between workers.)
    ec2 = boto3.client("ec2", region_name=region)
    subnet_resp = ec2.describe_subnets(SubnetIds=all_subnet_ids)
    if not subnet_resp.get("Subnets"):
        msg = f"No subnets found for IDs: {all_subnet_ids}"
        raise RuntimeError(msg)
    # describe_subnets returns arbitrary order; order the AZ list to match the
    # SSM subnet order (= launch-preference order), deduping shared AZs.
    az_by_subnet = {s["SubnetId"]: s["AvailabilityZone"] for s in subnet_resp["Subnets"]}
    azs = list(dict.fromkeys(az_by_subnet[sid] for sid in all_subnet_ids))
    config["provider"]["availability_zone"] = ",".join(azs)

    # Prefer a caller-PINNED AMI ID (the campaign resolves it once and threads it
    # through every fill) over re-reading the SSM pointer here: re-reading would let
    # a mid-campaign re-bake boot a different image than the fill's staging
    # fingerprint recorded. Fall back to the SSM lookup when unpinned (direct/dev
    # invocations), where the pointer is authoritative.
    resolved_ami_id = ami_id if ami_id is not None else ssm.get_parameter(Name=ami_ssm_name)["Parameter"]["Value"]

    for node_type_cfg in config["available_node_types"].values():
        nc = node_type_cfg["node_config"]
        nc["ImageId"] = resolved_ami_id
        nc["KeyName"] = key_name
        nc["SecurityGroupIds"] = sg_ids
        nc["IamInstanceProfile"] = iam_profile
        nc["SubnetIds"] = list(all_subnet_ids)
        if tag_specs:
            nc["TagSpecifications"] = tag_specs

    # Retrieve SSH private key from SSM (stored by AWS at key-pair creation time)
    key_pair_id = params["key-pair-id"]
    try:
        key_resp = ssm.get_parameter(Name=f"/ec2/keypair/{key_pair_id}", WithDecryption=True)
        ssh_key_material = key_resp["Parameter"]["Value"]
    except ssm.exceptions.ParameterNotFound as exc:
        msg = f"SSH key not found in SSM at /ec2/keypair/{key_pair_id}"
        raise RuntimeError(msg) from exc

    ssh_key_fd, ssh_key_path = tempfile.mkstemp(prefix="ray_ssh_", suffix=".pem")
    with os.fdopen(ssh_key_fd, "w") as kf:
        kf.write(ssh_key_material)
    Path(ssh_key_path).chmod(stat.S_IRUSR)  # 0o400

    config["auth"]["ssh_private_key"] = ssh_key_path

    # Inject the CloudWatch agent setup command (replaces any heredoc-style
    # cloudwatch entry in the template, which would break over SSH).
    #
    # Append it to the *_start_ray_commands, NOT setup_commands: setup_commands
    # run before `ray start`, so the agent would resolve its file paths while
    # /tmp/ray/session_latest is empty or stale, then `ray start` repoints the
    # session_latest symlink out from under it and no logs ship. Starting the
    # agent after `ray start` guarantees the session and its log files already
    # exist when fetch-config discovers them. Strip any pre-existing cloudwatch
    # entries from every command list so we don't double-start.
    cw_cmd = _build_cloudwatch_setup_command(cloudwatch_template, cloudwatch_log_group)
    for key in ("setup_commands", "head_start_ray_commands", "worker_start_ray_commands"):
        config[key] = [cmd for cmd in config.get(key, []) if "cloudwatch" not in str(cmd).lower()]
    config["head_start_ray_commands"].append(cw_cmd)
    config["worker_start_ray_commands"].append(cw_cmd)

    # Substitute {CODE_BUCKET} and {CODE_SUFFIX} in setup_commands.
    if code_bucket is not None:
        config["setup_commands"] = [
            cmd.replace("{CODE_BUCKET}", code_bucket).replace("{CODE_SUFFIX}", code_suffix)
            if isinstance(cmd, str)
            else cmd
            for cmd in config["setup_commands"]
        ]

    resolved_fd, resolved_path = tempfile.mkstemp(prefix="ray_cluster_", suffix=".yaml")
    with os.fdopen(resolved_fd, "w") as rf:
        yaml.dump(config, rf, default_flow_style=False)

    return resolved_path


def cleanup_ray_tempfiles(resolved_yaml: str | None) -> None:
    """Best-effort cleanup of resolved YAML and SSH key tempfiles.

    Safe to call from an on-cancellation hook: silently swallows all
    errors so a partially-resolved cluster doesn't block cancellation.
    """
    if not resolved_yaml:
        return
    with contextlib.suppress(Exception):
        with Path(resolved_yaml).open() as f:
            config = yaml.safe_load(f)
        ssh_key_path = config.get("auth", {}).get("ssh_private_key")
        if ssh_key_path:
            Path(ssh_key_path).unlink(missing_ok=True)
    with contextlib.suppress(Exception):
        Path(resolved_yaml).unlink(missing_ok=True)


def _sync_code_to_s3(
    src_dir: Path,
    s3_bucket: str,
    s3_key: str,
) -> None:
    """Tar a source directory and upload it to S3.

    Workers pull this tarball on startup instead of using Ray
    ``file_mounts``, which depends on SSH-based rsync and bottlenecks at
    100-500+ workers. The S3 download is parallel across all workers.

    Args:
        src_dir: Directory to package.
        s3_bucket: Destination S3 bucket name (no ``s3://`` prefix).
        s3_key: S3 key for the tarball (e.g. ``"code/src.tar.gz"``).
    """
    tarball_fd, tarball = tempfile.mkstemp(suffix=".tar.gz", prefix="src_sync_")
    os.close(tarball_fd)
    try:
        subprocess.run(
            ["tar", "-czf", tarball, "-C", str(src_dir.parent), src_dir.name],
            check=True,
        )
        boto3.client("s3").upload_file(tarball, s3_bucket, s3_key)
    finally:
        Path(tarball).unlink(missing_ok=True)


def _start_ray_cluster(
    resolved_yaml: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> str:
    """Launch a Ray cluster via ``ray up`` on a resolved YAML; return the head IP.

    Config resolution happens in the caller (:func:`ray_cluster`) so the
    resolved path is already bound when a failed launch unwinds to the
    teardown block — ``ray down`` must target the real (uuid-suffixed)
    cluster, not the unresolved template.
    """
    log.info("Starting Ray cluster from %s", resolved_yaml)
    ray_up = subprocess.run(
        ["ray", "up", resolved_yaml, "-y", "--no-config-cache"],
        capture_output=True,
        text=True,
    )
    if ray_up.returncode != 0:
        log.error("ray up failed (exit %d)", ray_up.returncode)
        log.error("ray up stdout:\n%s", ray_up.stdout[-5000:] if ray_up.stdout else "(empty)")
        log.error("ray up stderr:\n%s", ray_up.stderr[-5000:] if ray_up.stderr else "(empty)")
        msg = f"ray up failed with exit code {ray_up.returncode}"
        raise RuntimeError(msg)
    log.info("Ray cluster started")

    result = subprocess.run(
        ["ray", "get-head-ip", resolved_yaml],
        capture_output=True,
        text=True,
        check=True,
    )
    # ray get-head-ip emits log lines before the IP; take the last line
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("ray get-head-ip returned no output")
    head_ip = lines[-1].strip()
    log.info("Ray head node IP: %s", head_ip)

    return head_ip


def _log_ray_dashboard_ssm_command(
    cluster_name: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    region: str,
) -> None:
    """Log a copy-pasteable SSM command to port-forward the Ray dashboard.

    Finds the head node by its ``ray-cluster-name`` + ``ray-node-type=head``
    tags and emits an ``aws ssm start-session`` block that forwards the
    dashboard (port 8265) to ``localhost``. The head listens on 8265 with
    ``--dashboard-host=0.0.0.0`` and the EC2 role carries
    ``AmazonSSMManagedInstanceCore``, so the port-forward works against the
    instance ID.

    Uses ``AWS-StartPortForwardingSession``, which targets a port on the
    managed instance itself. The ``...ToRemoteHost`` variant addresses hosts
    reachable *from* the instance, and the SSM agent rejects loopback
    destinations for it.

    ``--region`` is filled in from the region the head node was looked up in,
    so an operator whose default region differs doesn't get a "target not
    connected" from SSM. No ``--profile`` is printed — credentials come from
    the operator's environment (``AWS_PROFILE`` or their default).

    Best-effort: warns and returns on any failure, never raises.

    Args:
        cluster_name: Resolved, unique ``ray-cluster-name`` tag value (with
            the uuid8 suffix) that Ray actually tagged instances with.
        log: Logger.
        region: AWS region.
    """
    try:
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:ray-cluster-name", "Values": [cluster_name]},
                {"Name": "tag:ray-node-type", "Values": ["head"]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ],
        )
        reservations = resp.get("Reservations", [])
        instances = reservations[0].get("Instances", []) if reservations else []
        if not instances:
            log.warning("Could not find Ray head node for dashboard command")
            return
        instance_id = instances[0]["InstanceId"]
        log.info(
            "To view the Ray dashboard, run:\n\n"
            "aws ssm start-session \\\n"
            f"  --target {instance_id} \\\n"
            "  --document-name AWS-StartPortForwardingSession \\\n"
            f"  --region {region} \\\n"
            '  --parameters \'{"portNumber":["8265"],"localPortNumber":["8265"]}\'\n\n'
            "Then open http://localhost:8265"
        )
    except Exception:
        log.warning("Could not generate Ray dashboard command", exc_info=True)


def make_instance_terminator(
    region: str = "us-west-2",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> Callable[[str], None]:
    """Create a callback that terminates a single EC2 instance by ID.

    Wired into the inference scheduler's ``on_actor_retire`` hook so that
    GPU nodes are terminated immediately after retiring idle actors,
    rather than waiting for the Ray autoscaler's idle timeout (which is
    unreliable after ``ray.kill()`` because it relies on the node
    self-reporting empty).

    Args:
        region: AWS region.
        log: Logger for termination events.
    """
    _log = log or logging.getLogger(__name__)
    ec2 = boto3.client("ec2", region_name=region)

    def _terminate(instance_id: str) -> None:
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            _log.info("Terminated EC2 instance %s", instance_id)
        except Exception:
            _log.warning("Failed to terminate EC2 instance %s", instance_id, exc_info=True)

    return _terminate


def terminate_ray_instances_by_tag(
    cluster_name: str,
    *,
    region: str = "us-west-2",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    prefix_match: bool = False,
) -> None:
    """Terminate all running/pending EC2 instances belonging to a Ray cluster.

    Fallback used when the resolved cluster YAML is unavailable (e.g. the
    flow was cancelled before ``ray up`` wrote it). Finds instances by
    the ``ray-cluster-name`` tag that Ray sets on every node it launches.

    Args:
        cluster_name: Value (or prefix) of the ``ray-cluster-name`` EC2 tag.
        region: AWS region.
        log: Optional logger; silently swallows errors if ``None``.
        prefix_match: If True, match clusters whose ``ray-cluster-name``
            *starts with* ``cluster_name``. Use with caution — this will
            terminate instances from ALL matching clusters.
    """
    _log = log or logging.getLogger(__name__)
    try:
        ec2 = boto3.client("ec2", region_name=region)
        tag_value = f"{cluster_name}*" if prefix_match else cluster_name
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:ray-cluster-name", "Values": [tag_value]},
                {"Name": "instance-state-name", "Values": ["running", "pending"]},
            ],
        )
        instance_ids = [i["InstanceId"] for r in resp.get("Reservations", []) for i in r.get("Instances", [])]
        if instance_ids:
            _log.info("Terminating %d Ray instances: %s", len(instance_ids), instance_ids)
            ec2.terminate_instances(InstanceIds=instance_ids)
        else:
            _log.info("No running Ray instances found for cluster '%s'", cluster_name)
    except Exception:
        _log.exception("Failed to terminate Ray instances by tag")


def _stop_ray_cluster(
    cluster_yaml: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> bool:
    """Tear down the Ray cluster via ``ray down``. Best-effort — does not raise.

    Returns True if ``ray down`` exited 0, False otherwise (caller may then fall
    back to tag-based termination so a head that ``ray down`` couldn't reach
    doesn't leak).

    BOUNDED by ``RAY_DOWN_TIMEOUT_S``, for the reason that constant documents: this
    SSHes into the head, and an unreachable head makes it hang indefinitely. Unbounded,
    the fallback in the sentence above is unreachable exactly when it is needed — the
    call never returns, so nothing terminates the fleet by tag and the GPUs bill on. A
    timeout is therefore reported as a failed ``ray down``, not raised.
    """
    log.info("Tearing down Ray cluster")
    try:
        result = subprocess.run(["ray", "down", cluster_yaml, "-y"], check=False, timeout=RAY_DOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning(
            "ray down did not finish within %ds — treating as failed so the caller falls "
            "back to terminating instances by ray-cluster-name tag.",
            RAY_DOWN_TIMEOUT_S,
        )
        return False
    if result.returncode == 0:
        log.info("Ray cluster stopped")
        return True
    log.warning(
        "ray down exited with code %d — cluster may still be running; idle workers "
        "self-drain after the idle timeout, but the head node does NOT self-terminate. "
        "Terminate by ray-cluster-name tag if this persists.",
        result.returncode,
    )
    return False


@contextlib.contextmanager
def ray_cluster(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    ami_ssm_name: str,
    ami_id: str | None = None,
    ray_address: str | None = None,
    cluster_yaml: str | Path | None = None,
    cluster_name: str | None = None,
    region: str = "us-west-2",
    ssm_prefix: str = DEFAULT_SSM_PREFIX,
    instance_tags: list[dict[str, str]] | None = None,
    sync_source_path: Path | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    cloudwatch_log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    idle_timeout_minutes: int | None = None,
) -> Iterator[str | None]:
    """Provision an AWS-backed Ray cluster; tear it down on exit.

    Two modes:

    * ``ray_address`` set — connect to an existing cluster at that
      address. No ``ray up``/``ray down`` is performed; the caller owns
      the lifecycle.
    * Default — call ``ray up`` against the resolved cluster YAML, then
      ``ray.init`` against the head node, and tear the cluster down on
      exit.

    Args:
        log: Logger.
        ami_ssm_name: SSM parameter name holding the worker AMI ID.
            Required even when ``ray_address`` is given (kept as a
            consistent signature; ignored in that path). Used only when
            ``ami_id`` is not given.
        ami_id: Pre-resolved AMI ID that PINS the worker image (bypasses the
            ``ami_ssm_name`` lookup). The campaign threads the AMI it resolved
            once into every fill so a mid-campaign re-bake can't boot a
            different image than the fill's staging fingerprint recorded.
        ray_address: Connect to an existing cluster instead of launching one.
        cluster_yaml: Path to the cluster YAML template. Defaults to the
            template shipped at :data:`DEFAULT_CLUSTER_TEMPLATE`.
        cluster_name: Override the YAML's ``cluster_name``. When ``None``,
            an auto-generated name is used so concurrent clusters get
            distinct EC2 tags.
        region: AWS region (must match the SSM region).
        ssm_prefix: SSM Parameter Store prefix for Ray resource IDs.
        instance_tags: EC2 tags applied to every node.
        sync_source_path: When provided, tar this directory and upload it
            to ``s3://{code_bucket}/code/src{code_suffix}.tar.gz`` before
            ``ray up``. Use for dev iteration; production deployments
            should bake source into the AMI and leave this ``None``.
        code_bucket: S3 bucket for the source tarball. Required when
            ``sync_source_path`` is set.
        code_suffix: Filename suffix for the tarball.
        cloudwatch_log_group: CloudWatch log group for Ray agent logs.
        cloudwatch_template: CloudWatch agent JSON template path.
        idle_timeout_minutes: Optional override of the template's autoscaler
            idle-down delay; ``None`` keeps the template's value. Rationale at
            :func:`_resolve_ray_config`.

    Yields:
        Path to the resolved cluster YAML tempfile when this context
        manages the cluster lifecycle, or ``None`` when ``ray_address``
        was supplied. The yielded path is intended for an
        on-cancellation hook to call :func:`cleanup_ray_tempfiles`.
    """
    resolved_yaml: str | None = None
    manages_cluster = False
    cluster_yaml = Path(cluster_yaml) if cluster_yaml is not None else DEFAULT_CLUSTER_TEMPLATE
    try:
        if ray_address:
            log.info("Connecting to Ray at %s", ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            manages_cluster = True
            if cluster_name is None:
                with cluster_yaml.open() as f:
                    base_name = yaml.safe_load(f).get("cluster_name", "tessera-inference")
                suffix = uuid.uuid4().hex[:8]
                cluster_name = f"{base_name}-{suffix}"
            log.info("Using cluster name: %s", cluster_name)

            if sync_source_path is not None:
                if not code_bucket:
                    raise ValueError("code_bucket is required when sync_source_path is set")
                s3_key = f"code/src{code_suffix}.tar.gz"
                log.info("Syncing %s → s3://%s/%s", sync_source_path, code_bucket, s3_key)
                _sync_code_to_s3(sync_source_path, code_bucket, s3_key)

            # Resolve BEFORE launching so `resolved_yaml` is bound when a
            # failed `ray up` unwinds to the finally-block: a partial launch
            # can leave a provisioned head behind, and `ray down` against the
            # unresolved template (whose cluster_name lacks the uuid suffix)
            # matches nothing — that exact path leaked a head on 2026-07-16.
            # _resolve_ray_config lists every private subnet on every node
            # type (multi-AZ with launch-time capacity failover).
            log.info("Resolving Ray cluster config from SSM (cluster_name=%s)", cluster_name)
            resolved_yaml = _resolve_ray_config(
                cluster_yaml,
                region=region,
                ami_ssm_name=ami_ssm_name,
                ami_id=ami_id,
                ssm_prefix=ssm_prefix,
                cluster_name=cluster_name,
                instance_tags=instance_tags,
                code_bucket=code_bucket,
                code_suffix=code_suffix,
                cloudwatch_log_group=cloudwatch_log_group,
                cloudwatch_template=cloudwatch_template,
                idle_timeout_minutes=idle_timeout_minutes,
            )
            head_ip = _start_ray_cluster(resolved_yaml, log)
            head_address = f"ray://{head_ip}:10001"
            log.info("Connecting to Ray at %s", head_address)
            ray.init(address=head_address, ignore_reinit_error=True)

            _log_ray_dashboard_ssm_command(cluster_name, log, region=region)

        yield resolved_yaml
    finally:
        ray.shutdown()
        if manages_cluster:
            if resolved_yaml is None:
                # Config resolution failed before launch, so no resolved YAML
                # exists. `ray down` on the UNRESOLVED template would target its
                # base cluster_name (no flow-specific uuid suffix) and could tear
                # down an unrelated `tessera-inference` cluster — skip it and
                # terminate anything tagged with our exact cluster_name instead.
                if cluster_name:
                    terminate_ray_instances_by_tag(cluster_name=cluster_name, region=region, log=log)
            elif not _stop_ray_cluster(resolved_yaml, log) and cluster_name:
                # When `ray down` can't tear the cluster down (unreachable head,
                # stale YAML), fall back to exact-tag termination so a normally-
                # completed run can't leave the fleet billing. The
                # cancellation/crash hook only covers cancelled/crashed flows,
                # not this path.
                terminate_ray_instances_by_tag(cluster_name=cluster_name, region=region, log=log)
            cleanup_ray_tempfiles(resolved_yaml)
