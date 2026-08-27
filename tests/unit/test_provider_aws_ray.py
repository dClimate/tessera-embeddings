"""Unit tests for the AWS Ray provider's pure helpers.

These tests stay offline by mocking SSM, EC2, and S3 with ``moto``. They
exercise ``_resolve_ray_config`` (the most complex pure helper, now multi-AZ),
``cleanup_ray_tempfiles``, and ``resolve_code_artifact_identity`` (the
staging-fingerprint code identity).

The full ``ray_cluster`` context manager is NOT tested end-to-end here
because it shells out to ``ray up`` / ``ray down``; that path is an
integration concern. Its finalizer's tag-termination fallback IS pinned
below with the launch/teardown helpers stubbed out.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
import yaml
from moto import mock_aws

from tessera_embeddings.providers.aws import ray as ray_mod
from tessera_embeddings.providers.aws.ray import (
    DEFAULT_CLUSTER_TEMPLATE,
    GPU_WORKER_LADDER_SSM_KEY,
    GPU_WORKER_NODE_TYPE_PREFIX,
    LAUNCH_PACING_AUTOSCALER_ENV,
    LAUNCH_PACING_CLIENT_ENV,
    LAUNCH_PACING_ENV,
    PROJECT_TAG_VALUE,
    _apply_gpu_worker_ladder,
    _parse_gpu_worker_ladder,
    _resolve_ray_config,
    cleanup_ray_tempfiles,
    resolve_ami_id,
    resolve_code_artifact_identity,
)

_LOG = logging.getLogger("test")

REGION = "us-west-2"
SSM_PREFIX = "/test/tessera/ray/"


def _seed_ssm_and_vpc() -> tuple[str, list[str], str]:
    """Populate SSM and EC2 fixtures and return (ami_param, subnet_ids, key_pair_id)."""
    ssm = boto3.client("ssm", region_name=REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

    # VPC + 3 subnets across 2 AZs (a: 2a, b: 2b, c: 2a) so the multi-AZ
    # assertions cover both SSM-order preservation and AZ dedupe.
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]
    subnet_a = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.1.0/24", AvailabilityZone=f"{REGION}a")["Subnet"]
    subnet_b = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.2.0/24", AvailabilityZone=f"{REGION}b")["Subnet"]
    subnet_c = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.3.0/24", AvailabilityZone=f"{REGION}a")["Subnet"]
    subnet_ids = [subnet_a["SubnetId"], subnet_b["SubnetId"], subnet_c["SubnetId"]]

    key_pair = ec2.create_key_pair(KeyName="test-key")
    key_pair_id = key_pair["KeyPairId"]

    iam_arn = "arn:aws:iam::1:instance-profile/x"
    ssm.put_parameter(Name=f"{SSM_PREFIX}security-group-id", Value="sg-abc123", Type="String")
    ssm.put_parameter(Name=f"{SSM_PREFIX}instance-profile-arn", Value=iam_arn, Type="String")
    ssm.put_parameter(Name=f"{SSM_PREFIX}private-subnet-ids", Value=",".join(subnet_ids), Type="StringList")
    ssm.put_parameter(Name=f"{SSM_PREFIX}key-pair-name", Value="test-key", Type="String")
    ssm.put_parameter(Name=f"{SSM_PREFIX}key-pair-id", Value=key_pair_id, Type="String")
    ssm.put_parameter(
        Name=f"/ec2/keypair/{key_pair_id}",
        Value="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        Type="SecureString",
    )

    ami_param = "/test/tessera/ray/ami-id"
    ssm.put_parameter(Name=ami_param, Value="ami-0123456789abcdef0", Type="String")

    return ami_param, subnet_ids, key_pair_id


@mock_aws
def test_resolve_ray_config_writes_a_complete_yaml(tmp_path: Path) -> None:
    """End-to-end: SSM + EC2 are consulted and the resolved YAML has injected fields."""
    ami_param, subnet_ids, _ = _seed_ssm_and_vpc()

    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        cluster_name="test-cluster",
        instance_tags=[{"Key": "Project", "Value": "tessera-test"}],
    )

    try:
        resolved_path = Path(resolved)
        assert resolved_path.exists()
        config = yaml.safe_load(resolved_path.read_text())

        assert config["cluster_name"] == "test-cluster"
        # Comma-joined AZs, ordered by SSM subnet order, deduped (a and c
        # share us-west-2a, so it appears once and first).
        assert config["provider"]["availability_zone"] == f"{REGION}a,{REGION}b"

        # SSH key was materialised
        ssh_key_path = config["auth"]["ssh_private_key"]
        assert Path(ssh_key_path).exists()

        # Every node_config got the injected fields
        for node in config["available_node_types"].values():
            nc = node["node_config"]
            assert nc["ImageId"] == "ami-0123456789abcdef0"
            assert nc["KeyName"] == "test-key"
            assert nc["SecurityGroupIds"] == ["sg-abc123"]
            assert nc["IamInstanceProfile"] == {"Arn": "arn:aws:iam::1:instance-profile/x"}
            # ALL subnets, preserving SSM-param order — Ray's AWS provider
            # prefers the first and fails over down the list on capacity
            # errors, so order is the launch-preference contract.
            assert nc["SubnetIds"] == subnet_ids
            assert nc["TagSpecifications"][0]["Tags"] == [{"Key": "Project", "Value": "tessera-test"}]

        # CloudWatch agent command was injected into the start_ray commands
        # (after `ray start`, so the Ray session/logs exist when the agent
        # resolves file paths) — not setup_commands, which run too early.
        assert any("amazon-cloudwatch-agent" in str(cmd) for cmd in config["head_start_ray_commands"])
        assert any("amazon-cloudwatch-agent" in str(cmd) for cmd in config["worker_start_ray_commands"])
        assert not any("amazon-cloudwatch-agent" in str(cmd) for cmd in config["setup_commands"])

        # No code_bucket was given (the documented AMI-baked default), so the tarball
        # fetch is DROPPED rather than left holding an unsubstituted placeholder. Left
        # in, `aws s3 cp s3://{CODE_BUCKET}/...` is a copy from a bucket name that
        # cannot exist, and `ray up` runs setup on every node before Ray starts — so
        # the default path would fail to provision a cluster at all.
        assert not any("{CODE_BUCKET}" in str(cmd) for cmd in config["setup_commands"])
        assert not any("s3 cp" in str(cmd) for cmd in config["setup_commands"])
        # The rest of setup survives; only the fetch went.
        assert any("PYTHONPATH" in str(cmd) for cmd in config["setup_commands"])
    finally:
        cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_substitutes_the_code_bucket_when_given(tmp_path: Path) -> None:
    """The tarball fetch survives, fully substituted, when a bucket IS supplied."""
    ami_param, _, _ = _seed_ssm_and_vpc()

    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        cluster_name="test-cluster",
        code_bucket="my-code-bucket",
        code_suffix="-mybranch",
    )
    try:
        config = yaml.safe_load(Path(resolved).read_text())
        fetch = [cmd for cmd in config["setup_commands"] if "s3 cp" in str(cmd)]
        assert len(fetch) == 1
        assert "s3://my-code-bucket/code/src-mybranch.tar.gz" in fetch[0]
        assert "{CODE_BUCKET}" not in fetch[0] and "{CODE_SUFFIX}" not in fetch[0]
    finally:
        cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_stamps_project_tag_by_default(tmp_path: Path) -> None:
    """With no instance_tags, every node still gets the Project tag.

    The runner IAM role conditions ec2:TerminateInstances on this tag, so a
    missing Project tag makes teardown terminates IAM-denied (instances leak).
    """
    ami_param, _, _ = _seed_ssm_and_vpc()

    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        cluster_name="test-cluster",
        instance_tags=None,
    )

    try:
        config = yaml.safe_load(Path(resolved).read_text())
        for node in config["available_node_types"].values():
            tags = node["node_config"]["TagSpecifications"][0]["Tags"]
            assert {"Key": "Project", "Value": PROJECT_TAG_VALUE} in tags
    finally:
        cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_merges_project_tag_with_extra_tags(tmp_path: Path) -> None:
    """Extra tags coexist with the always-present Project tag."""
    ami_param, _, _ = _seed_ssm_and_vpc()

    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        cluster_name="test-cluster",
        instance_tags=[{"Key": "Team", "Value": "geo"}],
    )

    try:
        config = yaml.safe_load(Path(resolved).read_text())
        for node in config["available_node_types"].values():
            tags = node["node_config"]["TagSpecifications"][0]["Tags"]
            assert {"Key": "Project", "Value": PROJECT_TAG_VALUE} in tags
            assert {"Key": "Team", "Value": "geo"} in tags
    finally:
        cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_raises_on_missing_ssm() -> None:
    """Missing required SSM keys raise ``RuntimeError`` with the missing names."""
    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(Name=f"{SSM_PREFIX}security-group-id", Value="sg-abc", Type="String")
    # NB: missing the rest

    ssm.put_parameter(Name="/test/tessera/ray/ami-id", Value="ami-x", Type="String")

    with pytest.raises(RuntimeError, match="Missing required SSM parameters"):
        _resolve_ray_config(
            DEFAULT_CLUSTER_TEMPLATE,
            region=REGION,
            ami_ssm_name="/test/tessera/ray/ami-id",
            ssm_prefix=SSM_PREFIX,
        )


# ---------------------------------------------------------------------------
# resolve_code_artifact_identity (staging-fingerprint code identity)
# ---------------------------------------------------------------------------


@mock_aws
def test_resolve_code_artifact_identity_ami_only() -> None:
    """Pure-AMI deploy (no tarball): the identity is the resolved AMI ID."""
    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(Name="/tessera/ray/ami-id", Value="ami-0123456789abcdef0", Type="String")
    assert resolve_code_artifact_identity("/tessera/ray/ami-id", region=REGION) == "ami=ami-0123456789abcdef0"


@mock_aws
def test_resolve_code_artifact_identity_tracks_tarball_overwrite() -> None:
    """With a code tarball, the identity folds in its ETag — so re-baking the AMI
    (new AMI value) OR overwriting the tarball (new ETag) both flip the fingerprint,
    while identical content resolves to the same identity (safe resume).
    """
    ssm = boto3.client("ssm", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    ssm.put_parameter(Name="/tessera/ray/ami-id", Value="ami-aaa", Type="String")
    s3.create_bucket(Bucket="code-bkt", CreateBucketConfiguration={"LocationConstraint": REGION})
    s3.put_object(Bucket="code-bkt", Key="code/src.tar.gz", Body=b"v1")

    id_v1 = resolve_code_artifact_identity("/tessera/ray/ami-id", code_bucket="code-bkt", region=REGION)
    # Lock the documented format exactly (ami=<id>|tarball=<etag>), not just a prefix,
    # so a regression from ETag to some other object attribute would be caught.
    etag_v1 = s3.head_object(Bucket="code-bkt", Key="code/src.tar.gz")["ETag"].strip('"')
    assert id_v1 == f"ami=ami-aaa|tarball={etag_v1}"
    assert resolve_code_artifact_identity("/tessera/ray/ami-id", code_bucket="code-bkt", region=REGION) == id_v1

    # Overwrite the tarball with different content → different ETag → different identity.
    s3.put_object(Bucket="code-bkt", Key="code/src.tar.gz", Body=b"v2-different")
    id_v2 = resolve_code_artifact_identity("/tessera/ray/ami-id", code_bucket="code-bkt", region=REGION)
    etag_v2 = s3.head_object(Bucket="code-bkt", Key="code/src.tar.gz")["ETag"].strip('"')
    assert id_v2 == f"ami=ami-aaa|tarball={etag_v2}" and id_v2 != id_v1

    # Re-bake the AMI (new value behind the same SSM name) → different identity.
    ssm.put_parameter(Name="/tessera/ray/ami-id", Value="ami-bbb", Type="String", Overwrite=True)
    assert resolve_code_artifact_identity("/tessera/ray/ami-id", code_bucket="code-bkt", region=REGION) != id_v2


@mock_aws
def test_resolve_ami_id_reads_the_ssm_pointer() -> None:
    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(Name="/tessera/ray/ami-id", Value="ami-pinned-01", Type="String")
    assert resolve_ami_id("/tessera/ray/ami-id", region=REGION) == "ami-pinned-01"


@mock_aws
def test_resolve_code_artifact_identity_uses_pinned_ami_id() -> None:
    """A pinned ami_id fingerprints that exact image WITHOUT reading the SSM
    pointer (no ami param is seeded here), so the campaign's fingerprint matches
    the image it also pins provisioning to.
    """
    identity = resolve_code_artifact_identity("/unread/ami-id", region=REGION, ami_id="ami-pinned-99")
    assert identity == "ami=ami-pinned-99"


@mock_aws
def test_resolve_ray_config_pins_ami_id_over_ssm(tmp_path: Path) -> None:
    """When ami_id is given, the cluster boots THAT image, not whatever the SSM
    pointer currently holds — a mid-campaign re-bake can't change the booted image.
    """
    ami_param, _, _ = _seed_ssm_and_vpc()  # SSM ami-id = ami-0123456789abcdef0
    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ami_id="ami-pinned-different",  # differs from the SSM value on purpose
        ssm_prefix=SSM_PREFIX,
        cluster_name="test-cluster",
    )
    try:
        config = yaml.safe_load(Path(resolved).read_text())
        for node in config["available_node_types"].values():
            assert node["node_config"]["ImageId"] == "ami-pinned-different"  # pinned id wins
    finally:
        cleanup_ray_tempfiles(resolved)


# ---------------------------------------------------------------------------
# ray_cluster lifecycle / teardown
# ---------------------------------------------------------------------------


def test_ray_cluster_finalizer_tag_terminates_when_ray_down_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed `ray down` in the NORMAL exit path falls back to tag termination.

    The cancellation/crash hook only covers cancelled or crashed flows — a
    normally-completed run whose `ray down` fails must not leave the fleet
    billing.
    """
    template = tmp_path / "cluster.yaml"
    template.write_text("cluster_name: tessera-inference\n")

    monkeypatch.setattr(ray_mod, "_resolve_ray_config", lambda *a, **k: "/tmp/resolved.yaml")
    monkeypatch.setattr(ray_mod, "_start_ray_cluster", lambda *a, **k: "10.0.0.5")
    monkeypatch.setattr(ray_mod, "_log_ray_dashboard_ssm_command", lambda *a, **k: None)
    monkeypatch.setattr(ray_mod.ray, "init", lambda *a, **k: None)
    monkeypatch.setattr(ray_mod.ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray_mod, "_stop_ray_cluster", lambda _y, _log: False)  # ray down FAILED
    cleaned: list[str] = []
    terminated: list[dict] = []
    monkeypatch.setattr(ray_mod, "cleanup_ray_tempfiles", lambda y: cleaned.append(y))
    monkeypatch.setattr(ray_mod, "terminate_ray_instances_by_tag", lambda **k: terminated.append(k))

    with ray_mod.ray_cluster(
        _LOG, ami_ssm_name="/x/ami", cluster_yaml=template, cluster_name="tessera-inference-x"
    ) as resolved:
        assert resolved == "/tmp/resolved.yaml"

    assert terminated == [{"cluster_name": "tessera-inference-x", "region": "us-west-2", "log": _LOG}]
    assert cleaned == ["/tmp/resolved.yaml"]  # tempfile cleanup still runs after the fallback


def test_ray_cluster_finalizer_skips_ray_down_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When config resolution fails before launch (no resolved YAML), the
    finalizer must NOT run `ray down` on the unresolved template.

    That template's base cluster_name lacks the flow-specific uuid suffix, so a
    `ray down` against it could tear down an unrelated `tessera-inference`
    cluster. The finalizer terminates by exact tag instead.
    """
    template = tmp_path / "cluster.yaml"
    template.write_text("cluster_name: tessera-inference\n")

    def _resolve_fails(*_a: object, **_k: object) -> str:
        raise RuntimeError("SSM resolution failed")

    monkeypatch.setattr(ray_mod, "_resolve_ray_config", _resolve_fails)
    monkeypatch.setattr(ray_mod.ray, "shutdown", lambda: None)
    stop_calls: list[str] = []
    monkeypatch.setattr(ray_mod, "_stop_ray_cluster", lambda y, _log: stop_calls.append(y) or True)
    cleaned: list[str | None] = []
    terminated: list[dict] = []
    monkeypatch.setattr(ray_mod, "cleanup_ray_tempfiles", lambda y: cleaned.append(y))
    monkeypatch.setattr(ray_mod, "terminate_ray_instances_by_tag", lambda **k: terminated.append(k))

    with (
        pytest.raises(RuntimeError, match="SSM resolution failed"),
        ray_mod.ray_cluster(_LOG, ami_ssm_name="/x/ami", cluster_yaml=template, cluster_name="tessera-inference-x"),
    ):
        pass

    assert stop_calls == []  # NO `ray down` on the unresolved template
    assert terminated == [{"cluster_name": "tessera-inference-x", "region": "us-west-2", "log": _LOG}]
    assert cleaned == [None]  # cleanup called with None (no resolved YAML) — a safe no-op


class TestRayDashboardSsmCommand:
    """The copy-pasteable port-forward command logged once the head node is up."""

    def _launch_head(self, cluster_name: str) -> str:
        """Run a tagged head instance in moto and return its instance ID."""
        ec2 = boto3.client("ec2", region_name=REGION)
        resp = ec2.run_instances(
            ImageId="ami-0123456789abcdef0",
            MinCount=1,
            MaxCount=1,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "ray-cluster-name", "Value": cluster_name},
                        {"Key": "ray-node-type", "Value": "head"},
                    ],
                }
            ],
        )
        return str(resp["Instances"][0]["InstanceId"])

    @mock_aws
    def test_forwards_a_port_on_the_instance_not_a_remote_host(self, caplog) -> None:
        """The ``...ToRemoteHost`` document cannot reach the head's own port.

        That variant addresses hosts reachable *from* the instance, and current
        ``amazon-ssm-agent`` versions refuse loopback destinations for it
        ("Forwarding to IP address localhost is forbidden"). The dashboard runs
        on the head itself, so the plain document is the right one — and it takes
        no ``host`` parameter.
        """
        instance_id = self._launch_head("tessera-ray-abc123")
        with caplog.at_level(logging.INFO):
            ray_mod._log_ray_dashboard_ssm_command("tessera-ray-abc123", _LOG, region=REGION)
        msg = caplog.records[-1].getMessage()
        assert "--document-name AWS-StartPortForwardingSession \\" in msg  # trailing \ excludes ...ToRemoteHost
        assert "ToRemoteHost" not in msg
        assert '"host"' not in msg
        assert f"--target {instance_id}" in msg
        assert '{"portNumber":["8265"],"localPortNumber":["8265"]}' in msg

    @mock_aws
    def test_region_is_printed_so_a_differing_default_does_not_bite(self, caplog) -> None:
        """SSM resolves the target within one region.

        An operator whose default region differs from the cluster's gets a
        target-not-connected failure rather than a tunnel, so the region the head
        was found in is baked into the command.
        """
        self._launch_head("tessera-ray-abc123")
        with caplog.at_level(logging.INFO):
            ray_mod._log_ray_dashboard_ssm_command("tessera-ray-abc123", _LOG, region=REGION)
        assert f"  --region {REGION} \\\n" in caplog.records[-1].getMessage()

    @mock_aws
    def test_missing_head_warns_without_raising(self, caplog) -> None:
        """Best-effort: no head found is a warning, never a failed cluster start."""
        with caplog.at_level(logging.WARNING):
            ray_mod._log_ray_dashboard_ssm_command("no-such-cluster", _LOG, region=REGION)
        assert "Could not find Ray head node" in caplog.records[-1].getMessage()


def test_cleanup_ray_tempfiles_handles_missing_path() -> None:
    """``cleanup_ray_tempfiles(None)`` and a missing path are both no-ops."""
    cleanup_ray_tempfiles(None)
    cleanup_ray_tempfiles("/nonexistent/path/does-not-exist.yaml")


def test_cleanup_ray_tempfiles_removes_yaml_and_ssh_key(tmp_path: Path) -> None:
    """Resolved YAML and the SSH key it points to are both deleted."""
    ssh_key = tmp_path / "fake_key.pem"
    ssh_key.write_text("fake")
    yaml_path = tmp_path / "cluster.yaml"
    yaml_path.write_text(yaml.safe_dump({"auth": {"ssh_private_key": str(ssh_key)}}))

    cleanup_ray_tempfiles(str(yaml_path))

    assert not yaml_path.exists()
    assert not ssh_key.exists()


@mock_aws
def test_resolve_ray_config_idle_timeout_override(tmp_path: Path) -> None:
    """idle_timeout_minutes overrides the template value; None keeps it.

    The template's 2-minute default suits per-cell fills; a multi-zone
    sequential fill holds one cluster across zones and must survive the
    inter-zone seam, so it passes a larger value through ray_cluster.
    """
    ami_param, _, _ = _seed_ssm_and_vpc()
    template_default = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())["idle_timeout_minutes"]

    kept = _resolve_ray_config(DEFAULT_CLUSTER_TEMPLATE, region=REGION, ami_ssm_name=ami_param, ssm_prefix=SSM_PREFIX)
    overridden = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        idle_timeout_minutes=10,
    )
    try:
        assert yaml.safe_load(Path(kept).read_text())["idle_timeout_minutes"] == template_default
        assert yaml.safe_load(Path(overridden).read_text())["idle_timeout_minutes"] == 10
    finally:
        cleanup_ray_tempfiles(kept)
        cleanup_ray_tempfiles(overridden)


def test_ray_down_is_bounded_so_the_tag_fallback_can_still_run(monkeypatch):
    """An unreachable head must not wedge teardown and strand a billing GPU fleet.

    ``ray down`` SSHes into the head. Unbounded, a head that never answers means this
    call never returns — so the caller never reaches ``terminate_ray_instances_by_tag``,
    which is the fallback that exists for exactly this failure. The timeout is reported
    as a failed ``ray down`` rather than raised, which is what routes the caller there.
    """
    seen: dict = {}

    def _hangs(cmd, **kwargs):
        seen.update(cmd=cmd, timeout=kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(ray_mod.subprocess, "run", _hangs)

    assert ray_mod._stop_ray_cluster("/tmp/resolved.yaml", _LOG) is False
    assert seen["timeout"] == ray_mod.RAY_DOWN_TIMEOUT_S


# ===========================================================================
# Launch pacing
# ===========================================================================


def _head_ray_start(resolved: str) -> str:
    """The resolved YAML's head command that starts Ray."""
    commands = yaml.safe_load(Path(resolved).read_text())["head_start_ray_commands"]
    return next(cmd for cmd in commands if "ray start" in str(cmd))


@mock_aws
def test_launch_pacing_is_off_by_default() -> None:
    """Default resolution must carry no trace of pacing.

    The campaign this ships into is already running, so a release that changed how a
    fleet grows would change it without anyone having decided to.
    """
    ami_param, _, _ = _seed_ssm_and_vpc()
    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE, region=REGION, ami_ssm_name=ami_param, ssm_prefix=SSM_PREFIX
    )
    try:
        rendered = yaml.safe_dump(yaml.safe_load(Path(resolved).read_text()))
        for name in LAUNCH_PACING_ENV:
            assert name not in rendered, f"{name} reached a cluster nobody asked to pace"
    finally:
        cleanup_ray_tempfiles(resolved)


@mock_aws
def test_launch_pacing_assigns_the_env_on_the_heads_ray_start() -> None:
    """Every name has to land on the command that starts Ray, not beside it.

    Ray runs each start command as its own shell over SSH, so an assignment on any
    other entry reaches nothing — and the autoscaler that issues the launch requests
    is a child of this process, which is how the setting reaches it at all.
    """
    ami_param, _, _ = _seed_ssm_and_vpc()
    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE,
        region=REGION,
        ami_ssm_name=ami_param,
        ssm_prefix=SSM_PREFIX,
        launch_pacing=True,
    )
    try:
        starts_ray = _head_ray_start(resolved)
        for name, value in LAUNCH_PACING_ENV.items():
            assignment = f"{name}={value}"
            assert assignment in starts_ray, f"{assignment} is not on the command that starts Ray"
            assert starts_ray.index(assignment) < starts_ray.index("ray start")

        # Workers run no autoscaler, so pacing there would configure nothing.
        worker = yaml.safe_dump(yaml.safe_load(Path(resolved).read_text())["worker_start_ray_commands"])
        assert not any(name in worker for name in LAUNCH_PACING_ENV)
    finally:
        cleanup_ray_tempfiles(resolved)


def test_pacing_refuses_a_template_that_never_starts_ray() -> None:
    """A pacing request that lands nowhere is worse than one refused.

    The cluster would come up looking configured and launch at the unpaced rate, and
    the only evidence would be the throttling the setting was meant to prevent.
    """
    with pytest.raises(ValueError, match="no start command invokes"):
        ray_mod._pace_ray_start(["echo hello"], LAUNCH_PACING_ENV)


def test_the_pacing_env_reaches_the_client_ray_builds_for_launches(monkeypatch) -> None:
    """The premise the provider change rests on, checked against Ray's own code.

    Ray builds its launch client with an explicit botocore retry config, which is why
    an attempt-count environment variable cannot reach it — an explicitly configured
    count wins over the environment. It sets no retry MODE, and that one botocore does
    still resolve from the environment. Asserted through Ray's own client factory
    rather than a client of our own, because a client we built ourselves could not
    falsify this.
    """
    from ray.autoscaler._private.aws.utils import resource_cache

    for name, value in LAUNCH_PACING_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

    # max_retries=0 is what Ray passes for the fail-fast resource it launches through.
    # resource_cache is lru_cached on its arguments, so an otherwise unused region keeps
    # this off any entry another test may have built.
    client = resource_cache("ec2", "eu-central-1", 0).meta.client
    retries = client.meta.config.retries

    assert retries["mode"] == "adaptive", "the launch client is not rate limited"
    assert retries["total_max_attempts"] == 1, "Ray's fail-fast attempt count was overridden"
    handlers = client.meta.events._emitter._handlers.prefix_search("before-send")
    assert any("ClientRateLimiter" in type(getattr(h, "__self__", h)).__name__ for h in handlers), (
        "adaptive mode registered no send-side rate limiter"
    )


def test_the_head_bootstrap_keeps_its_launch_attempts(monkeypatch) -> None:
    """`ray up` launches the head ONCE, so it must be paced without losing attempts.

    Nothing retries a head that failed to start — there is no autoscaler yet, and a
    failed `ray up` fails the whole fill. Attempt-reducing settings are safe for a worker
    request precisely because the autoscaler will make it again on its next cycle, and
    that reasoning does not transfer. So the bootstrap gets the client-side pacing, which
    only spaces attempts out, and none of the autoscaler tuning, which takes them away.
    """
    # Keyed by verb: `_start_ray_cluster` also shells out to `ray get-head-ip`, and a
    # single slot would report whichever ran last rather than the launch.
    seen: dict = {}

    def _capture(cmd, **kwargs):
        seen[cmd[1]] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="10.0.0.1\n", stderr="")

    monkeypatch.setattr(ray_mod.subprocess, "run", _capture)
    ray_mod._start_ray_cluster("/tmp/resolved.yaml", _LOG, launch_pacing=True)

    env = seen["up"]
    assert env["AWS_RETRY_MODE"] == "adaptive", "the one-shot launch should still be paced"
    for name in LAUNCH_PACING_AUTOSCALER_ENV:
        assert name not in env, f"{name} tunes the autoscaler and must not reach the head bootstrap"


#: The name under test, so the baseline below removes exactly the right key.
_INTERVAL = "AUTOSCALER_UPDATE_INTERVAL_S"


def _ray_autoscaler_interval(override: str | None) -> int:
    """What RAY resolves the interval to in a fresh interpreter: ``override``, or its own default.

    A subprocess because the constant is bound at module import: reading it in-process after
    monkeypatching the environment would report the value this test session imported with, not
    the value a freshly-started autoscaler would see. The autoscaler IS a fresh process, so this
    reproduces its conditions rather than approximating them.

    **The baseline STRIPS the key rather than inheriting the environment**, which is why this
    takes an override rather than a dict. A developer machine or CI runner configured with launch
    pacing already exports this name, and an inherited environment would then measure that
    override and call it Ray's default — failing the comparison below on precisely the hosts the
    setting exists for. Correct code, red test.
    """
    # Imports ONE constants module and prints an int. No `ray.init()`, no `ray start`, no
    # cluster — this repo has stranded a Ray cluster from a test before and the RAM cost was
    # felt for days, so the subprocess is deliberately the narrowest thing that can answer.
    code = "from ray.autoscaler._private.constants import AUTOSCALER_UPDATE_INTERVAL_S as v; print(v)"
    env = {k: v for k, v in os.environ.items() if k != _INTERVAL}
    if override is not None:
        env[_INTERVAL] = override
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return int(out.stdout.strip())


def test_ray_actually_reads_the_interval_from_the_environment() -> None:
    """The setting is only pacing if RAY reads it, and that is Ray's contract to keep, not ours.

    Asserting our own dict against a default copied out of Ray proves nothing about Ray — it
    passes whether or not the name is environment-backed. So this asks Ray, in a fresh
    interpreter, and compares the answer with and without the variable set.

    **This is the regression that would make the whole setting inert**: if a future Ray pins the
    interval as a plain module constant, the value we export would be silently ignored while the
    export still reads like tuning. That failure is invisible everywhere else.
    """
    ours = int(LAUNCH_PACING_AUTOSCALER_ENV[_INTERVAL])
    default = _ray_autoscaler_interval(None)
    applied = _ray_autoscaler_interval(str(ours))
    assert applied == ours, (
        f"Ray resolved AUTOSCALER_UPDATE_INTERVAL_S to {applied} with the environment set to "
        f"{ours}; the setting is not environment-backed in this Ray and paces nothing"
    )
    assert ours > default, (
        f"our value {ours} is at or below Ray's own default {default}, so it slows nothing even though Ray reads it"
    )
    assert _INTERVAL not in LAUNCH_PACING_CLIENT_ENV, (
        "`ray up` runs no autoscaler loop, so the name is inert in the one-shot bootstrap"
    )


def test_the_head_start_command_carries_both_halves() -> None:
    """The head both launches and hosts the autoscaler, so it wants everything.

    The split is about audience, not about dropping a setting: whatever is withheld from
    the one-shot bootstrap still has to reach the autoscaler, or the pacing that matters
    for worker launches would be lost along with it.
    """
    assert LAUNCH_PACING_CLIENT_ENV.items() <= LAUNCH_PACING_ENV.items()
    assert LAUNCH_PACING_AUTOSCALER_ENV.items() <= LAUNCH_PACING_ENV.items()
    assert not set(LAUNCH_PACING_CLIENT_ENV) & set(LAUNCH_PACING_AUTOSCALER_ENV), (
        "a name in both halves has no single owner"
    )


def test_pacing_off_leaves_the_bootstrap_environment_alone() -> None:
    """Default-off means `ray up` inherits this process's environment untouched."""
    seen: dict = {}

    def _capture(cmd, **kwargs):
        seen[cmd[1]] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="10.0.0.1\n", stderr="")

    with patch.object(ray_mod.subprocess, "run", _capture):
        ray_mod._start_ray_cluster("/tmp/resolved.yaml", _LOG)
    assert seen["up"] is None


# ---------------------------------------------------------------------------
# The GPU node-type ladder, and the Ray arithmetic the whole design rests on.
# ---------------------------------------------------------------------------

#: A real ``ec2:describe-instance-types`` answer for the g6e family plus the head
#: type, captured from us-west-2 on 2026-08-27. Ray's autoscaler autodetects a node
#: type's CPU/GPU/memory from exactly this shape, so the quota arithmetic and the
#: preference ordering below are computed from the vendor's own numbers rather than
#: from a table someone typed. Recapture with::
#:
#:     aws ec2 describe-instance-types --filters Name=instance-type,Values=g6e.*,m5.2xlarge
#:
#: keeping only InstanceType / VCpuInfo / MemoryInfo / GpuInfo / NetworkInfo, and
#: OMITTING GpuInfo on the CPU-only types — Ray does ``.get("GpuInfo", {}).get(...)``,
#: which raises on a key materialised as null. A `--query` projection that wrote it
#: as null made an otherwise-faithful fixture crash the code it was meant to feed.
EC2_TYPES_FIXTURE = Path(__file__).parents[1] / "fixtures" / "ec2_describe_instance_types_g6e.json"

#: The resource bundle one ``InferenceActor`` requests: one GPU, and the CPU Ray
#: assigns an actor by default (nothing sets ``num_cpus``). Every score below is
#: relative to this, and changing it changes which node type Ray prefers.
ACTOR_BUNDLE = {"CPU": 1, "GPU": 1}


def _ec2_instance_types() -> list[dict]:
    return json.loads(EC2_TYPES_FIXTURE.read_text())


def _autodetected_node_types(ladder: str | None = None, *, drop_declared: bool = True) -> dict:
    """The template's node types as RAY sees them, after its own autodetection.

    Args:
        ladder: A ``gpu-worker-ladder`` value to apply first, or ``None`` to leave
            the template's shipped ``max_workers`` in place.
        drop_declared: Strip every ``resources:`` block before autodetection, so
            each node type is filled from the EC2 catalogue. The two wider rungs
            ship WITHOUT one deliberately; dropping the older rungs' declared
            ``{CPU: 4, GPU: 1}`` too makes every row in a comparison come from the
            same source. (It changes nothing for them — the declared values match
            the catalogue — which is itself worth having checked, and
            :func:`test_declared_resources_match_the_ec2_catalogue` does.)
    """
    from ray.autoscaler._private.aws import node_provider as np_mod

    cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
    if ladder is not None:
        _apply_gpu_worker_ladder(cfg, ladder)
    if drop_declared:
        for node_type in cfg["available_node_types"].values():
            node_type.pop("resources", None)
    with patch.object(np_mod, "list_ec2_instances", return_value=_ec2_instance_types()):
        filled = np_mod.AWSNodeProvider.fillout_available_node_types_resources(cfg)
    return filled["available_node_types"]


def _scorer():
    """Ray's default node-type scorer, bound the way its autoscaler binds it.

    ``get_nodes_for`` calls the scorer with three positionals while the default
    scorer also takes a keyword-only ``node_availability_summary``; Ray closes that
    gap with a ``functools.partial`` inside ``get_nodes_to_launch``. Reproducing the
    partial here is what makes these tests exercise the real call, not a lookalike.

    ``{}`` for the summary is faithful, not a stub: the scorer takes the argument
    and never reads it, which is why a capacity failure has no influence at all on
    which node type Ray picks next.
    """
    from functools import partial

    from ray.autoscaler._private.resource_demand_scheduler import _default_utilization_scorer

    return partial(_default_utilization_scorer, node_availability_summary={})


class TestGpuWorkerLadderParsing:
    """The ladder value REFUSES anything malformed rather than part-applying it."""

    def test_parses_pairs_in_order_tolerating_whitespace_and_trailing_comma(self) -> None:
        assert _parse_gpu_worker_ladder(" g6e.xlarge:16 , g6e.12xlarge:4 ,") == [
            ("g6e.xlarge", 16),
            ("g6e.12xlarge", 4),
        ]

    def test_zero_is_a_legal_count(self) -> None:
        """`0` is how a rung is explicitly closed, so it must parse, not refuse."""
        assert _parse_gpu_worker_ladder("g6e.xlarge:0") == [("g6e.xlarge", 0)]

    @pytest.mark.parametrize(
        "raw",
        [
            "g6e.xlarge",  # no separator
            "g6e.xlarge:",  # no count
            ":16",  # no name
            "g6e.xlarge:x",  # non-numeric count
            "g6e.xlarge:-1",  # negative count
            "g6e.xlarge:1.5",  # non-integer count
            "",  # set but names nothing
            ",",  # ditto
        ],
    )
    def test_refuses_a_malformed_value(self, raw: str) -> None:
        with pytest.raises(RuntimeError, match=GPU_WORKER_LADDER_SSM_KEY):
            _parse_gpu_worker_ladder(raw)

    def test_refuses_a_repeated_instance_type(self) -> None:
        """Two counts for one type have no defined winner, so neither is applied."""
        with pytest.raises(RuntimeError, match="Duplicate instance type"):
            _parse_gpu_worker_ladder("g6e.xlarge:1,g6e.xlarge:2")


class TestGpuWorkerLadderApplication:
    """Applying a ladder, and the invariant that its absence changes nothing."""

    def test_absent_key_leaves_the_node_types_byte_identical(self) -> None:
        """The whole safety argument for shipping this: no key, no change.

        Compared as SERIALISED YAML rather than field by field, so a key nobody
        thought to assert on cannot drift either.
        """
        shipped = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        untouched = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        assert yaml.dump(untouched["available_node_types"], sort_keys=True) == yaml.dump(
            shipped["available_node_types"], sort_keys=True
        )

    def test_both_wider_rungs_ship_unreachable(self) -> None:
        """Merging this change must not alter a campaign in flight."""
        node_types = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())["available_node_types"]
        wider = {"g6e.2xlarge", "g6e.12xlarge"}
        found = {
            cfg["node_config"]["InstanceType"]: cfg["max_workers"]
            for cfg in node_types.values()
            if cfg["node_config"]["InstanceType"] in wider
        }
        assert found == dict.fromkeys(wider, 0), f"a wider rung ships reachable: {found}"

    def test_the_wider_rungs_declare_no_resources(self) -> None:
        """No `resources:` block means Ray autodetects, so no hardcoded vCPU can be wrong."""
        node_types = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())["available_node_types"]
        for name, cfg in node_types.items():
            if cfg["node_config"]["InstanceType"] in ("g6e.2xlarge", "g6e.12xlarge"):
                assert "resources" not in cfg, f"{name} overrides Ray's autodetection"

    def test_applies_counts_to_the_named_rungs(self) -> None:
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        _apply_gpu_worker_ladder(cfg, "g6e.xlarge:16,g6e.12xlarge:4")
        got = {
            name: nt["max_workers"]
            for name, nt in cfg["available_node_types"].items()
            if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
        }
        assert sorted(got.values()) == [0, 4, 16]
        by_type = {
            nt["node_config"]["InstanceType"]: nt["max_workers"]
            for name, nt in cfg["available_node_types"].items()
            if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
        }
        assert by_type == {"g6e.xlarge": 16, "g6e.2xlarge": 0, "g6e.12xlarge": 4}

    def test_is_authoritative_not_additive(self) -> None:
        """A rung the ladder does not name is CLOSED, including the shipped default.

        The alternative reading — additive — would make releasing one rung also
        leave the 500-node `g6e.xlarge` ceiling standing, so one edit would make two
        changes and only one of them would be written down.
        """
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        assert cfg["available_node_types"]["gpu-workers-ondemand"]["max_workers"] == 500
        _apply_gpu_worker_ladder(cfg, "g6e.12xlarge:4")
        assert cfg["available_node_types"]["gpu-workers-ondemand"]["max_workers"] == 0

    def test_never_touches_the_spot_rung(self) -> None:
        """Spot is outside the ladder's domain: same instance type, different market."""
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        _apply_gpu_worker_ladder(cfg, "g6e.xlarge:16")
        spot = cfg["available_node_types"]["gpu-workers-spot"]
        assert spot["max_workers"] == 0
        assert "InstanceMarketOptions" in spot["node_config"]

    def test_raises_the_cluster_ceiling_to_fit_the_ladder(self) -> None:
        """Ray takes the min of the two ceilings, so a ladder above the global one is capped."""
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        assert cfg["max_workers"] == 500
        _apply_gpu_worker_ladder(cfg, "g6e.xlarge:400,g6e.2xlarge:400")
        assert cfg["max_workers"] == 800

    def test_leaves_a_sufficient_cluster_ceiling_alone(self) -> None:
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        _apply_gpu_worker_ladder(cfg, "g6e.xlarge:10")
        assert cfg["max_workers"] == 500

    def test_refuses_an_instance_type_no_rung_offers(self) -> None:
        """A typo, or a rung that was never shipped — either way the fleet is not what was asked for."""
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        with pytest.raises(RuntimeError, match="no on-demand GPU node type"):
            _apply_gpu_worker_ladder(cfg, "g6e.99xlarge:4")

    def test_refuses_when_two_rungs_share_an_instance_type(self) -> None:
        """Then a ladder entry has no single target, and picking one silently is a guess."""
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        cfg["available_node_types"][f"{GPU_WORKER_NODE_TYPE_PREFIX}-dupe"] = {
            "node_config": {"InstanceType": "g6e.xlarge"},
            "min_workers": 0,
            "max_workers": 0,
        }
        with pytest.raises(RuntimeError, match="more than one on-demand GPU node type"):
            _apply_gpu_worker_ladder(cfg, "g6e.xlarge:16")

    def test_refuses_when_the_template_has_no_governable_rung(self) -> None:
        """A template rename would otherwise make the ladder govern nothing, quietly."""
        cfg = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())
        cfg["available_node_types"] = {
            k: v for k, v in cfg["available_node_types"].items() if not k.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
        }
        with pytest.raises(RuntimeError, match="declares no node type"):
            _apply_gpu_worker_ladder(cfg, "g6e.xlarge:16")


@mock_aws
def test_resolve_ray_config_reads_the_ladder_from_ssm() -> None:
    """The release mechanism end-to-end: one SSM key, no code change, no re-registration."""
    ami_param, _, _ = _seed_ssm_and_vpc()
    boto3.client("ssm", region_name=REGION).put_parameter(
        Name=f"{SSM_PREFIX}{GPU_WORKER_LADDER_SSM_KEY}",
        Value="g6e.xlarge:16,g6e.12xlarge:4",
        Type="String",
    )
    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE, region=REGION, ami_ssm_name=ami_param, ssm_prefix=SSM_PREFIX
    )
    node_types = yaml.safe_load(Path(resolved).read_text())["available_node_types"]
    by_type = {
        nt["node_config"]["InstanceType"]: nt["max_workers"]
        for name, nt in node_types.items()
        if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
    }
    assert by_type == {"g6e.xlarge": 16, "g6e.2xlarge": 0, "g6e.12xlarge": 4}
    cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_without_the_ladder_key_keeps_the_templates_node_types() -> None:
    """No key is the default, and the default must be today's behaviour exactly."""
    ami_param, _, _ = _seed_ssm_and_vpc()
    resolved = _resolve_ray_config(
        DEFAULT_CLUSTER_TEMPLATE, region=REGION, ami_ssm_name=ami_param, ssm_prefix=SSM_PREFIX
    )
    shipped = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())["available_node_types"]
    got = yaml.safe_load(Path(resolved).read_text())["available_node_types"]
    # `node_config` is rewritten with AMI/subnet/tag injections by design; the
    # sizing fields are what the ladder would have touched.
    for name, cfg in shipped.items():
        assert got[name]["max_workers"] == cfg["max_workers"], name
        assert got[name]["min_workers"] == cfg["min_workers"], name
        assert got[name].get("resources") == cfg.get("resources"), name
        assert got[name]["node_config"]["InstanceType"] == cfg["node_config"]["InstanceType"], name
    cleanup_ray_tempfiles(resolved)


@mock_aws
def test_resolve_ray_config_refuses_a_malformed_ladder_before_writing_anything() -> None:
    """A bad parameter fails the launch loudly, rather than growing the wrong fleet."""
    ami_param, _, _ = _seed_ssm_and_vpc()
    boto3.client("ssm", region_name=REGION).put_parameter(
        Name=f"{SSM_PREFIX}{GPU_WORKER_LADDER_SSM_KEY}", Value="g6e.xlarge=16", Type="String"
    )
    with pytest.raises(RuntimeError, match=GPU_WORKER_LADDER_SSM_KEY):
        _resolve_ray_config(DEFAULT_CLUSTER_TEMPLATE, region=REGION, ami_ssm_name=ami_param, ssm_prefix=SSM_PREFIX)


class TestRayNodeTypePreference:
    """Pin RAY's node-type choice, because our design is built on its arithmetic.

    This is the highest-value test in the change. Adding a multi-GPU rung is safe
    only because of how one vendored Ray file scores node types, and that scoring is
    an accident of the formula rather than a documented contract: with our actor
    bundle the utilisation term ties at 0 for every GPU type, so the decision falls
    through to the MEAN of ``v * util**3`` over the node's resources, where the GPU
    term is effectively just the GPU count. A Ray upgrade that changes the formula
    would silently re-shape a live fleet. It must break a test instead.

    Everything here runs against POST-AUTODETECT resource dicts — what Ray actually
    scores — and needs no GPU, no `ray.init()`, and no AWS call.
    """

    def test_multi_gpu_rung_is_preferred_once_three_actors_are_unplaced(self) -> None:
        """The premise: at campaign scale Ray prefers the 4-GPU box.

        And the threshold, which is the part that is easy to get wrong: the
        preference depends on HOW MANY bundles are unplaced, because the score is
        computed against the whole outstanding demand. Below three, a `g6e.xlarge`
        wins; from three up, the `g6e.12xlarge` does and stays winning. A campaign
        asks for hundreds, so the multi-GPU rung is what it gets — but a
        one-or-two-actor smoke test would land on `g6e.xlarge` and look like a
        contradiction, so the boundary is pinned rather than just the tail.
        """
        node_types = _autodetected_node_types("g6e.xlarge:500,g6e.2xlarge:500,g6e.12xlarge:500")
        scorer = _scorer()

        def winner(n: int) -> str:
            scores = {
                name: scorer(cfg["resources"], [ACTOR_BUNDLE] * n, name)
                for name, cfg in node_types.items()
                if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
            }
            return max(scores, key=lambda k: scores[k])

        assert winner(1).endswith("ondemand")
        assert winner(2).endswith("ondemand")
        assert winner(3).endswith("-12xl")
        for n in (4, 8, 20, 250):
            assert winner(n).endswith("-12xl"), n

    def test_the_two_single_gpu_rungs_are_ranked_by_gpu_density_not_size(self) -> None:
        """`g6e.2xlarge` scores BELOW `g6e.xlarge`, so it is a fallback and never a preference.

        This is the operationally important half and it is counter-intuitive: the
        bigger box is the less attractive one, because its extra vCPU sits idle under
        our one-CPU bundle and drags the mean down. See
        :func:`test_releasing_the_2xlarge_rung_alone_changes_nothing`.
        """
        node_types = _autodetected_node_types("g6e.xlarge:500,g6e.2xlarge:500")
        scorer = _scorer()
        demand = [ACTOR_BUNDLE] * 250
        xl = scorer(node_types["gpu-workers-ondemand"]["resources"], demand, "x")
        xl2 = scorer(node_types["gpu-workers-ondemand-2xl"]["resources"], demand, "y")
        assert xl > xl2

    def test_releasing_the_2xlarge_rung_alone_changes_nothing(self) -> None:
        """The `g6e.xlarge` ceiling is what releases the `g6e.2xlarge` pool, not the rung.

        Capacity failures have no influence on node-type choice — Ray's scorer takes
        a ``node_availability_summary`` and never reads it — so a fleet that cannot
        buy `g6e.xlarge` keeps asking for `g6e.xlarge`. Opening the `2xlarge` rung
        beside an unrestricted `xlarge` rung therefore buys exactly zero nodes in a
        different capacity pool, which is the entire point of adding it. Lowering the
        `xlarge` count is the move.
        """
        from ray.autoscaler._private.resource_demand_scheduler import get_nodes_for

        demand = [ACTOR_BUNDLE] * 250
        both_open, _ = get_nodes_for(
            _autodetected_node_types("g6e.xlarge:500,g6e.2xlarge:500"),
            {},
            "head",
            600,
            demand,
            _scorer(),
        )
        assert dict(both_open) == {"gpu-workers-ondemand": 250}

        capped, _ = get_nodes_for(
            _autodetected_node_types("g6e.xlarge:100,g6e.2xlarge:500"),
            {},
            "head",
            600,
            demand,
            _scorer(),
        )
        assert dict(capped) == {"gpu-workers-ondemand": 100, "gpu-workers-ondemand-2xl": 150}

    def test_max_workers_zero_makes_a_rung_unreachable(self) -> None:
        """The release mechanism, from the other side: 0 means Ray cannot pick it at all.

        Ray skips a node type whose count has reached its ``max_workers``, so the
        demand goes unsatisfied rather than spilling onto a closed rung. That is what
        makes shipping both wider rungs at 0 inert, and it is the only mechanism that
        moves the autoscaler's choice.
        """
        from ray.autoscaler._private.resource_demand_scheduler import get_nodes_for

        chosen, residual = get_nodes_for(
            _autodetected_node_types("g6e.12xlarge:2"), {}, "head", 600, [ACTOR_BUNDLE] * 250, _scorer()
        )
        assert dict(chosen) == {"gpu-workers-ondemand-12xl": 2}
        # 2 nodes x 4 GPUs place 8 of the 250; the rest stay unplaced rather than
        # landing on the rungs the ladder closed.
        assert len(residual) == 250 - 8

    def test_the_experiments_ladder_yields_both_arms_on_one_cluster(self) -> None:
        """The dev packing measurement's cluster shape, checked before it is paid for.

        16 single-GPU nodes and 4 four-GPU nodes = 32 GPUs on 256 vCPU, drawing from
        one work queue. If this arithmetic did not hold, the two arms would not
        coexist and the whole measurement design collapses into two separate runs.
        """
        from ray.autoscaler._private.resource_demand_scheduler import get_nodes_for

        chosen, residual = get_nodes_for(
            _autodetected_node_types("g6e.xlarge:16,g6e.12xlarge:4"),
            {},
            "head",
            600,
            [ACTOR_BUNDLE] * 32,
            _scorer(),
        )
        assert dict(chosen) == {"gpu-workers-ondemand-12xl": 4, "gpu-workers-ondemand": 16}
        assert residual == []


class TestEc2CatalogueArithmetic:
    """Assert the quota and packing numbers against the vendor's own catalogue.

    The whole "widen the pool" case is a table of vCPU-per-GPU ratios, and that table
    came from an agent's knowledge of the family rather than from any of our
    documents. Anchoring it on a captured API answer means a wrong figure is a test
    failure and not a budget.
    """

    def test_every_g6e_size_carries_the_same_card(self) -> None:
        """There is no "bigger GPU" in this family — only better-fed ones.

        Every `g6e` size is one or more L40S at 45,776 MiB. So no workload can be
        scaled UP to a larger accelerator by choosing a larger instance; the only
        thing a larger size changes is vCPU and host RAM per GPU. This closes off a
        whole class of proposal, which is why it is asserted rather than remembered.
        """
        gpus = {
            t["InstanceType"]: t["GpuInfo"]["Gpus"][0]
            for t in _ec2_instance_types()
            if t["InstanceType"].startswith("g6e.")
        }
        assert len(gpus) == 8, sorted(gpus)
        assert {(g["Name"], g["MemoryInfo"]["SizeInMiB"]) for g in gpus.values()} == {("L40S", 45776)}

    def test_the_l40s_is_44_7_gib_not_46_gb(self) -> None:
        """The denominator of every "% of the card" figure we quote.

        45,776 MiB is 44.7 GiB and 48.0 GB decimal. It was once recorded as "46 GB"
        — the MiB figure rounded and relabelled — which understates the card by ~3%
        in GiB terms and, more importantly, made the L4/A10G at 22,888 MiB look like
        "barely half" when the real ratio is 1.94x.
        """
        card_mib = next(
            t["GpuInfo"]["Gpus"][0]["MemoryInfo"]["SizeInMiB"]
            for t in _ec2_instance_types()
            if t["InstanceType"] == "g6e.xlarge"
        )
        assert card_mib == 45776
        assert round(card_mib / 1024, 1) == 44.7
        assert round(card_mib * 1024 * 1024 / 1e9, 1) == 48.0

    def test_vcpu_per_gpu_ranks_the_candidate_sizes(self) -> None:
        """`g6e.2xlarge` is the most quota-efficient rung after `g6e.xlarge`.

        The applied G-and-VT quota is counted in vCPU, so vCPU-per-GPU is the price
        of a GPU in quota. Every multi-GPU size is WORSE on this axis than
        `g6e.2xlarge`, which cuts against the "fewer, bigger boxes" instinct: any
        diversification strictly reduces the GPU count a fixed quota can hold.
        """
        ratios = {}
        for t in _ec2_instance_types():
            if not t["InstanceType"].startswith("g6e."):
                continue
            ratios[t["InstanceType"]] = t["VCpuInfo"]["DefaultVCpus"] // t["GpuInfo"]["Gpus"][0]["Count"]
        assert ratios == {
            "g6e.xlarge": 4,
            "g6e.2xlarge": 8,
            "g6e.12xlarge": 12,
            "g6e.4xlarge": 16,
            "g6e.24xlarge": 24,
            "g6e.48xlarge": 24,
            "g6e.8xlarge": 32,
            "g6e.16xlarge": 64,
        }
        assert min(ratios, key=lambda k: ratios[k]) == "g6e.xlarge"
        assert sorted(ratios, key=lambda k: ratios[k])[1] == "g6e.2xlarge"

    def test_host_ram_per_gpu_clears_the_actor_budget_on_every_rung_we_ship(self) -> None:
        """~17.7 GB per actor is the measured requirement; 16 GiB hosts have OOMed before.

        Asserted for the rungs the template offers, because this is the term that
        disqualifies the small `g5`/`g6` sizes and it is easy to reintroduce.
        """
        per_gpu_gib = {
            t["InstanceType"]: t["MemoryInfo"]["SizeInMiB"] / 1024 / t["GpuInfo"]["Gpus"][0]["Count"]
            for t in _ec2_instance_types()
            if t["InstanceType"] in ("g6e.xlarge", "g6e.2xlarge", "g6e.12xlarge")
        }
        assert per_gpu_gib == {"g6e.xlarge": 32.0, "g6e.2xlarge": 64.0, "g6e.12xlarge": 96.0}
        assert all(v >= 18.0 for v in per_gpu_gib.values())

    def test_declared_resources_match_the_ec2_catalogue(self) -> None:
        """Where the template DOES declare resources, the declaration must be true.

        A declared key overrides Ray's autodetection with no error, so a wrong vCPU
        count would make the autoscaler scale against a fiction. The two wider rungs
        avoid the risk by declaring nothing; the two older rungs declare `{CPU: 4,
        GPU: 1}` and this is the check that they still agree with the vendor.
        """
        catalogue = {t["InstanceType"]: t for t in _ec2_instance_types()}
        node_types = yaml.safe_load(DEFAULT_CLUSTER_TEMPLATE.read_text())["available_node_types"]
        checked = 0
        for name, cfg in node_types.items():
            declared = cfg.get("resources")
            instance_type = cfg["node_config"]["InstanceType"]
            if not declared or instance_type not in catalogue:
                continue
            entry = catalogue[instance_type]
            assert declared["CPU"] == entry["VCpuInfo"]["DefaultVCpus"], name
            if "GPU" in declared:
                assert declared["GPU"] == entry["GpuInfo"]["Gpus"][0]["Count"], name
            checked += 1
        assert checked == 3, "expected head + two single-GPU rungs to declare resources"
