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

import logging
from pathlib import Path

import boto3
import pytest
import yaml
from moto import mock_aws

from tessera_embeddings.providers.aws import ray as ray_mod
from tessera_embeddings.providers.aws.ray import (
    DEFAULT_CLUSTER_TEMPLATE,
    PROJECT_TAG_VALUE,
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
