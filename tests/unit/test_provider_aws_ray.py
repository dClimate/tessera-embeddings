"""Unit tests for the AWS Ray provider's pure helpers.

These tests stay offline by mocking SSM and EC2 with ``moto``. They
exercise ``_resolve_ray_config`` (the most complex pure helper),
``_pick_least_loaded_subnet``, and ``cleanup_ray_tempfiles``.

The full ``ray_cluster`` context manager is NOT tested here because it
shells out to ``ray up`` / ``ray down``; that path is an integration
concern.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
import yaml
from moto import mock_aws

from tessera_embeddings.providers.aws.ray import (
    DEFAULT_CLUSTER_TEMPLATE,
    PROJECT_TAG_VALUE,
    _pick_least_loaded_subnet,
    _resolve_ray_config,
    cleanup_ray_tempfiles,
)

REGION = "us-west-2"
SSM_PREFIX = "/test/tessera/ray/"


def _seed_ssm_and_vpc() -> tuple[str, list[str], str]:
    """Populate SSM and EC2 fixtures and return (ami_param, subnet_ids, key_pair_id)."""
    ssm = boto3.client("ssm", region_name=REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

    # VPC + 2 subnets (so _pick_least_loaded_subnet has something to choose between)
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]
    subnet_a = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.1.0/24", AvailabilityZone=f"{REGION}a")["Subnet"]
    subnet_b = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.2.0/24", AvailabilityZone=f"{REGION}b")["Subnet"]
    subnet_ids = [subnet_a["SubnetId"], subnet_b["SubnetId"]]

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
        assert config["provider"]["availability_zone"].startswith(REGION)

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
            assert nc["SubnetIds"][0] in subnet_ids
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


@mock_aws
def test_pick_least_loaded_subnet_returns_least_loaded() -> None:
    """Subnet with the fewest tagged Ray instances wins."""
    ec2 = boto3.client("ec2", region_name=REGION)
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]
    subnet_a = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.1.0/24", AvailabilityZone=f"{REGION}a")["Subnet"]
    subnet_b = ec2.create_subnet(VpcId=vpc["VpcId"], CidrBlock="10.0.2.0/24", AvailabilityZone=f"{REGION}b")["Subnet"]
    # Run 2 instances in subnet_a tagged for our cluster — subnet_b should win
    ec2.run_instances(
        ImageId="ami-x",
        InstanceType="t3.micro",
        MinCount=2,
        MaxCount=2,
        SubnetId=subnet_a["SubnetId"],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "ray-cluster-name", "Value": "test-cluster"}],
            }
        ],
    )

    subnets = [subnet_a, subnet_b]
    chosen = _pick_least_loaded_subnet(subnets, "test-cluster", ec2)
    assert chosen["SubnetId"] == subnet_b["SubnetId"]


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
