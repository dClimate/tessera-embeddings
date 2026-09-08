#!/usr/bin/env bash
# Provision the production-scale reproduction box in the dev account.
#
# What it does, in order:
#   1. Packages TWO source trees from this repository — the pre-fix `main` and the fix stack — as
#      tarballs, plus the branch-agnostic driver, and uploads them to the dev embeddings bucket.
#   2. Launches ONE c7i.48xlarge (192 vCPU / 384 GB / 50 Gbps) on Amazon Linux 2023 in the default
#      VPC, on the EXISTING `global-tessera-dev-ray-worker` instance profile (it already grants
#      read/write/delete on the dev store, SSM, and CloudWatch) — no IAM is created.
#   3. User data installs uv, py-spy and s5cmd, builds one virtualenv per source tree, and writes a
#      READY marker to S3. The instance SELF-TERMINATES after $TTL_HOURS whatever happens: shutdown
#      behaviour is `terminate` and a `shutdown -h` is scheduled at boot, so a forgotten box cannot
#      bill past the afternoon.
#
# Run from the repository root with the dev profile:
#   AWS_PROFILE=global-tessera-dev scripts/scoping/wedge_repro/provision.sh
# Prints the instance id. Wait for READY with:
#   aws s3 ls s3://global-tessera-embeddings-dev/scoping/prod-repro/box/READY
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-west-2}"
BUCKET="${BUCKET:-global-tessera-embeddings-dev}"
PREFIX="scoping/prod-repro"
FIX_REF="${FIX_REF:-dev/publication-density-harness}"
MAIN_REF="${MAIN_REF:-origin/main}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.48xlarge}"
TTL_HOURS="${TTL_HOURS:-8}"
PROFILE_NAME="global-tessera-dev-ray-worker"
SUBNET="${SUBNET:-subnet-0fb32f9a147121485}"   # IsolatedVPC public subnet, us-west-2a (IGW route, public IP)
# The Ray fleet's own security group: all egress permitted, and the same fleet whose instance
# profile this box borrows. NOT the VPC's "default" group — that one has NO egress rules, and the
# first attempt (2026-09-08) sat unreachable for 15 minutes because of it. Resolved by NAME below
# and checked for egress before launch, so a wrong id fails here rather than on the box.
SG_NAME="${SG_NAME:-global-tessera-dev-ray-cluster}"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
[ "$ACCOUNT" = "658132200637" ] || { echo "refusing: this is not the dev account ($ACCOUNT)"; exit 1; }

# --- pre-flight: the box must be able to reach the internet, S3 and SSM, or nothing else matters ---
VPC=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --query 'Subnets[0].VpcId' --output text)
SG="${SG:-$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC" "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)}"
[ -n "$SG" ] && [ "$SG" != "None" ] || { echo "refusing: security group '$SG_NAME' not found in $VPC"; exit 1; }
EGRESS=$(aws ec2 describe-security-groups --group-ids "$SG" --query 'length(SecurityGroups[0].IpPermissionsEgress)' --output text)
[ "$EGRESS" -ge 1 ] || { echo "refusing: security group $SG has no egress rules; the box could reach nothing"; exit 1; }
IGW=$(aws ec2 describe-route-tables --filters "Name=association.subnet-id,Values=$SUBNET" --query 'RouteTables[0].Routes[?GatewayId!=null && starts_with(GatewayId, `igw-`)].GatewayId | [0]' --output text)
[ -n "$IGW" ] && [ "$IGW" != "None" ] || { echo "refusing: subnet $SUBNET has no internet-gateway route"; exit 1; }
aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null || { echo "refusing: instance profile $PROFILE_NAME missing"; exit 1; }
echo "pre-flight ok: vpc=$VPC subnet=$SUBNET (igw $IGW) sg=$SG ($SG_NAME, $EGRESS egress rule(s)) profile=$PROFILE_NAME"
git rev-parse --is-inside-work-tree >/dev/null || { echo "run from the tessera-embeddings repo"; exit 1; }
git fetch -q origin

WORK=$(mktemp -d)
echo "packaging main=$(git rev-parse --short "$MAIN_REF") fix=$(git rev-parse --short "$FIX_REF")"
git archive --format=tar.gz -o "$WORK/src-main.tar.gz" "$MAIN_REF"
git archive --format=tar.gz -o "$WORK/src-fix.tar.gz" "$FIX_REF"
cp scripts/scoping/wedge_repro/prod_scale_repro.py "$WORK/prod_scale_repro.py"
aws s3 cp --only-show-errors "$WORK/src-main.tar.gz" "s3://$BUCKET/$PREFIX/box/src-main.tar.gz"
aws s3 cp --only-show-errors "$WORK/src-fix.tar.gz" "s3://$BUCKET/$PREFIX/box/src-fix.tar.gz"
aws s3 cp --only-show-errors "$WORK/prod_scale_repro.py" "s3://$BUCKET/$PREFIX/box/prod_scale_repro.py"
aws s3 rm --only-show-errors "s3://$BUCKET/$PREFIX/box/READY" 2>/dev/null || true

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query 'Parameter.Value' --output text)

USERDATA=$(cat <<EOF
#!/usr/bin/env bash
set -euxo pipefail
exec > >(tee /var/log/repro-bootstrap.log | logger -t repro) 2>&1
# The safety net first: nothing below can outlive it.
shutdown -h +$((TTL_HOURS * 60))
dnf install -y -q tar gzip which htop
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:\$PATH"
uv python install 3.12
uv tool install py-spy
curl -LsS https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
mkdir -p /opt/repro && cd /opt/repro
for arm in main fix; do
  mkdir -p \$arm && aws s3 cp --only-show-errors s3://$BUCKET/$PREFIX/box/src-\$arm.tar.gz - | tar -xz -C \$arm
  (cd \$arm && uv venv --python 3.12 .venv >/dev/null && uv sync --extra aws --no-dev --frozen 2>&1 | tail -3)
done
aws s3 cp --only-show-errors s3://$BUCKET/$PREFIX/box/prod_scale_repro.py /opt/repro/prod_scale_repro.py
ln -sf /root/.local/bin/py-spy /usr/local/bin/py-spy
mkdir -p /opt/repro/results
echo "ready \$(date -u +%FT%TZ) main=\$(cd main && git -C . rev-parse --short HEAD 2>/dev/null || echo tarball) fix=tarball" > /opt/repro/READY
aws s3 cp --only-show-errors /opt/repro/READY s3://$BUCKET/$PREFIX/box/READY
EOF
)

IID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1 \
  --subnet-id "$SUBNET" --security-group-ids "$SG" --associate-public-ip-address \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wedge-prod-repro},{Key=Purpose,Value=assembly-wedge-reproduction},{Key=Owner,Value=robert.banick},{Key=TTLHours,Value=$TTL_HOURS}]" \
  --user-data "$USERDATA" \
  --query 'Instances[0].InstanceId' --output text)
echo "launched $IID ($INSTANCE_TYPE, self-terminates in ${TTL_HOURS}h)"
echo "wait for readiness: aws s3 ls s3://$BUCKET/$PREFIX/box/READY   (bootstrap ~10 min: two uv syncs)"
echo "bootstrap log on the box: /var/log/repro-bootstrap.log"
echo "$IID" > "$WORK/instance-id"; echo "instance id also in $WORK/instance-id"
