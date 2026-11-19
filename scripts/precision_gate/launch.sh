#!/usr/bin/env bash
# Launch one precision-gate box: one pass over the subnets for each instance type given, no retry loop.
#
# usage: launch.sh <run> <role> <cap_minutes> <instance-type> [<fallback-type> ...]
#   role: gpu-l40s, gpu-a10g, gpu-l4, gpu-cand or cpu-score (see bootstrap.sh); for gpu-cand, GATE_CODE names
#   the candidate's tarball, for every other role today's code
#
# The account's infrastructure comes from the environment, so no account identifiers live in this
# repository:
#   GATE_AMI, GATE_SUBNETS (space-separated), GATE_SG, GATE_INSTANCE_PROFILE_ARN,
#   GATE_OUT (s3 root for runs), GATE_INPUTS (s3 root holding fp8-lattice/ and fp8-mixed/),
#   GATE_CHECKPOINT (s3 URI of the v2 checkpoint), GATE_CODE (code tarball name under $GATE_OUT/<run>/code/),
#   GATE_EXCEPTION (optional: a named exception's file, relative to the repository root),
#   AWS_PROFILE, AWS_REGION
# Prints "<role> <instance-id> <type> <az>" on success; exits 1 if no type had capacity.
set -euo pipefail
RUN=$1 ROLE=$2 CAP=$3
shift 3
: "${GATE_AMI:?}" "${GATE_SUBNETS:?}" "${GATE_SG:?}" "${GATE_INSTANCE_PROFILE_ARN:?}" "${GATE_OUT:?}" \
  "${GATE_INPUTS:?}" "${GATE_CHECKPOINT:?}" "${GATE_CODE:?}" "${AWS_REGION:?}"
here=$(cd "$(dirname "$0")" && pwd)
ud=$(mktemp)
trap 'rm -f "$ud"' EXIT
sed -e "s|__RUN__|$RUN|" -e "s|__ROLE__|$ROLE|" -e "s|__CODE__|$GATE_CODE|" -e "s|__OUT__|$GATE_OUT|" \
  -e "s|__INPUTS__|$GATE_INPUTS|" -e "s|__CHECKPOINT__|$GATE_CHECKPOINT|" \
  -e "s|__EXCEPTION__|${GATE_EXCEPTION:-}|" -e "s|__CAP_MIN__|$CAP|" "$here/bootstrap.sh" > "$ud"
if grep -q '__[A-Z_]*__' "$ud"; then
  echo "unfilled placeholder: $(grep -o '__[A-Z_]*__' "$ud" | head -1)" >&2
  exit 2
fi
for itype in "$@"; do
  for sn in $GATE_SUBNETS; do
    out=$(aws ec2 run-instances --region "$AWS_REGION" --image-id "$GATE_AMI" --instance-type "$itype" \
      --subnet-id "$sn" --security-group-ids "$GATE_SG" --iam-instance-profile "Arn=$GATE_INSTANCE_PROFILE_ARN" \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
      --metadata-options HttpTokens=required --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=gate-$RUN-$ROLE},{Key=purpose,Value=precision-gate}]" \
      --user-data "file://$ud" --query 'Instances[0].[InstanceId,Placement.AvailabilityZone]' --output text 2>&1) || true
    if echo "$out" | grep -q '^i-'; then
      echo "$ROLE $(echo "$out" | awk '{print $1}') $itype $(echo "$out" | awk '{print $2}')"
      exit 0
    fi
    echo "$ROLE $itype $sn: $(echo "$out" | grep -o 'InsufficientInstanceCapacity\|An error[^.]*' | head -1)" >&2
  done
done
exit 1
