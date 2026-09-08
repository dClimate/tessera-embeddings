#!/usr/bin/env bash
# Tear down the production-scale reproduction: terminate the box, delete the stores, KEEP the results.
#
#   AWS_PROFILE=global-tessera-dev scripts/scoping/wedge_repro/teardown.sh
#
# Deletes every icechunk store under scoping/prod-repro/*.icechunk and the box bootstrap files.
# Deliberately leaves scoping/prod-repro/results/ alone: that is the evidence (reports, logs, the
# py-spy dumps). Delete it by hand once it has been read.
set -euo pipefail
export AWS_REGION="${AWS_REGION:-us-west-2}"
BUCKET="${BUCKET:-global-tessera-embeddings-dev}"
PREFIX="scoping/prod-repro"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
[ "$ACCOUNT" = "658132200637" ] || { echo "refusing: not the dev account ($ACCOUNT)"; exit 1; }

IDS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=wedge-prod-repro" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$IDS" ]; then
  echo "terminating: $IDS"; aws ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
else
  echo "no wedge-prod-repro instance running"
fi

echo "stores under s3://$BUCKET/$PREFIX/ :"
aws s3 ls "s3://$BUCKET/$PREFIX/" | awk '{print $2}' | grep -E '\.icechunk/$' || echo "  (none)"
for store in $(aws s3 ls "s3://$BUCKET/$PREFIX/" | awk '{print $2}' | grep -E '\.icechunk/$' || true); do
  echo "deleting s3://$BUCKET/$PREFIX/$store"
  if command -v s5cmd >/dev/null; then s5cmd rm --all-versions "s3://$BUCKET/$PREFIX/${store}*" >/dev/null; else aws s3 rm --recursive --only-show-errors "s3://$BUCKET/$PREFIX/$store"; fi
done
aws s3 rm --recursive --only-show-errors "s3://$BUCKET/$PREFIX/box/" || true
echo "remaining under the prefix (results are kept on purpose):"
aws s3 ls --recursive --summarize "s3://$BUCKET/$PREFIX/" | tail -3
