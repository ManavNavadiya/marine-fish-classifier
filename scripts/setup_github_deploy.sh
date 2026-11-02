#!/usr/bin/env bash
# Creates an AWS user for GitHub Actions (push to ECR + run the deploy command
# via SSM) and stores its keys and the server details in the GitHub repo.
# Run from the project root:   bash scripts/setup_github_deploy.sh
set -euo pipefail

REPO=manavnavadiya/marine-fish-classifier
DEPLOY_USER=github-deployer
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

if [[ "$REPO" == "your-github-""username"/* ]]; then
  echo "Run first:  bash scripts/configure.sh <YourGitHubUsername>"
  exit 1
fi

step "Reading server details from Terraform"
cd "$ROOT/infra"
INSTANCE_ID=$(terraform output -raw instance_id)
PUBLIC_IP=$(terraform output -raw public_ip)
ECR_URL=$(terraform output -raw ecr_repository_url)
echo "Instance: $INSTANCE_ID   IP: $PUBLIC_IP"
echo "ECR:      $ECR_URL"

step "Creating AWS user '$DEPLOY_USER' for GitHub Actions"
if aws iam get-user --user-name "$DEPLOY_USER" >/dev/null 2>&1; then
  echo "User already exists (probably from your earlier project) - reusing it"
else
  aws iam create-user --user-name "$DEPLOY_USER" >/dev/null
  echo "Created"
fi
# Push images to ECR, and send the deploy command to EC2 through Systems Manager.
aws iam attach-user-policy --user-name "$DEPLOY_USER" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
aws iam attach-user-policy --user-name "$DEPLOY_USER" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMFullAccess

step "Creating an access key"
KEY_COUNT=$(aws iam list-access-keys --user-name "$DEPLOY_USER" --query 'length(AccessKeyMetadata)' --output text)
if [ "$KEY_COUNT" -ge 2 ]; then
  echo "$DEPLOY_USER already has 2 access keys (the AWS maximum). Delete an old one first:"
  aws iam list-access-keys --user-name "$DEPLOY_USER" --query 'AccessKeyMetadata[].[AccessKeyId,CreateDate]' --output table
  echo "  aws iam delete-access-key --user-name $DEPLOY_USER --access-key-id <OLD_KEY_ID>"
  exit 1
fi
read -r ACCESS_KEY SECRET_KEY <<< "$(aws iam create-access-key --user-name "$DEPLOY_USER" \
  --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)"
echo "Access key: $ACCESS_KEY"

step "Saving to GitHub ($REPO)"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh secret set AWS_ACCESS_KEY_ID     -R "$REPO" -b "$ACCESS_KEY"
  gh secret set AWS_SECRET_ACCESS_KEY -R "$REPO" -b "$SECRET_KEY"
  gh variable set ECR_REPOSITORY      -R "$REPO" -b "$ECR_URL"
  gh variable set EC2_INSTANCE_ID     -R "$REPO" -b "$INSTANCE_ID"
  gh variable set EC2_PUBLIC_IP       -R "$REPO" -b "$PUBLIC_IP"
  echo "Secrets and variables saved."
else
  cat <<MSG
GitHub CLI not available, so add these by hand:
GitHub repo -> Settings -> Secrets and variables -> Actions

  Secrets tab:
    AWS_ACCESS_KEY_ID      = $ACCESS_KEY
    AWS_SECRET_ACCESS_KEY  = $SECRET_KEY

  Variables tab:
    ECR_REPOSITORY         = $ECR_URL
    EC2_INSTANCE_ID        = $INSTANCE_ID
    EC2_PUBLIC_IP          = $PUBLIC_IP

Then clear this terminal (Cmd+K) so the secret key isn't left on screen.
MSG
fi

step "Checking the server is registered with Systems Manager"
aws ssm describe-instance-information --region ap-south-1 \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' --output text \
  || echo "(Couldn't check: your terraform-mlops user has no SSM permission. That's fine; the deploy job uses github-deployer.)"
echo "(Should print 'Online'. 'None' means wait a few minutes and check again.)"
