#!/usr/bin/env bash
# Phase 5: provision AWS with Terraform and deploy the app to EC2.
# Run from the project root:   bash scripts/aws_deploy.sh
# Safe to re-run: Terraform only changes what's different, and the deploy
# step just replaces the running container.
set -euo pipefail

REGION=ap-south-1
KEY_NAME=mlops-key
KEY_FILE="$HOME/.ssh/$KEY_NAME.pem"
SOURCE_IMAGE=ghcr.io/manavnavadiya/marine-fish-classifier:latest
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFRA="$ROOT/infra"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "1/8 Checking tools and AWS login"
for tool in aws terraform docker curl ssh; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool"; exit 1; }
done
if [[ "$SOURCE_IMAGE" == *"your-github-""username"* ]]; then
  echo "Run first:  bash scripts/configure.sh <YourGitHubUsername>"
  exit 1
fi
aws sts get-caller-identity --query Arn --output text
docker info >/dev/null 2>&1 || { echo "Docker Desktop isn't running. Start it and re-run."; exit 1; }

# ---------------------------------------------------------------------------
step "2/8 Your public IP (only this IP may SSH into the server)"
MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
echo "my_ip_cidr = \"$MY_IP/32\"" > "$INFRA/terraform.tfvars"
echo "$MY_IP/32"

# ---------------------------------------------------------------------------
step "3/8 SSH key pair"
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
  if [ -f "$KEY_FILE" ]; then
    echo "Using existing key pair and $KEY_FILE"
  else
    echo "Key pair '$KEY_NAME' exists in AWS, but $KEY_FILE is missing on this Mac."
    echo "Delete the AWS copy, then re-run this script:"
    echo "  aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION"
    exit 1
  fi
else
  mkdir -p "$HOME/.ssh"
  [ -f "$KEY_FILE" ] && chmod 600 "$KEY_FILE"
  aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
  echo "Created $KEY_FILE"
fi

# ---------------------------------------------------------------------------
step "4/8 Terraform: init, validate, plan"
cd "$INFRA"
terraform init -input=false
terraform fmt
terraform validate
terraform plan -input=false -out=tfplan

echo
read -r -p "Create/update these AWS resources (about Rs 2-3 per hour while running)? Enter = yes, Ctrl+C = stop: "

step "5/8 Terraform: apply"
terraform apply -input=false tfplan
rm -f tfplan

ECR_URL=$(terraform output -raw ecr_repository_url)
IP=$(terraform output -raw public_ip)
REGISTRY=${ECR_URL%%/*}

# ---------------------------------------------------------------------------
step "6/8 Copying the image from GHCR to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
if ! docker buildx imagetools create --tag "$ECR_URL:latest" "$SOURCE_IMAGE"; then
  echo "Registry-to-registry copy failed; falling back to pull, tag, push (amd64 only)"
  docker pull --platform linux/amd64 "$SOURCE_IMAGE"
  docker tag "$SOURCE_IMAGE" "$ECR_URL:latest"
  docker push "$ECR_URL:latest"
fi

# ---------------------------------------------------------------------------
step "7/8 Waiting for the server to finish booting, then deploying"
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ec2-user@$IP")
for i in $(seq 1 30); do
  "${SSH[@]}" true 2>/dev/null && break
  echo "  SSH not ready yet ($i/30)..."
  sleep 10
done
"${SSH[@]}" 'cloud-init status --wait' || true
"${SSH[@]}" 'sudo /usr/local/bin/deploy-app.sh'

# ---------------------------------------------------------------------------
step "8/8 Verifying from your Mac over the public IP"
curl -fsS "http://$IP/health"; echo
curl -fsS -H "Content-Type: application/json" \
  -d '{"length_cm":9,"weight_kg":0.02,"height_cm":4,"fin_count":7,"water_temp_c":26}' \
  "http://$IP/predict"; echo

step "Done"
echo "App in browser:  http://$IP"
echo "SSH:             ssh -i $KEY_FILE ec2-user@$IP"
echo "Tear down later: cd infra && terraform destroy"
