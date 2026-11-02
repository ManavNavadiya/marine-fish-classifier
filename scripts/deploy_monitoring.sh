#!/usr/bin/env bash
# Phase 6: open the monitoring ports and start Prometheus + Grafana on EC2.
# Run from the project root:   bash scripts/deploy_monitoring.sh
set -euo pipefail

KEY_FILE="$HOME/.ssh/mlops-key.pem"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

cd "$ROOT/infra"
IP=$(terraform output -raw public_ip)
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ec2-user@$IP")

# ---------------------------------------------------------------------------
step "1/5 Checking the deployed app exposes /metrics"
if ! curl -fs "http://$IP/metrics" | grep -q "predictions_total"; then
  echo "http://$IP/metrics isn't serving metrics yet."
  echo "Make sure the app is deployed (GitHub Actions deploy job green), then re-run."
  exit 1
fi
echo "OK"

# ---------------------------------------------------------------------------
step "2/5 Terraform: make sure ports 3000 (Grafana) and 9090 (Prometheus) are open for your current IP"
echo "my_ip_cidr = \"$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')/32\"" > terraform.tfvars
terraform plan -input=false -out=tfplan
echo
echo "Expect: 0 to add, 0 or 1 to change (only if your IP changed), 0 to destroy."
read -r -p "Apply? Enter = yes, Ctrl+C = stop: "
terraform apply -input=false tfplan
rm -f tfplan

# ---------------------------------------------------------------------------
step "3/5 Copying monitoring config to the server"
"${SSH[@]}" 'rm -rf ~/monitoring'
scp -i "$KEY_FILE" -r "$ROOT/monitoring" "ec2-user@$IP:~/"
"${SSH[@]}" 'chmod -R a+rX ~/monitoring'

# ---------------------------------------------------------------------------
step "4/5 Installing Docker Compose (if needed) and starting the stack"
"${SSH[@]}" 'docker compose version >/dev/null 2>&1 || {
  sudo mkdir -p /usr/local/lib/docker/cli-plugins &&
  sudo curl -fsSL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose &&
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose; }'
"${SSH[@]}" 'cd ~/monitoring && sudo docker compose up -d && sudo docker compose ps'

# ---------------------------------------------------------------------------
step "5/5 Checking Prometheus can scrape the app"
for i in $(seq 1 20); do
  if "${SSH[@]}" "curl -s 'http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22marine-fish-classifier%22%7D' | grep -q '\"1\"\]'"; then
    echo "Prometheus target marine-fish-classifier is UP"
    break
  fi
  echo "  waiting for first scrape ($i/20)..."
  sleep 3
done

step "Done"
echo "Grafana:    http://$IP:3000   (login admin / mlops-demo)"
echo "Prometheus: http://$IP:9090/targets"
echo "Generate traffic:"
echo "  python scripts/load_test.py --url http://$IP --requests 300"
