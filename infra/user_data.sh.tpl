#!/bin/bash
# Runs once, as root, on the server's first boot.
# Terraform fills in the region and ECR values; "$$" in this file is a literal "$" for bash.
set -euxo pipefail

dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

# One deploy script, used by you over SSH and by GitHub Actions (via SSM).
cat > /usr/local/bin/deploy-app.sh <<'SCRIPT'
#!/bin/bash
set -euo pipefail
TAG="$${1:-latest}"
IMAGE="${repository_url}:$TAG"

aws ecr get-login-password --region ${region} \
  | docker login --username AWS --password-stdin ${registry}

docker pull "$IMAGE"
docker rm -f marine-fish-classifier-api 2>/dev/null || true
docker run -d --name marine-fish-classifier-api --restart unless-stopped -p 80:8000 "$IMAGE"

for i in $(seq 1 45); do
  if curl -fs http://localhost/health | grep -q '"model_loaded":true'; then
    echo "Deployed $IMAGE - healthy"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done
echo "Container did not become healthy"
docker logs marine-fish-classifier-api
exit 1
SCRIPT
chmod +x /usr/local/bin/deploy-app.sh

# ECR is empty on first boot, so this is expected to fail the first time.
/usr/local/bin/deploy-app.sh || echo "No image in ECR yet - push one, then run: sudo /usr/local/bin/deploy-app.sh"
