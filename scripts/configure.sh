#!/usr/bin/env bash
# One-time setup: put your GitHub username (lowercased) into the files that need it.
# Usage (from the project root):   bash scripts/configure.sh YourGitHubUsername
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: bash scripts/configure.sh <YourGitHubUsername>"
  exit 1
fi

# Docker image names must be lowercase, so always lowercase the username.
GH_USER=$(echo "$1" | tr '[:upper:]' '[:lower:]')
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

FILES=(
  "$ROOT/.github/workflows/ci.yml"
  "$ROOT/scripts/aws_deploy.sh"
  "$ROOT/scripts/setup_github_deploy.sh"
  "$ROOT/README.md"
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    perl -pi -e "s/your-github-username/$GH_USER/g" "$f"
    echo "Updated ${f#"$ROOT"/}"
  fi
done

echo
echo "Done. Image name: ghcr.io/$GH_USER/marine-fish-classifier"
grep -n "ghcr.io/" "$ROOT/.github/workflows/ci.yml" | head -1
