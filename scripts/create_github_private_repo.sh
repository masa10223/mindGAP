#!/usr/bin/env bash
# Create private GitHub repo masa10223/mindGAP and push current branch.
# Prerequisites:
#   1) git remote not yet set, or origin points where you want
#   2) gh auth login   # interactive — YOU do this
#   3) run this script from the repository root
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer local scripts/gh when system gh is missing
export PATH="$ROOT/scripts:$PATH"

REPO_NAME="${REPO_NAME:-mindGAP}"
OWNER="${OWNER:-masa10223}"
VISIBILITY="${VISIBILITY:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh not found. Install GitHub CLI or place binary at scripts/gh." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated."
  echo "Run:  gh auth login"
  echo "Then re-run:  ./scripts/create_github_private_repo.sh"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not a git repository" >&2
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [ -z "$BRANCH" ]; then
  BRANCH=main
fi

echo "Creating ${VISIBILITY} repo: ${OWNER}/${REPO_NAME} (branch: ${BRANCH})"
echo "Using gh: $(command -v gh)"

# Prefer SSH remote; fall back to HTTPS if SSH is not configured
if gh repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "Repo already exists on GitHub: ${OWNER}/${REPO_NAME}"
else
  gh repo create "${OWNER}/${REPO_NAME}" \
    --"${VISIBILITY}" \
    --source=. \
    --remote=origin \
    --description "mindGAP: GPU Jupyter environment & ELA demo (dummy data)"
fi

# Ensure remote URL
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "git@github.com:${OWNER}/${REPO_NAME}.git" || \
    git remote set-url origin "https://github.com/${OWNER}/${REPO_NAME}.git"
else
  git remote add origin "git@github.com:${OWNER}/${REPO_NAME}.git" || \
    git remote add origin "https://github.com/${OWNER}/${REPO_NAME}.git"
fi

git push -u origin "${BRANCH}"
echo
echo "Done: https://github.com/${OWNER}/${REPO_NAME}"
