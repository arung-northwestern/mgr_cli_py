#!/usr/bin/env bash
# Push the current repo tree to your Hugging Face Docker Space.
#
# Prerequisites:
#   - hf auth login   (or git credentials for huggingface.co)
#   - Run from repo root, or any cwd (script cd's to repo root).
#
# A push to main triggers an automatic Space rebuild — that is the "restart".
#
# Environment (optional):
#   HF_SPACE_URL   default: https://huggingface.co/spaces/arung-northwestern/mofography-web-demo
#   HF_CLONE_DIR   local cache for the Space git clone (default: /tmp/mofography-web-demo-hf-clone)
#   HF_BUNDLE_DIR  temp bundle output (default: /tmp/mofography-hf-bundle)
#
# Flags:
#   --pull         git pull --ff-only on this repo first (use after pushing to GitHub)
#   -m "message"   custom commit message on the Space repo
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_SPACE_URL="${HF_SPACE_URL:-https://huggingface.co/spaces/arung-northwestern/mofography-web-demo}"
HF_CLONE_DIR="${HF_CLONE_DIR:-/tmp/mofography-web-demo-hf-clone}"
BUNDLE="${HF_BUNDLE_DIR:-/tmp/mofography-hf-bundle}"

GIT_PULL=0
COMMIT_MSG=""

usage() {
  echo "Usage: $0 [--pull] [-m \"commit message\"]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)
      GIT_PULL=1
      shift
      ;;
    -m|--message)
      [[ $# -ge 2 ]] || usage
      COMMIT_MSG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

if [[ "$GIT_PULL" == 1 ]]; then
  if git rev-parse --verify origin/master >/dev/null 2>&1; then
    git pull --ff-only origin master
  elif git rev-parse --verify origin/main >/dev/null 2>&1; then
    git pull --ff-only origin main
  else
    echo "No origin/master or origin/main; skipping --pull" >&2
  fi
fi

echo "Building Space bundle in: ${BUNDLE}"
"${REPO_ROOT}/scripts/deploy_to_hf_space.sh" "${BUNDLE}"

# Never ship local HTS artifacts, test junk, or binary logos (Hub rejects binaries without Xet)
rm -f "${BUNDLE}/ZIF-8-RASPA.cif" "${BUNDLE}"/streamlit_*.png "${BUNDLE}/test_streamlit_app.py" 2>/dev/null || true
rm -f "${BUNDLE}"/queue_*.png "${BUNDLE}"/docs/*.png 2>/dev/null || true
rm -rf "${BUNDLE}/hts_output" 2>/dev/null || true

if [[ ! -d "${HF_CLONE_DIR}/.git" ]]; then
  rm -rf "${HF_CLONE_DIR}"
  echo "Cloning Space repo to: ${HF_CLONE_DIR}"
  git clone "${HF_SPACE_URL}" "${HF_CLONE_DIR}"
else
  git -C "${HF_CLONE_DIR}" fetch origin
  git -C "${HF_CLONE_DIR}" reset --hard origin/main
fi

rsync -a --delete \
  --exclude='.git' \
  --exclude='hts_output' \
  "${BUNDLE}/" "${HF_CLONE_DIR}/"

# Valid Space card metadata (avoids empty-template warnings on the Hub)
cp "${REPO_ROOT}/spaces/README.hf.md" "${HF_CLONE_DIR}/README.md"

if [[ -z "${COMMIT_MSG}" ]]; then
  COMMIT_MSG="Deploy: sync from mgr_cli_py $(date -u +%Y-%m-%dT%H:%MZ)"
fi

git -C "${HF_CLONE_DIR}" add -A
if git -C "${HF_CLONE_DIR}" diff --cached --quiet; then
  echo "No file changes vs last Space commit; nothing to push."
  exit 0
fi

git -C "${HF_CLONE_DIR}" commit -m "${COMMIT_MSG}"
git -C "${HF_CLONE_DIR}" push origin main

echo ""
echo "Done. Pushed to: ${HF_SPACE_URL}"
echo "The Space will rebuild automatically (same as a restart). Check Logs on the Space page."
