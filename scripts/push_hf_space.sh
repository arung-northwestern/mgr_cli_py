#!/usr/bin/env bash
# Push the current repo tree to your Hugging Face Docker Space.
#
# SAFETY (default): targets PRIVATE TESTING only. Production push is blocked
# unless you pass --allow-production (and should only be used with explicit user consent).
#
# Prerequisites:
#   - hf auth login   (or git credentials for huggingface.co)
#   - Run from repo root, or any cwd (script cd's to repo root).
#
# A push to main triggers an automatic Space rebuild — that is the "restart".
#
# Environment (optional):
#   HF_SPACE_URL   default: private testing Space (see PROD_URL / PRIVATE_URL below)
#   HF_CLONE_DIR   local cache for the Space git clone
#   HF_BUNDLE_DIR  temp bundle output (default: /tmp/mofography-hf-bundle)
#
# Flags:
#   --pull              git pull --ff-only on this repo first
#   -m "message"        custom commit message on the Space repo
#   --confirm-push      REQUIRED to git commit + push (without this: bundle only, no push)
#   --allow-production  allow HF_SPACE_URL pointing at production (still needs --confirm-push)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROD_URL="https://huggingface.co/spaces/arung-northwestern/mofography-web-demo"
PRIVATE_URL="https://huggingface.co/spaces/arung-northwestern/mofography-web-demo-private-testing"

HF_SPACE_URL="${HF_SPACE_URL:-${PRIVATE_URL}}"
BUNDLE="${HF_BUNDLE_DIR:-/tmp/mofography-hf-bundle}"

GIT_PULL=0
COMMIT_MSG=""
CONFIRM_PUSH=0
ALLOW_PRODUCTION=0

usage() {
  echo "Usage: $0 [--pull] [-m \"commit message\"] [--confirm-push] [--allow-production]" >&2
  echo "" >&2
  echo "Default target: ${PRIVATE_URL}" >&2
  echo "Production (${PROD_URL}) is BLOCKED unless --allow-production." >&2
  echo "Git push runs ONLY with --confirm-push (explicit user consent)." >&2
  exit 1
}

is_production_url() {
  local url="$1"
  case "${url}" in
    *mofography-web-demo-private-testing*) return 1 ;;
    *mofography-web-demo*) return 0 ;;
    *) return 1 ;;
  esac
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
    --confirm-push)
      CONFIRM_PUSH=1
      shift
      ;;
    --allow-production)
      ALLOW_PRODUCTION=1
      shift
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

if is_production_url "${HF_SPACE_URL}"; then
  if [[ "${ALLOW_PRODUCTION}" != 1 ]]; then
    echo "ERROR: Refusing to use production Space URL:" >&2
    echo "  ${HF_SPACE_URL}" >&2
    echo "" >&2
    echo "Use private testing (default), or pass --allow-production only after explicit user approval." >&2
    exit 1
  fi
  echo "WARNING: Production Space target — ${HF_SPACE_URL}" >&2
fi

# Resolve clone dir after final target selection.
if [[ -z "${HF_CLONE_DIR:-}" ]]; then
  if is_production_url "${HF_SPACE_URL}"; then
    HF_CLONE_DIR="/tmp/mofography-web-demo-hf-clone"
  else
    HF_CLONE_DIR="/tmp/mofography-web-demo-private-testing-hf-clone"
  fi
fi

echo "HF deploy target: ${HF_SPACE_URL}"
echo "HF clone dir:     ${HF_CLONE_DIR}"
if [[ "${CONFIRM_PUSH}" != 1 ]]; then
  echo "NOTE: --confirm-push not set — will bundle/sync but NOT git push."
fi

if [[ "$GIT_PULL" == 1 ]]; then
  if git rev-parse --verify origin/master >/dev/null 2>&1; then
    git pull --ff-only origin master
  elif git rev-parse --verify origin/main >/dev/null 2>&1; then
    git pull --ff-only origin main
  else
    echo "No origin/master or origin/main; skipping --pull" >&2
  fi
fi

# Block broken Python from reaching the Space (catches IndentationError, truncated files, etc.)
INTERACTIVE_APP="${REPO_ROOT}/python_cli/mgrcli/interactive_app.py"
if [[ ! -f "${INTERACTIVE_APP}" ]]; then
  echo "ERROR: missing ${INTERACTIVE_APP}" >&2
  exit 1
fi
if grep -qE '^[0-9]+[a-z]{2}\|' "${INTERACTIVE_APP}"; then
  echo "ERROR: ${INTERACTIVE_APP} contains Read-tool hashline prefixes (e.g. 1uh|). Restore from git before deploy." >&2
  exit 1
fi
python3 -m py_compile "${INTERACTIVE_APP}" || {
  echo "ERROR: ${INTERACTIVE_APP} does not compile. Fix before push_hf_space.sh." >&2
  exit 1
}
echo "OK: ${INTERACTIVE_APP} compiles"

echo "Building Space bundle in: ${BUNDLE}"
"${REPO_ROOT}/scripts/deploy_to_hf_space.sh" "${BUNDLE}"

# Guardrail: ensure Space build uses the symmetry-aware Rust crate, not repo-root fallback.
DOCKERFILE_IN_BUNDLE="${BUNDLE}/Dockerfile"
if [[ ! -f "${DOCKERFILE_IN_BUNDLE}" ]]; then
  echo "ERROR: Missing Dockerfile in bundle: ${DOCKERFILE_IN_BUNDLE}" >&2
  exit 1
fi
if ! grep -Fq "maturin build --release --manifest-path mgr_src/mgrcli_rust/Cargo.toml" "${DOCKERFILE_IN_BUNDLE}"; then
  echo "ERROR: Bundle Dockerfile is not pinned to mgr_src/mgrcli_rust manifest path." >&2
  echo "Refusing push to avoid building the wrong Rust CIF parser crate." >&2
  exit 1
fi
if ! grep -Fq "COPY --from=builder /app/mgr_src/mgrcli_rust/target/wheels/*.whl" "${DOCKERFILE_IN_BUNDLE}"; then
  echo "ERROR: Bundle Dockerfile wheel copy path does not target mgr_src/mgrcli_rust." >&2
  echo "Refusing push to avoid stale/wrong wheel artifacts." >&2
  exit 1
fi

# Never ship local HTS artifacts, test junk, or binary logos (Hub rejects binaries without Xet)
rm -f "${BUNDLE}/ZIF-8-RASPA.cif" "${BUNDLE}"/streamlit_*.png "${BUNDLE}/test_streamlit_app.py" 2>/dev/null || true
rm -f "${BUNDLE}"/queue_*.png "${BUNDLE}"/docs/*.png 2>/dev/null || true
rm -rf "${BUNDLE}/hts_output" "${BUNDLE}/.todos" "${BUNDLE}/.pytest_cache" "${BUNDLE}/.cursor" 2>/dev/null || true

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

if [[ "${CONFIRM_PUSH}" != 1 ]]; then
  echo ""
  echo "Prepared clone at: ${HF_CLONE_DIR}"
  echo "No git push (missing --confirm-push). User must explicitly approve, then re-run:"
  echo "  ./scripts/push_hf_space.sh --confirm-push -m \"your message\""
  exit 0
fi

if [[ -z "${COMMIT_MSG}" ]]; then
  COMMIT_MSG="Deploy: sync from mgr_cli_py $(date -u +%Y-%m-%dT%H:%MZ)"
fi

echo ""
echo "About to commit and push to: ${HF_SPACE_URL}"
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
