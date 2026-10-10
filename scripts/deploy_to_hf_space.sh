#!/usr/bin/env bash
# Prepare a directory that can be pushed to a Hugging Face Space (Docker).
# Run from repo root. Creates build_space/ with Dockerfile at root and app code.
# Then push build_space/ to your Space repo (see spaces/README.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build_space}"

echo "Preparing Hugging Face Space bundle in: ${BUILD_DIR}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Copy repo into build_space (exclude things that aren't needed for the image)
rsync -a \
  --exclude='.git' \
  --exclude='target' \
  --exclude='.venv' \
  --exclude='env' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='artifacts' \
  --exclude='test_outputs' \
  --exclude='node_modules' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tar.zst' \
  --exclude='*.zst' \
  --exclude='*.gz' \
  --exclude='build_space' \
  --exclude='hts_output' \
  --exclude='graphify-out' \
  --exclude='.todos' \
  --exclude='.pytest_cache' \
  --exclude='.cursor' \
  --exclude='tmp' \
  --exclude='.doom-cockpit' \
  --exclude='docs/theme_detection_issue.md' \
  --exclude='docs/*.png' \
  --exclude='docs/*.pdf' \
  "${REPO_ROOT}/" "${BUILD_DIR}/"

# Use the Spaces Dockerfile at root (port 7860, MGR_DEPLOYMENT=spaces)
cp "${REPO_ROOT}/spaces/Dockerfile" "${BUILD_DIR}/Dockerfile"

# Hub requires YAML front matter in README.md (sdk: docker, app_port, etc.)
cp "${REPO_ROOT}/spaces/README.hf.md" "${BUILD_DIR}/README.md"

echo "Done. Directory ready to push to a Hugging Face Space."
echo ""
echo "Next steps:"
echo "  1. Create a new Space at https://huggingface.co/new-space (choose Docker)."
echo "  2. Push the contents of build_space/ to the Space repo:"
echo ""
echo "     cd ${BUILD_DIR}"
echo "     git init"
echo "     git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME"
echo "     git add ."
echo "     git commit -m 'Initial Space deployment'"
echo "     git push space main"
echo ""
echo "  3. Wait for the Space to build, then open the Space URL."
echo ""
echo "See spaces/README.md for more options."
