#!/usr/bin/env bash
# Cross-platform Docker build using buildx. Use this instead of `docker build`
# so the image runs on Windows (linux/amd64), Intel/AMD Linux, and ARM Mac/Linux.
#
# The image is built *inside* the container for each platform, so the Rust
# backend (maturin) is compiled for the target platform automatically.
#
# Usage:
#   ./scripts/docker_buildx.sh
#
# Build for current platform only (faster, load into Docker):
#   PLATFORMS=linux/arm64 ./scripts/docker_buildx.sh   # ARM Mac
#   PLATFORMS=linux/amd64 ./scripts/docker_buildx.sh   # Windows / Intel Linux
#
# Build for multiple platforms and export to tar files (no registry needed):
#   EXPORT_TAR=1 ./scripts/docker_buildx.sh
#   Produces mgrcli-windows_linux_64.tar (Windows + Intel/AMD Linux) and
#   mgrcli-mac_arm64.tar (ARM Mac); share via Google Drive etc. Load on target:
#   docker load -i mgrcli-windows_linux_64.tar   # or mgrcli-mac_arm64.tar
#
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mgrcli:local}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
EXPORT_TAR="${EXPORT_TAR:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

# Friendly tar names for sharing (e.g. Google Drive) so colabs know which to use
arch_to_tar_name() {
  case "$1" in
    amd64) echo "windows_linux_64" ;;
    arm64) echo "mac_arm64" ;;
    *)     echo "$1" ;;
  esac
}

# Split platforms for multi-platform handling
IFS=',' read -ra PLATFORM_ARRAY <<< "$PLATFORMS"
NPLATFORMS=${#PLATFORM_ARRAY[@]}

if ! docker buildx inspect mgrcli_builder >/dev/null 2>&1; then
  docker buildx create --name mgrcli_builder --use
else
  docker buildx use mgrcli_builder
fi

build_one() {
  local platform="$1"
  local arch
  arch=$(echo "$platform" | sed 's/linux\///')
  docker buildx build \
    --platform "$platform" \
    -t "${IMAGE_NAME}" \
    "$@"
}

if [[ "$EXPORT_TAR" == "1" ]] && [[ "$NPLATFORMS" -gt 0 ]]; then
  # Build each platform and export to a tar file (for transfer without a registry)
  for platform in "${PLATFORM_ARRAY[@]}"; do
    arch=$(echo "$platform" | sed 's/linux\///')
    friendly=$(arch_to_tar_name "$arch")
    out="${OUTPUT_DIR}/mgrcli-${friendly}.tar"
    echo "Building ${platform} -> ${out}"
    docker buildx build \
      --platform "$platform" \
      -t "${IMAGE_NAME}" \
      --output "type=docker,dest=${out}" \
      .
    echo "Saved: ${out}"
  done
  echo "Done. Load on target machine: docker load -i $(basename "${out}")"
elif [[ "$NPLATFORMS" -eq 1 ]]; then
  # Single platform: build and load into local Docker
  build_one "${PLATFORM_ARRAY[0]}" --load .
else
  # Multiple platforms but no export: can only load one. Build first platform and load.
  echo "Multiple platforms (${PLATFORMS}) but EXPORT_TAR not set: building and loading first platform only (${PLATFORM_ARRAY[0]}). Use EXPORT_TAR=1 to export all to tar files."
  docker buildx build \
    --platform "${PLATFORM_ARRAY[0]}" \
    -t "${IMAGE_NAME}" \
    --load \
    .
fi
