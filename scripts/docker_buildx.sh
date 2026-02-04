#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mgrcli:local}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

if ! docker buildx inspect mgrcli_builder >/dev/null 2>&1; then
  docker buildx create --name mgrcli_builder --use
else
  docker buildx use mgrcli_builder
fi

docker buildx build \
  --platform "${PLATFORMS}" \
  -t "${IMAGE_NAME}" \
  --load \
  .
