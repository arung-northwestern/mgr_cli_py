#!/usr/bin/env bash
# Save mgrcli:local Docker image as a zstd-compressed tarball.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mgrcli:local}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${REPO_ROOT}/mgrcli_local.tar.zst}"

echo "Saving ${IMAGE_NAME} to ${OUTPUT} (zstd)..."
docker save "$IMAGE_NAME" | zstd -o "$OUTPUT"
echo "Done. Size: $(du -h "$OUTPUT" | cut -f1)"
