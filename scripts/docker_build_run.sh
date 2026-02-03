#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mgrcli:local}"
PORT="${PORT:-8501}"
DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR:-$HOME/MGR_Results}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "$OUTPUT_DIR" ]]; then
  read -r -p "Output folder [${DEFAULT_OUTPUT_DIR}]: " OUTPUT_DIR
  OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
fi

mkdir -p "$OUTPUT_DIR"

if [[ -z "${RAYON_NUM_THREADS:-}" ]]; then
  if [[ -n "${MGR_RAYON_THREADS:-}" ]]; then
    RAYON_NUM_THREADS="$MGR_RAYON_THREADS"
  else
    MEM_TOTAL_BYTES="$(docker info --format '{{.MemTotal}}' 2>/dev/null || true)"
    GIB=$((1024 * 1024 * 1024))
    if [[ -n "$MEM_TOTAL_BYTES" ]]; then
      if (( MEM_TOTAL_BYTES <= 6 * GIB )); then
        RAYON_NUM_THREADS=2
      elif (( MEM_TOTAL_BYTES <= 10 * GIB )); then
        RAYON_NUM_THREADS=4
      elif (( MEM_TOTAL_BYTES <= 14 * GIB )); then
        RAYON_NUM_THREADS=6
      elif (( MEM_TOTAL_BYTES <= 20 * GIB )); then
        RAYON_NUM_THREADS=8
      else
        RAYON_NUM_THREADS=10
      fi
    else
      RAYON_NUM_THREADS=4
    fi
  fi
fi

echo "Building image ${IMAGE_NAME}..."
docker build -t "$IMAGE_NAME" .

echo "Starting container on port ${PORT}..."
docker run --rm \
  -p "${PORT}:8501" \
  -v "${OUTPUT_DIR}:/results" \
  -e MGR_RESULTS_DIR=/results \
  -e RAYON_NUM_THREADS="${RAYON_NUM_THREADS}" \
  "$IMAGE_NAME"
