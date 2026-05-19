#!/usr/bin/env sh
set -eu

get_mem_limit_bytes() {
  if [ -f /sys/fs/cgroup/memory.max ]; then
    val=$(cat /sys/fs/cgroup/memory.max)
    if [ "$val" != "max" ]; then
      echo "$val"
      return
    fi
  fi
  if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes
    return
  fi
  awk '/MemTotal:/ {print $2 * 1024}' /proc/meminfo
}

calc_threads_from_mem() {
  mem_bytes="$1"
  gib=$((1024 * 1024 * 1024))
  if [ "$mem_bytes" -le $((6 * gib)) ]; then
    echo 2
  elif [ "$mem_bytes" -le $((10 * gib)) ]; then
    echo 4
  elif [ "$mem_bytes" -le $((14 * gib)) ]; then
    echo 6
  elif [ "$mem_bytes" -le $((20 * gib)) ]; then
    echo 8
  else
    echo 10
  fi
}

if [ -z "${RAYON_NUM_THREADS:-}" ]; then
  mem_limit=$(get_mem_limit_bytes)
  threads=$(calc_threads_from_mem "$mem_limit")
  export RAYON_NUM_THREADS="$threads"
fi
echo "MGR CLI Docker: before running in Docker Desktop GUI,"
echo "map host port 8501 -> container 8501, then open http://localhost:8501"
echo ""

exec "$@"
