#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ $(uname -n) == i1 ]]
trap 'rc=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$rc" "$$" > job_result.json.tmp; mv job_result.json.tmp job_result.json' EXIT
exec 9>../../.scratch/a100.lock
while true; do
  if flock -n 9; then
    IFS=',' read -r free util < <(nvidia-smi --id=0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
    active=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | awk 'NF {n++} END {print n+0}')
    if (( free >= 72 * 1024 && util <= 5 && active == 0 )); then
      break
    fi
    flock -u 9
  fi
  printf '%s waiting for shared A100 lock/idle GPU\n' "$(date -Is)"
  sleep 60
done
unset PYTHONHOME LD_LIBRARY_PATH TF_FORCE_UNIFIED_MEMORY XLA_FLAGS XLA_PYTHON_CLIENT_ALLOCATOR
mkdir -p .scratch
export TMPDIR="$PWD/.scratch" XDG_CACHE_HOME="$PWD/.scratch" CUDA_CACHE_PATH="$PWD/.scratch/cuda" TRITON_CACHE_DIR="$PWD/.scratch/triton"
export PYTHONPATH="$PWD/af3/src:$PWD/site-packages"
export LIBCIFPP_DATA_DIR="$PWD/site-packages/share/libcifpp"
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.75
export AF3_CAPTURE_COMPRESS=1 AF3_CAPTURE_COMPRESS_LEVEL=1 AF3_CAPTURE_MIN_FREE_GB=6 AF3_CAPTURE_WAIT_FREE_SECONDS=900
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
nvidia-smi --id=0 --query-gpu=name,memory.total,memory.free --format=csv
timeout --kill-after=20s 50m nice -n 10 python3.12 -u af3/scripts/capture_trunk.py \
  --run-existing --native-gpu --attention-implementation=triton \
  --bundle-provenance="$PWD/source_manifest.json" --output-dir="$PWD/out" > out/inference.log 2>&1
