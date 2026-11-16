#!/usr/bin/env bash
#SBATCH --job-name=af3-demo-r10
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
trap 'rc=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$rc" "${SLURM_JOB_ID}" > job_result.json.tmp; mv job_result.json.tmp job_result.json' EXIT
unset PYTHONHOME LD_LIBRARY_PATH TF_FORCE_UNIFIED_MEMORY XLA_FLAGS XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONPATH="$PWD/af3/src:$PWD/site-packages"
export LIBCIFPP_DATA_DIR="$PWD/site-packages/share/libcifpp"
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.85
export AF3_CAPTURE_COMPRESS=1 AF3_CAPTURE_MIN_FREE_GB=6 AF3_CAPTURE_WAIT_FREE_SECONDS=3600
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python3.12 -u af3/scripts/capture_trunk.py --run-existing --native-gpu \
  --attention-implementation=triton --bundle-provenance="$PWD/source_manifest.json" \
  --output-dir="$PWD/out" > out/inference.log 2>&1
