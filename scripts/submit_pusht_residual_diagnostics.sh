#!/usr/bin/env bash
# Submit residual PushT diagnostics A/B to a GPU worker.
# Usage:
#   RUN_DIR=/abs/path/to/formal_run NODE=4090node3 bash scripts/submit_pusht_residual_diagnostics.sh A
#   RUN_DIR=/abs/path/to/formal_run NODE=4090node3 bash scripts/submit_pusht_residual_diagnostics.sh B
#   RUN_DIR=/abs/path/to/formal_run NODE=4090node3 bash scripts/submit_pusht_residual_diagnostics.sh both
set -euo pipefail

TARGET="${1:-both}"
if [[ "$TARGET" != "A" && "$TARGET" != "B" && "$TARGET" != "both" ]]; then
  echo "Usage: RUN_DIR=... [NODE=4090node3] bash $0 {A|B|both}" >&2
  exit 2
fi

REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
NODE="${NODE:-4090node3}"
CONDA_SH="${CONDA_SH:-/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lewm}"
STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
RUN_DIR="${RUN_DIR:-$REPO/outputs/pusht_official_diagnostic/formal_20260903T024414Z_3108082}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/recordings/lewm.pt" || ! -f "$RUN_DIR/recordings/ald_tf.pt" ]]; then
  echo "Missing recordings under: $RUN_DIR/recordings" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT_ROOT="$REPO/outputs/pusht_residual_diagnostics/${TARGET}_${STAMP}"
GEN_DIR="$REPO/slurm/generated_residual_diagnostics"
LOG_DIR="$REPO/logs"
mkdir -p "$OUT_ROOT" "$GEN_DIR" "$LOG_DIR"
JOB="$GEN_DIR/${TARGET}_${STAMP}.slurm"

cat > "$JOB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=jepa_resid_${TARGET}
#SBATCH --partition=GPU
#SBATCH --nodelist=${NODE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_DIR}/residual_${TARGET}_${STAMP}_%j.out
#SBATCH --error=${LOG_DIR}/residual_${TARGET}_${STAMP}_%j.err
set -euo pipefail

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO}"

export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

echo "node=\$(hostname)"
echo "python=\$(which python)"
python --version
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"

if [[ "${TARGET}" == "A" || "${TARGET}" == "both" ]]; then
python -u eval_pusht_diagnostic_a_case77.py \
  --config-name=pusht.yaml \
  solver=cem \
  seed=42 \
  "+diagnostic.run_dir=${RUN_DIR}" \
  "+diagnostic.output_dir=${OUT_ROOT}/diagnostic_A_case77" \
  +diagnostic.case=77 \
  +diagnostic.solves='[0,1]' \
  +diagnostic.sources='[lewm,ald_tf]' \
  +diagnostic.iterations='[0,3,9,19,29]' \
  +diagnostic.model_batch_size=64 \
  +diagnostic.pair_margin_frac=0.02
fi

if [[ "${TARGET}" == "B" || "${TARGET}" == "both" ]]; then
python -u eval_pusht_diagnostic_b_case93.py \
  --config-name=pusht.yaml \
  solver=cem \
  seed=42 \
  "+diagnostic.run_dir=${RUN_DIR}" \
  "+diagnostic.output_dir=${OUT_ROOT}/diagnostic_B_case93" \
  +diagnostic.case=93 \
  +diagnostic.solve=1 \
  +diagnostic.source=ald_tf \
  +diagnostic.mode=formal \
  "+diagnostic.designs=300x5,900x5,300x7,300x10" \
  +diagnostic.restarts=4 \
  +diagnostic.n_steps=30 \
  +diagnostic.replay_iterations='[0,3,9,19,29]' \
  +diagnostic.topk=30 \
  +diagnostic.base_seed=42
fi

echo "=== DONE ==="
echo "Results: ${OUT_ROOT}"
EOF

bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
echo "Submitted: $JID"
echo "Node: $NODE"
echo "Results: $OUT_ROOT"
echo "Log: $LOG_DIR/residual_${TARGET}_${STAMP}_${JID%%;*}.out"
