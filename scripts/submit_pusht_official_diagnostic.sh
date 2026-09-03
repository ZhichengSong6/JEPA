#!/usr/bin/env bash
# One paired benchmark followed by same-run mechanism/Factor diagnostics.
# DRY_RUN=1 generates and checks the Slurm script without submitting it.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != smoke && "$MODE" != formal ]]; then
  echo "Usage: [FACTOR_POLICY=...] [NODE=4090node3] bash $0 {smoke|formal}" >&2
  exit 2
fi
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
NODE="${NODE:-4090node3}"
CONDA_SH="${CONDA_SH:-/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lewm}"
STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
LEWM_POLICY="${LEWM_POLICY:-lewm_epoch_10}"
ALD_TF_POLICY="${ALD_TF_POLICY:-pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10}"
FACTOR_POLICY="${FACTOR_POLICY:-}"
STAGE="${STAGE:-all}"
if [[ "$STAGE" != all && "$STAGE" != replay ]]; then
  echo "STAGE must be all or replay" >&2
  exit 2
fi
if [[ "$STAGE" == replay && -z "${RUN_DIR:-}" ]]; then
  echo "STAGE=replay requires RUN_DIR from the original run" >&2
  exit 2
fi
if [[ ! "$NODE" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Invalid NODE" >&2
  exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="${RUN_DIR:-$REPO/outputs/pusht_official_diagnostic/${MODE}_${STAMP}}"
# Relative overrides are resolved against the repository, not the submitter's cwd.
if [[ "$RUN_DIR" != /* ]]; then RUN_DIR="$REPO/$RUN_DIR"; fi
GEN_DIR="$REPO/slurm/generated_official_diagnostic"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"
FILE="$GEN_DIR/${MODE}_${STAMP}.slurm"
if [[ "$MODE" == formal ]]; then
  N=300; ITERS=30; K=30; NUM_EVAL=100; REPLAY='[0,3,9,19,29]'; CONTROLS=3
else
  N=30; ITERS=2; K=3; NUM_EVAL=6; REPLAY='[0,1]'; CONTROLS=1
fi

{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=pusht_%s\n' "$MODE"
  printf '#SBATCH --partition=GPU\n#SBATCH --nodelist=%s\n' "$NODE"
  printf '#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=20\n#SBATCH --gres=gpu:1\n'
  printf '#SBATCH --output=%s/official_%s_%s_%%j.out\n' "$LOG_DIR" "$MODE" "$STAMP"
  printf '#SBATCH --error=%s/official_%s_%s_%%j.err\n' "$LOG_DIR" "$MODE" "$STAMP"
  printf 'set -euo pipefail\n'
  # Shell-quote data once; keep the executable job body in a literal heredoc.
  for name in REPO CONDA_SH CONDA_ENV STABLEWM_HOME LEWM_POLICY ALD_TF_POLICY FACTOR_POLICY MODE STAGE RUN_DIR N ITERS K NUM_EVAL REPLAY CONTROLS; do
    printf '%s=%q\n' "$name" "${!name}"
  done
  cat <<'JOB'
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
export STABLEWM_HOME MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

python -u eval_pusht_official_diagnostic.py \
  --config-name=pusht.yaml solver=cem seed=42 \
  solver.num_samples="$N" solver.n_steps="$ITERS" solver.topk="$K" solver.batch_size=1 \
  eval.num_eval="$NUM_EVAL" eval.eval_budget=50 \
  +diagnostic.mode="$MODE" +diagnostic.stage="$STAGE" \
  "+diagnostic.output_dir='$RUN_DIR'" \
  "+diagnostic.lewm_policy='$LEWM_POLICY'" \
  "+diagnostic.ald_policy='$ALD_TF_POLICY'" \
  "+diagnostic.factor_policy='$FACTOR_POLICY'" \
  +diagnostic.replay_iterations="$REPLAY" \
  +diagnostic.max_success_controls="$CONTROLS" \
  +diagnostic.model_batch_size=64

bash scripts/package_pusht_official_diagnostic.sh "$RUN_DIR"
echo "=== SLURM DONE ==="
JOB
} > "$FILE"
bash -n "$FILE"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Generated: $FILE"
  echo "Run directory: $RUN_DIR"
  exit 0
fi
JID="$(sbatch --parsable "$FILE")"
echo "Submitted $MODE: $JID on $NODE"
echo "Run directory: $RUN_DIR"
echo "tail -f $LOG_DIR/official_${MODE}_${STAMP}_${JID%%;*}.out"
echo "After completion upload: ${RUN_DIR}_bundle.tar.gz"
