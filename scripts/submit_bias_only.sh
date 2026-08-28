#!/usr/bin/env bash
# Train and evaluate strict Bias-Only calibration.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_bias_only.sh smoke
#   NODE=4090node3 bash scripts/submit_bias_only.sh formal
#
# Formal outputs:
#   model:
#     $STABLEWM_HOME/pusht_bias_only_h5_seed3072_ep10_ddp4/
#   offline endpoint eval:
#     <repo>/outputs/bias_only_offline_formal/
#   CEM iteration-budget eval:
#     <repo>/outputs/bias_only_cem_budget/
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_bias_only.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

SMOKE_DIR="pusht_bias_only_smoke_ddp4"
SMOKE_MODEL="lewm_bias_only_smoke_ddp4"
FORMAL_DIR="pusht_bias_only_h5_seed3072_ep10_ddp4"
FORMAL_MODEL="lewm_bias_only_h5_ddp4"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_bias_only"
OFFLINE_DIR="$REPO/outputs/bias_only_offline_formal"
CEM_DIR="$REPO/outputs/bias_only_cem_budget"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  bias_only_calibration.py \
  train_bias_only.py \
  bias_cem.py \
  eval_bias_only_offline.py

COMMON_HEADER=$(cat <<EOF
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$WORLD_SIZE
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:$WORLD_SIZE
EOF
)

COMMON_SETUP=$(cat <<EOF
set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
EOF
)

COMMON_HYDRA="seed=3072 num_workers=$NUM_WORKERS_PER_RANK trainer.devices=$WORLD_SIZE trainer.num_nodes=1 trainer.strategy=ddp trainer.sync_batchnorm=false loader.batch_size=$LOCAL_BATCH"

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
  FILE="$GEN_DIR/bias_only_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=bias_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/bias_only_smoke_%j.out
#SBATCH --error=$LOG_DIR/bias_only_smoke_%j.err

$COMMON_SETUP

echo "=== S1: 2-step Bias-Only DDP smoke training ==="
srun --kill-on-bad-exit=1 python -u train_bias_only.py \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR

SMOKE_POLICY="$SMOKE_DIR/${SMOKE_MODEL}_epoch_1"

echo
echo "=== S2: tiny Bias-Only CEM load/planning smoke ==="
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  --config-name=pusht.yaml \
  solver=bias_cem \
  policy="\$SMOKE_POLICY" \
  solver.n_steps=2 \
  solver.num_samples=30 \
  solver.topk=3 \
  eval.num_eval=2 \
  output.filename=bias_only_smoke_eval.txt

echo
echo "=== DONE ==="
EOF
else
  if [[ -e "$STABLEWM_HOME/$FORMAL_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "ERROR: formal output already exists: $STABLEWM_HOME/$FORMAL_DIR" >&2
    echo "Rename/remove it, or set ALLOW_EXISTING=1 intentionally." >&2
    exit 3
  fi

  rm -rf "$OFFLINE_DIR" "$CEM_DIR"
  mkdir -p "$OFFLINE_DIR" "$CEM_DIR"

  FILE="$GEN_DIR/bias_only_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=bias_h5_4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/bias_only_formal_%j.out
#SBATCH --error=$LOG_DIR/bias_only_formal_%j.err

$COMMON_SETUP

echo "============================================================"
echo "Strict Bias-Only formal experiment"
echo "node=\$(hostname)"
echo "Frozen model: official LeWM epoch 10"
echo "Trainable: one 2-layer latent bias calibrator"
echo "Loss: endpoint calibration MSE only"
echo "Global batch: $GLOBAL_BATCH"
echo "============================================================"

echo
echo "=== B1: Bias-Only training ==="
srun --kill-on-bad-exit=1 python -u train_bias_only.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR

POLICY="$FORMAL_DIR/${FORMAL_MODEL}_epoch_10"

echo
echo "=== B2: held-out endpoint calibration ==="
CUDA_VISIBLE_DEVICES=0 python -u eval_bias_only_offline.py \
  --policy "\$POLICY" \
  --seed 3072 \
  --num-batches 50 \
  --batch-size 64 \
  --output-dir "$OFFLINE_DIR"

echo
echo "=== B3: paired CEM iteration-budget evaluation ==="
for I in 1 3 5 10 30; do
  OUT="$CEM_DIR/bias_only_n300_i${I}_ep100.txt"
  rm -f "\$OUT"
  echo
  echo "--- Bias-Only CEM: N=300 I=\$I K=30 episodes=100 ---"
  CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --config-name=pusht.yaml \
    solver=bias_cem \
    policy="\$POLICY" \
    seed=42 \
    solver.num_samples=300 \
    solver.n_steps="\$I" \
    solver.topk=30 \
    eval.num_eval=100 \
    eval.eval_budget=50 \
    output.filename="\$OUT"
done

echo
echo "=== DONE ==="
echo "Policy: \$POLICY"
echo "Offline: $OFFLINE_DIR"
echo "CEM: $CEM_DIR"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted Bias-Only ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "Log:"
  echo "  tail -f $LOG_DIR/bias_only_smoke_${JID}.out"
  echo "After === DONE ===:"
  echo "  NODE=$NODE bash scripts/submit_bias_only.sh formal"
else
  echo "Log:"
  echo "  tail -f $LOG_DIR/bias_only_formal_${JID}.out"
  echo "After === DONE ===:"
  echo "  bash scripts/package_bias_only_results.sh"
  echo "Final policy:"
  echo "  $FORMAL_DIR/${FORMAL_MODEL}_epoch_10"
fi
