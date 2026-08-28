#!/usr/bin/env bash
# Submit the controlled Pure-ALD experiment.
#
# Usage:
#   NODE=4090node2 bash scripts/submit_pure_ald.sh smoke
#   NODE=4090node2 bash scripts/submit_pure_ald.sh formal
#
# Formal automatically runs:
#   B1) 10-epoch 4-GPU Pure-ALD training
#   B2) matched H5 local landscape diagnostic: LeWM vs Full-ALD vs Pure-ALD
#   B3) 100-episode CEM sweep for Pure-ALD, N=300 K=30 I={1,3,5,10,30}
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_pure_ald.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"

WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

SMOKE_DIR="pusht_pure_ald_smoke_ddp4"
SMOKE_MODEL="lewm_pure_ald_smoke_ddp4"
FORMAL_DIR="pusht_pure_ald_h5_seed3072_ep10_ddp4"
FORMAL_MODEL="lewm_pure_ald_h5_ddp4"

LEWM_POLICY="lewm_epoch_10"
FULL_ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_pure_ald"
LOCAL_DIR="$REPO/outputs/pure_ald_h5_local_formal"
CEM_DIR="$REPO/outputs/pure_ald_cem_budget"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  pure_ald.py \
  train_pure_ald.py \
  anchored_local_dynamics.py \
  stage1_bias_calibration.py \
  eval_pusht_horizon_directional.py

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
  FILE="$GEN_DIR/pure_ald_smoke.slurm"

  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=pald_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/pure_ald_smoke_%j.out
#SBATCH --error=$LOG_DIR/pure_ald_smoke_%j.err

$COMMON_SETUP

echo "=== S1: 2-step Pure-ALD 4-GPU DDP training ==="
srun --kill-on-bad-exit=1 python -u train_pure_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR

SMOKE_POLICY="$SMOKE_DIR/${SMOKE_MODEL}_epoch_1"

echo
echo "=== S2: tiny standard-CEM policy load/planning smoke ==="
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  --config-name=pusht.yaml \
  policy="\$SMOKE_POLICY" \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  eval.num_eval=2 \
  eval.eval_budget=50 \
  output.filename=pure_ald_smoke_eval.txt

echo
echo "=== DONE ==="
EOF

else
  if [[ -e "$STABLEWM_HOME/$FORMAL_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "ERROR: formal output already exists: $STABLEWM_HOME/$FORMAL_DIR" >&2
    echo "Rename/remove it, or set ALLOW_EXISTING=1 intentionally." >&2
    exit 3
  fi

  rm -rf "$LOCAL_DIR" "$CEM_DIR"
  mkdir -p "$LOCAL_DIR" "$CEM_DIR"

  FILE="$GEN_DIR/pure_ald_formal.slurm"

  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=pald_h5_4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/pure_ald_formal_%j.out
#SBATCH --error=$LOG_DIR/pure_ald_formal_%j.err

$COMMON_SETUP

echo "============================================================"
echo "Pure-ALD formal controlled ablation"
echo "node=\$(hostname)"
echo "Student init: official LeWM epoch 10"
echo "Frozen: encoder + projector + teacher"
echo "Trainable: action_encoder + predictor + pred_proj"
echo "ONLY objective: L = L_ALD"
echo "H=5 radius=0.1565 Kprobe=4"
echo "Global batch: $GLOBAL_BATCH"
echo "============================================================"

echo
echo "=== B1: 10-epoch Pure-ALD training ==="
srun --kill-on-bad-exit=1 python -u train_pure_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR

PURE_POLICY="$FORMAL_DIR/${FORMAL_MODEL}_epoch_10"

echo
echo "=== B2: matched H5 local diagnostic ==="
rm -rf "$LOCAL_DIR"
CUDA_VISIBLE_DEVICES=0 python -u eval_pusht_horizon_directional.py \
  --policies "$LEWM_POLICY" "$FULL_ALD_POLICY" "\$PURE_POLICY" \
  --labels lewm full_ald pure_ald \
  --num-anchors 50 \
  --horizons 5 \
  --reference-horizon 5 \
  --radii 0.1565 \
  --radius-scaling fixed \
  --num-directions 32 \
  --seed 42 \
  --device cuda:0 \
  --output-dir "$LOCAL_DIR"

echo
echo "=== B3: 100-episode Pure-ALD CEM iteration sweep ==="
for I in 1 3 5 10 30; do
  OUT="$CEM_DIR/pure_ald_n300_i\${I}_ep100.txt"
  rm -f "\$OUT"
  echo
  echo "--- Pure-ALD: N=300 I=\$I K=30 episodes=100 ---"
  CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --config-name=pusht.yaml \
    policy="\$PURE_POLICY" \
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
echo "Policy: \$PURE_POLICY"
echo "Local diagnostic: $LOCAL_DIR"
echo "CEM results: $CEM_DIR"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted Pure-ALD ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "Log:"
  echo "  tail -f $LOG_DIR/pure_ald_smoke_${JID}.out"
  echo "After === DONE ===:"
  echo "  NODE=$NODE bash scripts/submit_pure_ald.sh formal"
else
  echo "Log:"
  echo "  tail -f $LOG_DIR/pure_ald_formal_${JID}.out"
  echo "After === DONE ===:"
  echo "  bash scripts/package_pure_ald_results.sh"
  echo "Final policy:"
  echo "  $FORMAL_DIR/${FORMAL_MODEL}_epoch_10"
fi
