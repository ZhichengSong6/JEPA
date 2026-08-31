#!/usr/bin/env bash
# Minimal ALD loss-composition ablation on full PushT.
#
# Two controlled variants:
#   A) ALD + TF       : tf_weight=1, rollout_weight=0, ald.weight=1
#   B) ALD + rollout  : tf_weight=0, rollout_weight=1, ald.weight=1
#
# Everything else is identical to the established Full-ALD run:
#   official LeWM epoch-10 init/teacher, H=5, radius=0.1565, Kprobe=4,
#   seed=3072, global batch=64, 10 epochs, 4-GPU DDP.
#
# Usage:
#   NODE=4090node2 bash scripts/submit_ald_minimal_ablation.sh smoke
#   NODE=4090node2 bash scripts/submit_ald_minimal_ablation.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_ald_minimal_ablation.sh {smoke|formal}" >&2
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

TF_DIR="pusht_ald_tf_h5_seed3072_ep10_ddp4"
TF_MODEL="lewm_ald_tf_h5_ddp4"
RO_DIR="pusht_ald_rollout_h5_seed3072_ep10_ddp4"
RO_MODEL="lewm_ald_rollout_h5_ddp4"

TF_SMOKE_DIR="pusht_ald_tf_smoke_ddp4"
TF_SMOKE_MODEL="lewm_ald_tf_smoke_ddp4"
RO_SMOKE_DIR="pusht_ald_rollout_smoke_ddp4"
RO_SMOKE_MODEL="lewm_ald_rollout_smoke_ddp4"

LEWM_POLICY="lewm_epoch_10"
FULL_ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"
PURE_ALD_POLICY="pusht_pure_ald_h5_seed3072_ep10_ddp4/lewm_pure_ald_h5_ddp4_epoch_10"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_ald_minimal_ablation"
LOCAL_DIR="$REPO/outputs/ald_minimal_ablation_h5_local"
CEM_DIR="$REPO/outputs/ald_minimal_ablation_cem"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  stage1_bias_calibration.py \
  anchored_local_dynamics.py \
  train_ald.py \
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
  rm -rf \
    "$STABLEWM_HOME/$TF_SMOKE_DIR" \
    "$STABLEWM_HOME/$RO_SMOKE_DIR"

  FILE="$GEN_DIR/ald_minimal_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=aldab_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/ald_minimal_smoke_%j.out
#SBATCH --error=$LOG_DIR/ald_minimal_smoke_%j.err

$COMMON_SETUP

echo "=== S1: ALD+TF two-step smoke ==="
srun --kill-on-bad-exit=1 python -u train_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  ald.tf_weight=1.0 \
  ald.rollout_weight=0.0 \
  ald.weight=1.0 \
  output_model_name=$TF_SMOKE_MODEL \
  subdir=$TF_SMOKE_DIR

echo
echo "=== S2: ALD+rollout two-step smoke ==="
srun --kill-on-bad-exit=1 python -u train_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  ald.tf_weight=0.0 \
  ald.rollout_weight=1.0 \
  ald.weight=1.0 \
  output_model_name=$RO_SMOKE_MODEL \
  subdir=$RO_SMOKE_DIR

TF_SMOKE_POLICY="$TF_SMOKE_DIR/${TF_SMOKE_MODEL}_epoch_1"
RO_SMOKE_POLICY="$RO_SMOKE_DIR/${RO_SMOKE_MODEL}_epoch_1"

echo
echo "=== S3: tiny standard-CEM load/planning smoke ==="
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  --config-name=pusht.yaml \
  policy="\$TF_SMOKE_POLICY" \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  eval.num_eval=2 \
  eval.eval_budget=50 \
  output.filename=ald_tf_smoke_eval.txt

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  --config-name=pusht.yaml \
  policy="\$RO_SMOKE_POLICY" \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  eval.num_eval=2 \
  eval.eval_budget=50 \
  output.filename=ald_rollout_smoke_eval.txt

echo
echo "=== DONE ==="
EOF

else
  for d in "$TF_DIR" "$RO_DIR"; do
    if [[ -e "$STABLEWM_HOME/$d" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
      echo "ERROR: formal output already exists: $STABLEWM_HOME/$d" >&2
      echo "Rename/remove it, or intentionally set ALLOW_EXISTING=1." >&2
      exit 3
    fi
  done

  rm -rf "$LOCAL_DIR" "$CEM_DIR"
  mkdir -p "$LOCAL_DIR" "$CEM_DIR"

  FILE="$GEN_DIR/ald_minimal_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=aldab_h5_4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/ald_minimal_formal_%j.out
#SBATCH --error=$LOG_DIR/ald_minimal_formal_%j.err

$COMMON_SETUP

echo "============================================================"
echo "Minimal ALD loss-composition ablation"
echo "node=\$(hostname)"
echo "Variant A: L = L_ALD + L_TF"
echo "Variant B: L = L_ALD + L_rollout"
echo "All other settings match Full-ALD exactly."
echo "Global batch: $GLOBAL_BATCH"
echo "============================================================"

echo
echo "=== A1: train ALD+TF, 10 epochs ==="
srun --kill-on-bad-exit=1 python -u train_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  ald.tf_weight=1.0 \
  ald.rollout_weight=0.0 \
  ald.weight=1.0 \
  output_model_name=$TF_MODEL \
  subdir=$TF_DIR

echo
echo "=== A2: train ALD+rollout, 10 epochs ==="
srun --kill-on-bad-exit=1 python -u train_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  ald.tf_weight=0.0 \
  ald.rollout_weight=1.0 \
  ald.weight=1.0 \
  output_model_name=$RO_MODEL \
  subdir=$RO_DIR

TF_POLICY="$TF_DIR/${TF_MODEL}_epoch_10"
RO_POLICY="$RO_DIR/${RO_MODEL}_epoch_10"

echo
echo "=== B1: matched H5 local diagnostic ==="
rm -rf "$LOCAL_DIR"
CUDA_VISIBLE_DEVICES=0 python -u eval_pusht_horizon_directional.py \
  --policies \
    "$LEWM_POLICY" \
    "$FULL_ALD_POLICY" \
    "$PURE_ALD_POLICY" \
    "\$TF_POLICY" \
    "\$RO_POLICY" \
  --labels \
    lewm \
    full_ald \
    pure_ald \
    ald_tf \
    ald_rollout \
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
echo "=== B2: 100-episode CEM iteration sweeps ==="
for LABEL in ald_tf ald_rollout; do
  if [[ "\$LABEL" == "ald_tf" ]]; then
    POLICY="\$TF_POLICY"
  else
    POLICY="\$RO_POLICY"
  fi

  for I in 1 3 5 10 30; do
    OUT="$CEM_DIR/\${LABEL}_n300_i\${I}_ep100.txt"
    rm -f "\$OUT"
    echo
    echo "--- \$LABEL: N=300 I=\$I K=30 episodes=100 ---"
    CUDA_VISIBLE_DEVICES=0 python -u eval.py \
      --config-name=pusht.yaml \
      policy="\$POLICY" \
      seed=42 \
      solver.num_samples=300 \
      solver.n_steps="\$I" \
      solver.topk=30 \
      eval.num_eval=100 \
      eval.eval_budget=50 \
      output.filename="\$OUT"
  done
done

echo
echo "=== DONE ==="
echo "ALD+TF policy: \$TF_POLICY"
echo "ALD+rollout policy: \$RO_POLICY"
echo "Local diagnostic: $LOCAL_DIR"
echo "CEM results: $CEM_DIR"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted minimal ALD ablation ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "Log:"
  echo "  tail -f $LOG_DIR/ald_minimal_smoke_${JID}.out"
  echo "After === DONE ===:"
  echo "  NODE=$NODE bash scripts/submit_ald_minimal_ablation.sh formal"
else
  echo "Log:"
  echo "  tail -f $LOG_DIR/ald_minimal_formal_${JID}.out"
  echo "After === DONE ===:"
  echo "  bash scripts/package_ald_minimal_ablation_results.sh"
fi
