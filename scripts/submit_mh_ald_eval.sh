#!/usr/bin/env bash
# Evaluate Multi-Horizon ALD (MH-ALD) against official LeWM and successful ALD.
#
# Usage:
#   NODE=4090node2 bash scripts/submit_mh_ald_eval.sh smoke
#   NODE=4090node2 bash scripts/submit_mh_ald_eval.sh formal

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_eval.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "ERROR: eval jobs are restricted to 4090node2 or 4090node3." >&2
  exit 3
fi

LEWM_POLICY="lewm_epoch_10"
ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"
MHALD_POLICY="pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10"

MHALD_CKPT="$STABLEWM_HOME/${MHALD_POLICY}_object.ckpt"
if [[ ! -f "$MHALD_CKPT" ]]; then
  echo "ERROR: MH-ALD checkpoint not found:" >&2
  echo "  $MHALD_CKPT" >&2
  exit 4
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$REPO/outputs/mh_ald_eval_runs/$RUN_TAG"
HORIZON_DIR="$RUN_ROOT/horizon"
CEM_DIR="$RUN_ROOT/cem"
SUM_DIR="$RUN_ROOT/summary"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_eval"

mkdir -p "$HORIZON_DIR" "$CEM_DIR" "$SUM_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$RUN_ROOT" "$REPO/outputs/mh_ald_eval_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  eval_pusht_horizon_directional.py \
  scripts/summarize_cem_budget_sweep.py

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/mh_ald_eval_smoke_${RUN_TAG}.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=mhald_ev_sm
#SBATCH --output=$LOG_DIR/mh_ald_eval_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_ald_eval_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "============================================================"
echo "MH-ALD eval smoke"
echo "run_tag=$RUN_TAG"
echo "node=\$(hostname)"
echo "run_root=$RUN_ROOT"
echo "============================================================"

echo
echo "=== A: tiny H1/H5 directional diagnostic ==="
CUDA_VISIBLE_DEVICES=3 python -u eval_pusht_horizon_directional.py \
  --policies "$LEWM_POLICY" "$ALD_POLICY" "$MHALD_POLICY" \
  --labels lewm ald mh_ald \
  --num-anchors 2 \
  --horizons 1 5 \
  --reference-horizon 5 \
  --radii 0.1565 \
  --radius-scaling rms \
  --num-directions 2 \
  --seed 42 \
  --device cuda:0 \
  --output-dir "$HORIZON_DIR"

echo
echo "=== B: tiny CEM smoke ==="
for SPEC in \
  "0 lewm $LEWM_POLICY" \
  "1 ald $ALD_POLICY" \
  "2 mh_ald $MHALD_POLICY"
do
  read -r GPU LABEL POLICY <<< "\$SPEC"
  OUT="$CEM_DIR/\${LABEL}_n30_i1_k3_ep2.txt"
  CUDA_VISIBLE_DEVICES="\$GPU" python -u eval.py \
    --config-name=pusht.yaml \
    policy="\$POLICY" \
    seed=42 \
    solver.num_samples=30 \
    solver.n_steps=1 \
    solver.topk=3 \
    eval.num_eval=2 \
    eval.eval_budget=50 \
    output.filename="\$OUT"
done

echo
echo "=== SMOKE DONE ==="
echo "run_root=$RUN_ROOT"
EOF
else
  FILE="$GEN_DIR/mh_ald_eval_formal_${RUN_TAG}.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=mhald_eval
#SBATCH --output=$LOG_DIR/mh_ald_eval_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_ald_eval_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

run_sweep () {
  local GPU="\$1"
  local LABEL="\$2"
  local POLICY="\$3"

  for N in 30 100 300; do
    K=\$((N / 10))
    for I in 1 3 5 10; do
      OUT="$CEM_DIR/\${LABEL}_n\${N}_i\${I}_k\${K}_ep100.txt"
      echo "--- \$LABEL GPU=\$GPU N=\$N I=\$I K=\$K B=\$((N*I)) ---"
      CUDA_VISIBLE_DEVICES="\$GPU" python -u eval.py \
        --config-name=pusht.yaml \
        policy="\$POLICY" \
        seed=42 \
        solver.num_samples="\$N" \
        solver.n_steps="\$I" \
        solver.topk="\$K" \
        eval.num_eval=100 \
        eval.eval_budget=50 \
        output.filename="\$OUT"
    done
  done

  OUT="$CEM_DIR/\${LABEL}_n300_i30_k30_ep100.txt"
  echo "--- \$LABEL GPU=\$GPU N=300 I=30 K=30 B=9000 [FULL] ---"
  CUDA_VISIBLE_DEVICES="\$GPU" python -u eval.py \
    --config-name=pusht.yaml \
    policy="\$POLICY" \
    seed=42 \
    solver.num_samples=300 \
    solver.n_steps=30 \
    solver.topk=30 \
    eval.num_eval=100 \
    eval.eval_budget=50 \
    output.filename="\$OUT"
}

echo "============================================================"
echo "MH-ALD formal evaluation"
echo "run_tag=$RUN_TAG"
echo "node=\$(hostname)"
echo "run_root=$RUN_ROOT"
echo "============================================================"

echo
echo "=== A: H1-H5 directional / ranking diagnostic ==="
CUDA_VISIBLE_DEVICES=3 python -u eval_pusht_horizon_directional.py \
  --policies "$LEWM_POLICY" "$ALD_POLICY" "$MHALD_POLICY" \
  --labels lewm ald mh_ald \
  --num-anchors 50 \
  --horizons 1 2 3 4 5 \
  --reference-horizon 5 \
  --radii 0.1565 0.35 0.70 \
  --radius-scaling rms \
  --num-directions 32 \
  --seed 42 \
  --device cuda:0 \
  --output-dir "$HORIZON_DIR" &
PID_H=\$!

echo
echo "=== B: CEM budget sweeps ==="
run_sweep 0 lewm "$LEWM_POLICY" &
PID0=\$!
run_sweep 1 ald "$ALD_POLICY" &
PID1=\$!
run_sweep 2 mh_ald "$MHALD_POLICY" &
PID2=\$!

wait "\$PID_H"
wait "\$PID0"
wait "\$PID1"
wait "\$PID2"

echo
echo "=== C: summarize success-vs-budget ==="
python -u scripts/summarize_cem_budget_sweep.py \
  --input-dir "$CEM_DIR" \
  --output-dir "$SUM_DIR" \
  --reference-label lewm \
  --reference-n 300 \
  --reference-i 10 \
  --thresholds 80 90 91 94 95 100

echo
echo "=== FORMAL DONE ==="
echo "H1-H5 diagnostic : $HORIZON_DIR"
echo "Raw CEM results  : $CEM_DIR"
echo "Budget summary   : $SUM_DIR"
echo "Run root         : $RUN_ROOT"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted MH-ALD eval ($MODE): $JID"
echo "node     : $NODE"
echo "run_tag  : $RUN_TAG"
echo "run_root : $RUN_ROOT"
echo
squeue -j "$JID" -o "%.18i %.14j %.2t %.12M %.12l %.20R" || true
echo
echo "stdout: tail -f $LOG_DIR/mh_ald_eval_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/mh_ald_eval_${RUN_TAG}_${JID}.err"
