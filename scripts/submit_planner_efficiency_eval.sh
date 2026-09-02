#!/usr/bin/env bash
# Planner-efficiency evaluation: causal-prefix diagnostic + 2-D CEM budget sweep.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_planner_efficiency_eval.sh smoke
#   NODE=4090node3 bash scripts/submit_planner_efficiency_eval.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_planner_efficiency_eval.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

LEWM_POLICY="lewm_epoch_10"
FULL_ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"
PURE_ALD_POLICY="pusht_pure_ald_h5_seed3072_ep10_ddp4/lewm_pure_ald_h5_ddp4_epoch_10"
ALD_TF_POLICY="pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10"
ALD_ROLLOUT_POLICY="pusht_ald_rollout_h5_seed3072_ep10_ddp4/lewm_ald_rollout_h5_ddp4_epoch_10"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_planner_efficiency"
CTX_DIR="$REPO/outputs/planner_efficiency_context"
CEM_DIR="$REPO/outputs/planner_efficiency_cem"
SUM_DIR="$REPO/outputs/planner_efficiency_summary"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile   eval_pusht_context_prefix.py   eval_pusht_horizon_directional.py   scripts/summarize_cem_budget_sweep.py

COMMON_HEADER=$(cat <<EOF
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
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
export OMP_NUM_THREADS=4
EOF
)

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/planner_efficiency_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=peff_sm
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/planner_efficiency_smoke_%j.out
#SBATCH --error=$LOG_DIR/planner_efficiency_smoke_%j.err

$COMMON_SETUP

rm -rf "$CTX_DIR/smoke" "$CEM_DIR/smoke" "$SUM_DIR/smoke"
mkdir -p "$CTX_DIR/smoke" "$CEM_DIR/smoke" "$SUM_DIR/smoke"

echo "=== S1: context-prefix smoke ==="
CUDA_VISIBLE_DEVICES=0 python -u eval_pusht_context_prefix.py   --policies "$LEWM_POLICY" "$FULL_ALD_POLICY" "$PURE_ALD_POLICY" "$ALD_TF_POLICY" "$ALD_ROLLOUT_POLICY"   --labels lewm full_ald pure_ald ald_tf ald_rollout   --num-anchors 2   --contexts 1 2 3   --horizon 5   --radius 0.1565   --num-directions 2   --seed 42   --device cuda:0   --output-dir "$CTX_DIR/smoke"

echo "=== S2: tiny budget smoke ==="
for LABEL in lewm ald_tf; do
  if [[ "\$LABEL" == "lewm" ]]; then POLICY="$LEWM_POLICY"; else POLICY="$ALD_TF_POLICY"; fi
  OUT="$CEM_DIR/smoke/\${LABEL}_n30_i1_k3_ep2.txt"
  CUDA_VISIBLE_DEVICES=0 python -u eval.py     --config-name=pusht.yaml     policy="\$POLICY" seed=42     solver.num_samples=30 solver.n_steps=1 solver.topk=3     eval.num_eval=2 eval.eval_budget=50 output.filename="\$OUT"
done

echo "=== DONE ==="
EOF
else
  FILE="$GEN_DIR/planner_efficiency_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=peff_h5
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/planner_efficiency_formal_%j.out
#SBATCH --error=$LOG_DIR/planner_efficiency_formal_%j.err

$COMMON_SETUP

rm -rf "$CTX_DIR" "$CEM_DIR" "$SUM_DIR"
mkdir -p "$CTX_DIR" "$CEM_DIR" "$SUM_DIR"

echo "============================================================"
echo "Planner-efficiency formal evaluation"
echo "node=\$(hostname)"
echo "Primary target: >=10x reduction versus LeWM N=300,I=10 (B=3000, ~94%)"
echo "Thus >=10x target region is B=N*I <= 300."
echo "Elite fraction fixed at 10%: N=30->K=3, N=100->K=10, N=300->K=30."
echo "============================================================"

echo
echo "=== A: causal-prefix diagnostic, C={1,2,3} ==="
CUDA_VISIBLE_DEVICES=3 python -u eval_pusht_context_prefix.py   --policies     "$LEWM_POLICY"     "$FULL_ALD_POLICY"     "$PURE_ALD_POLICY"     "$ALD_TF_POLICY"     "$ALD_ROLLOUT_POLICY"   --labels lewm full_ald pure_ald ald_tf ald_rollout   --num-anchors 50   --contexts 1 2 3   --horizon 5   --radius 0.1565   --num-directions 32   --seed 42   --device cuda:0   --output-dir "$CTX_DIR"

run_sweep () {
  local GPU="\$1"
  local LABEL="\$2"
  local POLICY="\$3"

  for N in 30 100 300; do
    K=\$((N / 10))
    for I in 1 3 5 10; do
      OUT="$CEM_DIR/\${LABEL}_n\${N}_i\${I}_k\${K}_ep100.txt"
      rm -f "\$OUT"
      echo "--- \$LABEL GPU=\$GPU N=\$N I=\$I K=\$K B=\$((N*I)) ---"
      CUDA_VISIBLE_DEVICES="\$GPU" python -u eval.py         --config-name=pusht.yaml         policy="\$POLICY"         seed=42         solver.num_samples="\$N"         solver.n_steps="\$I"         solver.topk="\$K"         eval.num_eval=100         eval.eval_budget=50         output.filename="\$OUT"
    done
  done
}

echo
echo "=== B: 2-D CEM budget sweep (three models in parallel) ==="
run_sweep 0 lewm "$LEWM_POLICY" &
PID0=\$!
run_sweep 1 full_ald "$FULL_ALD_POLICY" &
PID1=\$!
run_sweep 2 ald_tf "$ALD_TF_POLICY" &
PID2=\$!

wait "\$PID0"
wait "\$PID1"
wait "\$PID2"

echo
echo "=== C: summarize Success-vs-Budget Pareto ==="
python -u scripts/summarize_cem_budget_sweep.py   --input-dir "$CEM_DIR"   --output-dir "$SUM_DIR"   --reference-label lewm   --reference-n 300   --reference-i 10   --thresholds 80 90 94 95 100

echo
echo "=== DONE ==="
echo "Context diagnostic: $CTX_DIR"
echo "Raw CEM sweep      : $CEM_DIR"
echo "Budget summary     : $SUM_DIR"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted planner-efficiency eval ($MODE): $JID"
echo "node: $NODE"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "tail -f $LOG_DIR/planner_efficiency_smoke_${JID}.out"
else
  echo "tail -f $LOG_DIR/planner_efficiency_formal_${JID}.out"
  echo "After === DONE ===:"
  echo "  bash scripts/package_planner_efficiency_eval.sh"
fi
