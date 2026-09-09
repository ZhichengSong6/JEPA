#!/usr/bin/env bash
# Formal evaluation for teacher-replacement Exp1.
#
# Runs:
#   1) paired original-MH vs Exp1 official CEM at B=500 and B=1000,
#      seeds 42..46, 100 episodes each.
#   2) the same 40-anchor teacher-response oracle diagnostic with
#      teacher=original MH and student=Exp1.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_teacher_exp1_eval.sh formal

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_teacher_exp1_eval.sh formal" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "ERROR: eval is restricted to 4090node2 or 4090node3." >&2
  exit 3
fi

MH_POLICY="pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10"
EXP1_POLICY="pusht_mh_ald_teacher_mh_h5_seed3072_ep10_ddp4/lewm_mh_ald_teacher_mh_h5_ddp4_epoch_10"

for P in "$MH_POLICY" "$EXP1_POLICY"; do
  [[ -s "$STABLEWM_HOME/${P}_object.ckpt" ]] || {
    echo "ERROR: missing model: $STABLEWM_HOME/${P}_object.ckpt" >&2
    exit 4
  }
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$REPO/outputs/mh_teacher_exp1_eval/$RUN_TAG"
CEM_DIR="$RUN_ROOT/cem"
RESP_DIR="$RUN_ROOT/teacher_response"
SUM_DIR="$RUN_ROOT/summary"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_teacher_exp1_eval"
mkdir -p "$CEM_DIR" "$RESP_DIR" "$SUM_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$RUN_ROOT" "$REPO/outputs/mh_teacher_exp1_eval_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile \
  eval_mh_ald_teacher_response_oracle.py \
  scripts/summarize_mh_teacher_exp1_eval.py

FILE="$GEN_DIR/mh_teacher_exp1_eval_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=mht1_eval
#SBATCH --output=$LOG_DIR/mh_teacher_exp1_eval_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_teacher_exp1_eval_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

run_one () {
  local GPU="\$1"
  local LABEL="\$2"
  local POLICY="\$3"
  local SEED="\$4"
  local BUDGET="\$5"

  if [[ "\$BUDGET" == "500" ]]; then
    I=5
  elif [[ "\$BUDGET" == "1000" ]]; then
    I=10
  else
    echo "bad budget \$BUDGET" >&2
    return 2
  fi

  OUT="$CEM_DIR/\${LABEL}_seed\${SEED}_b\${BUDGET}.txt"
  echo "--- \$LABEL seed=\$SEED B=\$BUDGET N=100 I=\$I K=10 GPU=\$GPU ---"
  CUDA_VISIBLE_DEVICES="\$GPU" python -u eval.py \
    --config-name=pusht.yaml \
    policy="\$POLICY" \
    seed="\$SEED" \
    solver.num_samples=100 \
    solver.n_steps="\$I" \
    solver.topk=10 \
    eval.num_eval=100 \
    eval.eval_budget=50 \
    output.filename="\$OUT"
}

run_seed_group () {
  local GPU="\$1"
  local LABEL="\$2"
  local POLICY="\$3"
  shift 3
  for SEED in "\$@"; do
    run_one "\$GPU" "\$LABEL" "\$POLICY" "\$SEED" 500
    run_one "\$GPU" "\$LABEL" "\$POLICY" "\$SEED" 1000
  done
}

echo "============================================================"
echo "Teacher replacement Exp1 formal evaluation"
echo "node=\$(hostname)"
echo "run_root=$RUN_ROOT"
echo "MH=$MH_POLICY"
echo "Exp1=$EXP1_POLICY"
echo "============================================================"

echo
echo "=== A: paired official CEM ==="
run_seed_group 0 mh "$MH_POLICY" 42 43 44 &
PID0=\$!
run_seed_group 1 mh "$MH_POLICY" 45 46 &
PID1=\$!
run_seed_group 2 exp1 "$EXP1_POLICY" 42 43 44 &
PID2=\$!
run_seed_group 3 exp1 "$EXP1_POLICY" 45 46 &
PID3=\$!

wait "\$PID0"
wait "\$PID1"
wait "\$PID2"
wait "\$PID3"

echo
echo "=== B: paired CEM summary ==="
python -u scripts/summarize_mh_teacher_exp1_eval.py \
  --input-dir "$CEM_DIR" \
  --output-dir "$SUM_DIR"

echo
echo "=== C: teacher-response oracle (original MH teacher -> Exp1 student) ==="
CUDA_VISIBLE_DEVICES=3 python -u eval_mh_ald_teacher_response_oracle.py \
  --teacher-policy "$MH_POLICY" \
  --student-policy "$EXP1_POLICY" \
  --num-anchors 40 \
  --directions-per-position 4 \
  --perturb-radius 0.1565 \
  --history-size 3 \
  --horizon 5 \
  --action-block 5 \
  --seed 42 \
  --env-seed 12000 \
  --device cuda:0 \
  --output-dir "$RESP_DIR"

echo
echo "=== EXP1 EVAL ARTIFACT VALIDATION ==="
for f in \
  "$SUM_DIR/paired_seed_results.csv" \
  "$SUM_DIR/paired_summary.json" \
  "$RESP_DIR/response_cells.csv" \
  "$RESP_DIR/anchor_metrics.csv" \
  "$RESP_DIR/summary.json"
do
  [[ -s "\$f" ]] || { echo "ERROR: missing/empty \$f" >&2; exit 8; }
done

echo
echo "=== EXP1 EVAL DONE ==="
echo "run_root=$RUN_ROOT"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted Exp1 eval: $JID"
echo "node: $NODE"
echo "run_root: $RUN_ROOT"
echo "stdout: tail -f $LOG_DIR/mh_teacher_exp1_eval_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/mh_teacher_exp1_eval_${RUN_TAG}_${JID}.err"
