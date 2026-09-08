#!/usr/bin/env bash
# MH-ALD teacher-response oracle diagnostic.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_teacher_response_oracle.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_teacher_response_oracle.sh formal

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/submit_mh_ald_teacher_response_oracle.sh {smoke|formal}" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_teacher_response_oracle.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO/outputs/mh_ald_teacher_response_oracle/$RUN_TAG"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_teacher_response_oracle"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$OUT_DIR" "$REPO/outputs/mh_ald_teacher_response_oracle_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile eval_mh_ald_teacher_response_oracle.py

if [[ "$MODE" == "smoke" ]]; then
  NUM_ANCHORS=4
  DIRECTIONS=1
else
  NUM_ANCHORS=40
  DIRECTIONS=4
fi

FILE="$GEN_DIR/mh_ald_teacher_response_oracle_${MODE}_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=mh_tresp
#SBATCH --output=$LOG_DIR/mh_teacher_response_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_teacher_response_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "============================================================"
echo "MH-ALD teacher response oracle ($MODE)"
echo "node=\$(hostname)"
echo "output=$OUT_DIR"
echo "anchors=$NUM_ANCHORS directions/position=$DIRECTIONS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_mh_ald_teacher_response_oracle.py \
  --num-anchors $NUM_ANCHORS \
  --directions-per-position $DIRECTIONS \
  --perturb-radius 0.1565 \
  --history-size 3 \
  --horizon 5 \
  --action-block 5 \
  --seed 42 \
  --env-seed 12000 \
  --output-dir "$OUT_DIR"

echo
echo "=== TEACHER RESPONSE ARTIFACT VALIDATION ==="
REQ=(
  "$OUT_DIR/response_cells.csv"
  "$OUT_DIR/anchor_metrics.csv"
  "$OUT_DIR/summary.json"
)
for f in "\${REQ[@]}"; do
  [[ -s "\$f" ]] || { echo "ERROR: missing/empty \$f" >&2; exit 4; }
done

python - <<'PY'
import json
from pathlib import Path
p = Path("$OUT_DIR") / "summary.json"
s = json.loads(p.read_text())
mx = float(s["latent_frame_max_abs_teacher_vs_student"])
good = int(s["replay_good"]["anchor"]["replay_endpoint_factor_error"]["count"])
total = int(s["all"]["anchor"]["replay_endpoint_factor_error"]["count"])
print(f"[teacher-response] latent-frame max abs={mx:.3e}")
print(f"[teacher-response] replay-good anchors={good}/{total}")
if mx > 2e-5:
    raise SystemExit("teacher/student latent frame mismatch")
if total <= 0:
    raise SystemExit("no anchors evaluated")
PY

if [[ "$MODE" == "smoke" ]]; then
  echo
  echo "=== TEACHER RESPONSE SMOKE PACKAGE VALIDATION ==="
  bash scripts/package_mh_ald_teacher_response_oracle.sh
fi

echo
echo "=== TEACHER RESPONSE JOB DONE ==="
echo "Results: $OUT_DIR"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")
echo "Submitted teacher-response oracle ($MODE): $JID"
echo "node: $NODE"
echo "run_root: $OUT_DIR"
echo "status: squeue -j $JID"
echo "stdout: tail -f $LOG_DIR/mh_teacher_response_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/mh_teacher_response_${RUN_TAG}_${JID}.err"
