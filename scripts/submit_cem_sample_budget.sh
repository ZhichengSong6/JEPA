#!/usr/bin/env bash
# Submit the paired CEM sample-budget sweep for LeWM vs ALD.
#
# Experiment:
#   CEM refinement steps: fixed at 5
#   population size N:    30, 100, 300, 1000
#   elite fraction:       10%  -> K = 3, 10, 30, 100
#   evaluation episodes:  100
#   environment budget:   50
#   seed:                 42
#
# Usage:
#   NODE=4090node3 bash scripts/submit_cem_sample_budget.sh
# Optional overwrite:
#   OVERWRITE=1 NODE=4090node3 bash scripts/submit_cem_sample_budget.sh
#
# Results:
#   <repo>/outputs/cem_sample_budget/

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"

NODE="${NODE:-4090node3}"
OVERWRITE="${OVERWRITE:-0}"

LEWM="lewm_epoch_10"
ALD="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"

CEM_STEPS=5
NUM_EVAL=100
EVAL_BUDGET=50
SEED=42
SAMPLES=(30 100 300 1000)

RESULT_DIR="$REPO/outputs/cem_sample_budget"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_cem_sample_budget"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$GEN_DIR"

SLURM_FILE="$GEN_DIR/cem_sample_budget.slurm"

cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=cem_samp
#SBATCH --output=$LOG_DIR/cem_sample_budget_%j.out
#SBATCH --error=$LOG_DIR/cem_sample_budget_%j.err

set -euo pipefail

source "$CONDA_SH"
conda activate lewm
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

LEWM="$LEWM"
ALD="$ALD"
RESULT_DIR="$RESULT_DIR"
CEM_STEPS=$CEM_STEPS
NUM_EVAL=$NUM_EVAL
EVAL_BUDGET=$EVAL_BUDGET
SEED=$SEED
OVERWRITE=$OVERWRITE

mkdir -p "\$RESULT_DIR"

run_one () {
    local label="\$1"
    local policy="\$2"
    local n="\$3"
    local k="\$4"
    local out="\$RESULT_DIR/cem_samplebudget_\${label}_n\${n}_i\${CEM_STEPS}_ep\${NUM_EVAL}.txt"

    if [[ -e "\$out" && "\$OVERWRITE" != "1" ]]; then
        echo "ERROR: result already exists: \$out" >&2
        echo "Set OVERWRITE=1 only if you intentionally want to replace it." >&2
        exit 3
    fi
    rm -f "\$out"

    echo
    echo "================================================================"
    echo "model=\$label N=\$n K=\$k I=\$CEM_STEPS episodes=\$NUM_EVAL"
    echo "output=\$out"
    echo "================================================================"

    python -u eval.py \
        seed=\$SEED \
        policy="\$policy" \
        solver.num_samples=\$n \
        solver.n_steps=\$CEM_STEPS \
        solver.topk=\$k \
        eval.num_eval=\$NUM_EVAL \
        eval.eval_budget=\$EVAL_BUDGET \
        output.filename="\$out"
}

for n in 30 100 300 1000; do
    k=\$((n / 10))
    run_one lewm "\$LEWM" "\$n" "\$k"
done

for n in 30 100 300 1000; do
    k=\$((n / 10))
    run_one ald "\$ALD" "\$n" "\$k"
done

echo
echo "All sample-budget runs completed."
echo "Results:"
ls -lh "\$RESULT_DIR"/cem_samplebudget_*_i\${CEM_STEPS}_ep\${NUM_EVAL}.txt
EOF

bash -n "$SLURM_FILE"
JID=$(sbatch --parsable "$SLURM_FILE")

echo "Submitted CEM sample-budget sweep: $JID"
echo "node       : $NODE"
echo "GPU        : 1"
echo "CEM steps  : $CEM_STEPS"
echo "samples    : ${SAMPLES[*]}"
echo "episodes   : $NUM_EVAL"
echo "results    : $RESULT_DIR"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true
echo
echo "Log:"
echo "  tail -f $LOG_DIR/cem_sample_budget_${JID}.out"
