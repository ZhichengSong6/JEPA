#!/usr/bin/env bash
# Submit Reachability-Raw smoke -> formal training on one RTX 4090.
# The formal run matches the original LeWM training seed/batch/10-epoch setup,
# while loading six coarse frames only for the auxiliary raw-geometry loss.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"
NODE="${NODE:-4090node3}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi

SEED=3072
SMOKE_DIR="pusht_reachraw_h5_seed3072_smoke"
FORMAL_DIR="pusht_reachraw_h5_seed3072_ep10"
SMOKE_MODEL="lewm_reachraw_h5_smoke"
FORMAL_MODEL="lewm_reachraw_h5"

GEN_DIR="$REPO/slurm/generated_reachability_raw"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile train.py

rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
if [[ -e "$STABLEWM_HOME/$FORMAL_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: formal output already exists: $STABLEWM_HOME/$FORMAL_DIR" >&2
  echo "Delete it first if you intentionally want to retrain." >&2
  exit 2
fi

COMMON_SETUP=$(cat <<EOF
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
echo "node=\$(hostname)"
echo "python=\$(which python)"
python --version
EOF
)

cat > "$GEN_DIR/00_smoke.slurm" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=reachraw_sm
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/reachraw_smoke_%j.out
#SBATCH --error=$LOG_DIR/reachraw_smoke_%j.err

$COMMON_SETUP

python -u train.py \
  data=pusht_reachability \
  seed=$SEED \
  reachability.enabled=true \
  factor.enabled=false \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR
EOF

cat > "$GEN_DIR/10_formal.slurm" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=reachraw10
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/reachraw_formal_%j.out
#SBATCH --error=$LOG_DIR/reachraw_formal_%j.err

$COMMON_SETUP

python -u train.py \
  data=pusht_reachability \
  seed=$SEED \
  reachability.enabled=true \
  factor.enabled=false \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR

# Keep the trained epoch-10 model, not ten redundant full model objects.
find "$STABLEWM_HOME/$FORMAL_DIR" -maxdepth 1 -type f \
  -name "${FORMAL_MODEL}_epoch_*_object.ckpt" \
  ! -name "${FORMAL_MODEL}_epoch_10_object.ckpt" \
  -print -delete
rm -f "$STABLEWM_HOME/$FORMAL_DIR/${FORMAL_MODEL}_weights.ckpt"
rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
EOF

bash -n "$GEN_DIR/00_smoke.slurm"
bash -n "$GEN_DIR/10_formal.slurm"

JID_SMOKE="$(sbatch --parsable "$GEN_DIR/00_smoke.slurm")"
JID_FORMAL="$(sbatch --parsable --dependency="afterok:$JID_SMOKE" "$GEN_DIR/10_formal.slurm")"

cat <<EOF
Submitted Reachability-Raw on $NODE.

smoke : $JID_SMOKE
formal: $JID_FORMAL (afterok:$JID_SMOKE)

Formal output:
  $STABLEWM_HOME/$FORMAL_DIR/${FORMAL_MODEL}_epoch_10_object.ckpt

Queue:
  squeue -u zsong469 -o '%.18i %.12j %.2t %.10M %.24R'

Smoke log:
  tail -f $LOG_DIR/reachraw_smoke_${JID_SMOKE}.out

Formal log:
  tail -f $LOG_DIR/reachraw_formal_${JID_FORMAL}.out
EOF
