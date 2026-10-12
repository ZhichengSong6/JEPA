#!/usr/bin/env bash
# Analyze-only recovery; no new survey, CEM capture or training.
# bash pc_mh_failure_v1/submit_recovery.sh v1
# SOURCE_OUT overrides the failed output (default pcmh_failure_v1).
set -euo pipefail
TAG="${1:-v1}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then echo 'Invalid recovery tag' >&2; exit 2; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
SOURCE_OUT="${SOURCE_OUT:-$STABLEWM_HOME/pcmh_failure_v1}"
OUT="$STABLEWM_HOME/pcmh_failure_recovery_$TAG"
LOGS="$STABLEWM_HOME/pcmh_failure_recovery_logs"
CONDA_SH='/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh'
for FILE in protocol.json survey_complete.json trace_complete.json BLOCKED.json; do
    test -f "$SOURCE_OUT/$FILE" || { echo "Missing source $FILE; do not rerun collection blindly" >&2; exit 3; }
done
if [[ -e "$OUT" ]]; then echo "Preserve $OUT; use a new recovery tag" >&2; exit 4; fi
mkdir -p "$LOGS"
JOB="$(mktemp "$LOGS/recover_${TAG}_XXXXXX.sbatch")"
{
    echo '#!/usr/bin/env bash'
    echo '#SBATCH --job-name=pcmh_recover'
    echo '#SBATCH --partition=GPU'
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --ntasks=1'
    echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --cpus-per-task=6'
    echo '#SBATCH --time=12:00:00'
    echo '#SBATCH --exclude=3090node1'
    echo "#SBATCH --output=$LOGS/recover_${TAG}_%j.out"
    echo "#SBATCH --error=$LOGS/recover_${TAG}_%j.err"
    echo 'set -euo pipefail'
    printf 'source %q\n' "$CONDA_SH"
    echo 'conda activate lewm'
    printf 'cd %q\n' "$REPO"
    printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
    echo 'export TMPDIR="$STABLEWM_HOME/t${SLURM_JOB_ID}"'
    echo 'mkdir -p "$TMPDIR"'
    echo 'export PYTHONUNBUFFERED=1 MUJOCO_GL=egl OMP_NUM_THREADS=2'
    printf 'srun --kill-on-bad-exit=1 python -u %q --source %q --output %q\n' \
        "$HERE/recover_analysis.py" "$SOURCE_OUT" "$OUT"
} > "$JOB"
bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
printf 'Submitted analysis-only recovery: %s\nSource (read-only): %s\nOutput: %s\n' "$JID" "$SOURCE_OUT" "$OUT"
printf 'stdout: tail -n 80 %q\nstderr: tail -n 80 %q\n' \
    "$LOGS/recover_${TAG}_${JID%%;*}.out" "$LOGS/recover_${TAG}_${JID%%;*}.err"
