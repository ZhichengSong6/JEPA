#!/usr/bin/env bash
set -euo pipefail
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
NODE="${NODE:-4090node3}"
case "$NODE" in
  4090node2|4090node3|3090node1|3090node2|3090node3) ;;
  *) echo "This script only permits 4090node2-3 / 3090node1-3; got $NODE" >&2; exit 2 ;;
esac
command -v sbatch >/dev/null || { echo "sbatch not found; run this on your Slurm submit host." >&2; exit 2; }
LOG_DIR="${LOG_DIR:-$STABLEWM_HOME/pcqr_tta_v1_runs/slurm}"
mkdir -p "$LOG_DIR"
# No --mem request, no package upgrades, no git reset and no old-result writes.
# A cluster-specific partition/account may be supplied as extra sbatch args.
sbatch --job-name=pcqr_tta_smoke --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --time=01:00:00 \
  --export=ALL --chdir="$REPO" --output="$LOG_DIR/%j.out" \
  "$@" "$REPO/pcqr_tta_v1/scripts/smoke.sh"
