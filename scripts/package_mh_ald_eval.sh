#!/usr/bin/env bash
# Package one MH-ALD evaluation run for upload/analysis.
#
# Usage:
#   bash scripts/package_mh_ald_eval.sh
#   bash scripts/package_mh_ald_eval.sh /absolute/path/to/run_root
#
# With no argument, outputs/mh_ald_eval_latest is used.

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
cd "$REPO"

if [[ $# -ge 1 ]]; then
  RUN_ROOT="$1"
else
  LATEST="$REPO/outputs/mh_ald_eval_latest"
  if [[ ! -e "$LATEST" ]]; then
    echo "ERROR: $LATEST does not exist." >&2
    exit 2
  fi
  RUN_ROOT="$(readlink -f "$LATEST")"
fi

HORIZON_DIR="$RUN_ROOT/horizon"
CEM_DIR="$RUN_ROOT/cem"
SUM_DIR="$RUN_ROOT/summary"

REQ=(
  "$HORIZON_DIR/anchor_horizon_radius_metrics.csv"
  "$HORIZON_DIR/candidate_metrics.npz"
  "$HORIZON_DIR/summary.json"
  "$SUM_DIR/budget_summary.csv"
  "$SUM_DIR/budget_summary.json"
)

echo "==== CHECK RESULTS ===="
for p in "${REQ[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "MISSING: $p" >&2
    exit 3
  fi
  echo "OK: $p"
done

RUN_TAG="$(basename "$RUN_ROOT")"
BUNDLE_DIR="$RUN_ROOT/upload_bundle"
ARCHIVE="$RUN_ROOT/mh_ald_eval_${RUN_TAG}.tar.gz"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

cp "$HORIZON_DIR/anchor_horizon_radius_metrics.csv" "$BUNDLE_DIR/"
cp "$HORIZON_DIR/candidate_metrics.npz" "$BUNDLE_DIR/horizon_candidate_metrics.npz"
cp "$HORIZON_DIR/summary.json" "$BUNDLE_DIR/horizon_summary.json"
cp "$SUM_DIR/budget_summary.csv" "$BUNDLE_DIR/"
cp "$SUM_DIR/budget_summary.json" "$BUNDLE_DIR/"

{
  while IFS= read -r f; do
    echo
    echo "################################################################"
    echo "FILE: $f"
    echo "################################################################"
    cat "$f"
  done < <(find "$CEM_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
} > "$BUNDLE_DIR/cem_raw_results.txt"

{
  echo "===== GIT ====="
  git rev-parse HEAD
  git branch --show-current
  git status --short
  echo
  echo "===== RUN ====="
  echo "run_root=$RUN_ROOT"
  echo "run_tag=$RUN_TAG"
} > "$BUNDLE_DIR/git_meta.txt"

# Collect only this run's Slurm logs when available.
find "$REPO/logs" -maxdepth 1 -type f \
  \( -name "mh_ald_eval_${RUN_TAG}_*.out" -o -name "mh_ald_eval_${RUN_TAG}_*.err" \) \
  -exec cp {} "$BUNDLE_DIR/" \;

cat > "$BUNDLE_DIR/README.txt" <<EOF
MH-ALD evaluation bundle

Compared models:
  lewm
  ald
  mh_ald

Main evidence:
  1) horizon_summary.json / anchor_horizon_radius_metrics.csv
     H1-H5 directional and candidate-ranking quality.
  2) budget_summary.json / budget_summary.csv
     success-vs-CEM-budget with identical planner/cost.
  3) cem_raw_results.txt
     raw 100-case evaluation logs, including N=300,I=30 full-budget points.

Run:
  $RUN_ROOT
EOF

rm -f "$ARCHIVE"
tar -C "$BUNDLE_DIR" -czf "$ARCHIVE" .

echo
echo "==== BUNDLE ===="
echo "$ARCHIVE"
ls -lh "$ARCHIVE"
echo
echo "Upload this single tar.gz file for analysis."
