#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_teacher_exp1_eval_latest"

[[ -L "$LATEST" || -d "$LATEST" ]] || {
  echo "ERROR: latest Exp1 eval not found: $LATEST" >&2
  exit 2
}
RUN_DIR="$(readlink -f "$LATEST")"
TAG="$(basename "$RUN_DIR")"
BUNDLE="$RUN_DIR/mh_teacher_exp1_eval_${TAG}.tar.gz"

for f in \
  "$RUN_DIR/summary/paired_seed_results.csv" \
  "$RUN_DIR/summary/paired_summary.json" \
  "$RUN_DIR/teacher_response/response_cells.csv" \
  "$RUN_DIR/teacher_response/anchor_metrics.csv" \
  "$RUN_DIR/teacher_response/summary.json"
do
  [[ -s "$f" ]] || { echo "ERROR: missing/empty $f" >&2; exit 3; }
done

tar -C "$RUN_DIR" -czf "$BUNDLE" \
  summary/paired_seed_results.csv \
  summary/paired_summary.json \
  teacher_response/response_cells.csv \
  teacher_response/anchor_metrics.csv \
  teacher_response/summary.json

echo "Bundle: $BUNDLE"
ls -lh "$BUNDLE"
