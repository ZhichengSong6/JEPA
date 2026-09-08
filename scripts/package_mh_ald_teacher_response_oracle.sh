#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_ald_teacher_response_oracle_latest"

[[ -L "$LATEST" || -d "$LATEST" ]] || {
  echo "ERROR: latest teacher-response output not found: $LATEST" >&2
  exit 2
}

RUN_DIR="$(readlink -f "$LATEST")"
[[ -d "$RUN_DIR" ]] || {
  echo "ERROR: resolved run directory missing: $RUN_DIR" >&2
  exit 2
}

for f in response_cells.csv anchor_metrics.csv summary.json; do
  [[ -s "$RUN_DIR/$f" ]] || {
    echo "ERROR: missing/empty $RUN_DIR/$f" >&2
    exit 3
  }
done

TAG="$(basename "$RUN_DIR")"
BUNDLE="$RUN_DIR/mh_ald_teacher_response_oracle_${TAG}.tar.gz"

tar -C "$RUN_DIR" -czf "$BUNDLE" \
  response_cells.csv \
  anchor_metrics.csv \
  summary.json

echo "Bundle: $BUNDLE"
ls -lh "$BUNDLE"
