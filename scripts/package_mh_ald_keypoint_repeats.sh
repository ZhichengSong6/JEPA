#!/usr/bin/env bash
# Package the latest MH-ALD keypoint repeat run.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_ald_keypoint_repeats_latest"
if [[ ! -e "$LATEST" ]]; then
  echo "ERROR: missing $LATEST" >&2
  exit 2
fi
RUN_ROOT="$(readlink -f "$LATEST")"
RUN_TAG="$(basename "$RUN_ROOT")"
BUNDLE="$RUN_ROOT/mh_ald_keypoint_repeats_$RUN_TAG.tar.gz"

REQ=(
  "$RUN_ROOT/summary/per_run.csv"
  "$RUN_ROOT/summary/aggregate.csv"
  "$RUN_ROOT/summary/paired_deltas.csv"
  "$RUN_ROOT/summary/paired_delta_summary.csv"
  "$RUN_ROOT/summary/summary.json"
)
for f in "${REQ[@]}"; do
  [[ -s "$f" ]] || { echo "MISSING/EMPTY: $f" >&2; exit 3; }
done

TMP="$RUN_ROOT/upload_bundle"
rm -rf "$TMP"
mkdir -p "$TMP"
cp -r "$RUN_ROOT/summary" "$TMP/"
cat "$RUN_ROOT"/raw/*.txt > "$TMP/raw_results.txt"

cd "$REPO"
{
  git rev-parse HEAD
  git branch --show-current
  git status --short
} > "$TMP/git_meta.txt"

rm -f "$BUNDLE"
tar -C "$TMP" -czf "$BUNDLE" .
echo "Bundle: $BUNDLE"
ls -lh "$BUNDLE"
