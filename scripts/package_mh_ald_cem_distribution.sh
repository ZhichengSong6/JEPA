#!/usr/bin/env bash
# Package the latest MH-ALD actual-CEM-distribution diagnostic.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_ald_cem_distribution_latest"
if [[ ! -e "$LATEST" ]]; then
  echo "ERROR: missing $LATEST" >&2
  exit 2
fi
RUN_ROOT="$(readlink -f "$LATEST")"
RUN_TAG="$(basename "$RUN_ROOT")"
BUNDLE="$RUN_ROOT/mh_ald_cem_distribution_$RUN_TAG.tar.gz"

REQ=(
  "$RUN_ROOT/cross_on_ald_cem/population_metrics.csv"
  "$RUN_ROOT/cross_on_ald_cem/summary.csv"
  "$RUN_ROOT/cross_on_ald_cem/summary.json"
  "$RUN_ROOT/cross_on_mh_ald_cem/population_metrics.csv"
  "$RUN_ROOT/cross_on_mh_ald_cem/summary.csv"
  "$RUN_ROOT/cross_on_mh_ald_cem/summary.json"
  "$RUN_ROOT/comparison/distribution_comparison.csv"
  "$RUN_ROOT/comparison/distribution_comparison.json"
)
for f in "${REQ[@]}"; do
  [[ -s "$f" ]] || { echo "MISSING/EMPTY: $f" >&2; exit 3; }
done

TMP="$RUN_ROOT/upload_bundle"
rm -rf "$TMP"
mkdir -p "$TMP/ald_cem" "$TMP/mh_cem" "$TMP/comparison"
cp "$RUN_ROOT/cross_on_ald_cem/population_metrics.csv" "$TMP/ald_cem/"
cp "$RUN_ROOT/cross_on_ald_cem/summary.csv" "$TMP/ald_cem/"
cp "$RUN_ROOT/cross_on_ald_cem/summary.json" "$TMP/ald_cem/"
cp "$RUN_ROOT/cross_on_mh_ald_cem/population_metrics.csv" "$TMP/mh_cem/"
cp "$RUN_ROOT/cross_on_mh_ald_cem/summary.csv" "$TMP/mh_cem/"
cp "$RUN_ROOT/cross_on_mh_ald_cem/summary.json" "$TMP/mh_cem/"
cp "$RUN_ROOT/comparison/"* "$TMP/comparison/"
cp "$RUN_ROOT/ald_trace_closed_loop.txt" "$TMP/" || true
cp "$RUN_ROOT/mh_trace_closed_loop.txt" "$TMP/" || true

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
