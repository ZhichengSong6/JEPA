#!/usr/bin/env bash
# Package latest focused B=1000 MH-ALD failure autopsy.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_ald_b1000_failure_autopsy_latest"
[[ -e "$LATEST" ]] || { echo "ERROR: missing $LATEST" >&2; exit 2; }
RUN_ROOT="$(readlink -f "$LATEST")"
RUN_TAG="$(basename "$RUN_ROOT")"
BUNDLE="$RUN_ROOT/mh_ald_b1000_failure_autopsy_$RUN_TAG.tar.gz"

REQ=(
  "$RUN_ROOT/population_autopsy.csv"
  "$RUN_ROOT/candidate_autopsy.npz"
  "$RUN_ROOT/case_summary.csv"
  "$RUN_ROOT/iteration_group_summary.csv"
  "$RUN_ROOT/failure_autopsy_summary.json"
)
for f in "${REQ[@]}"; do
  [[ -s "$f" ]] || { echo "MISSING/EMPTY: $f" >&2; exit 3; }
done

TMP="$RUN_ROOT/upload_bundle"
rm -rf "$TMP"
mkdir -p "$TMP"
cp "${REQ[@]}" "$TMP/"

CEILING="$REPO/outputs/mh_ald_b1000_ceiling_latest"
if [[ -s "$CEILING/ceiling_case_manifest.csv" ]]; then
  cp "$CEILING/ceiling_case_manifest.csv" "$TMP/"
fi
if [[ -s "$CEILING/ceiling_summary.json" ]]; then
  cp "$CEILING/ceiling_summary.json" "$TMP/"
fi

cd "$REPO"
{
  echo "===== GIT ====="
  git rev-parse HEAD
  git branch --show-current
  git status --short
  echo
  echo "===== RUN ====="
  echo "run_root=$RUN_ROOT"
} > "$TMP/git_meta.txt"

OUT=$(find logs -maxdepth 1 -type f -name "b1000_failure_autopsy_${RUN_TAG}_*.out" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name "b1000_failure_autopsy_${RUN_TAG}_*.err" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${OUT:-}" ]] && cp "$OUT" "$TMP/formal.out"
[[ -n "${ERR:-}" ]] && cp "$ERR" "$TMP/formal.err"

rm -f "$BUNDLE"
tar -C "$TMP" -czf "$BUNDLE" .
echo "Bundle: $BUNDLE"
ls -lh "$BUNDLE"
