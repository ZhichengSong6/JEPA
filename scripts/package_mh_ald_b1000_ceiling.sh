#!/usr/bin/env bash
# Package latest B=1000 ceiling experiment.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
LATEST="$REPO/outputs/mh_ald_b1000_ceiling_latest"
[[ -e "$LATEST" ]] || { echo "ERROR: missing $LATEST" >&2; exit 2; }
RUN_ROOT="$(readlink -f "$LATEST")"
RUN_TAG="$(basename "$RUN_ROOT")"
BUNDLE="$RUN_ROOT/mh_ald_b1000_ceiling_$RUN_TAG.tar.gz"

REQ=(
  "$RUN_ROOT/baseline.json"
  "$RUN_ROOT/encoder_oracle.json"
  "$RUN_ROOT/physical_oracle.json"
  "$RUN_ROOT/encoder_solver_diagnostics.csv"
  "$RUN_ROOT/physical_solver_diagnostics.csv"
  "$RUN_ROOT/ceiling_case_manifest.csv"
  "$RUN_ROOT/ceiling_summary.json"
)
for f in "${REQ[@]}"; do
  [[ -s "$f" ]] || { echo "MISSING/EMPTY: $f" >&2; exit 3; }
done

TMP="$RUN_ROOT/upload_bundle"
rm -rf "$TMP"
mkdir -p "$TMP"
cp "${REQ[@]}" "$TMP/"

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

OUT=$(find logs -maxdepth 1 -type f -name "b1000_ceiling_${RUN_TAG}_*.out" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name "b1000_ceiling_${RUN_TAG}_*.err" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${OUT:-}" ]] && cp "$OUT" "$TMP/formal.out"
[[ -n "${ERR:-}" ]] && cp "$ERR" "$TMP/formal.err"

rm -f "$BUNDLE"
tar -C "$TMP" -czf "$BUNDLE" .
echo "Bundle: $BUNDLE"
ls -lh "$BUNDLE"
