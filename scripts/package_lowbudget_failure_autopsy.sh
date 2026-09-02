#!/usr/bin/env bash
# Collect all low-budget failure-autopsy outputs into one UNCOMPRESSED upload dir.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
SRC="$REPO/outputs/lowbudget_failure_autopsy/formal"
DST="$REPO/outputs/lowbudget_failure_autopsy_upload"

REQ=(
  "$SRC/closed_loop_eval.json"
  "$SRC/case_manifest.csv"
  "$SRC/population_metrics.csv"
  "$SRC/episode_summary.csv"
  "$SRC/autopsy_summary.json"
)

cd "$REPO"

echo "==== CHECK OUTPUTS ===="
for f in "${REQ[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "MISSING/EMPTY: $f" >&2
    exit 2
  fi
  echo "OK: $f"
done

OUT=$(find logs -maxdepth 1 -type f -name 'failure_autopsy_formal_*.out' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'failure_autopsy_formal_*.err' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "Formal logs not found." >&2
  exit 3
fi

rm -rf "$DST"
mkdir -p "$DST"

cp "$SRC/closed_loop_eval.json" "$DST/"
cp "$SRC/case_manifest.csv" "$DST/"
cp "$SRC/population_metrics.csv" "$DST/"
cp "$SRC/episode_summary.csv" "$DST/"
cp "$SRC/autopsy_summary.json" "$DST/"
cp "$OUT" "$DST/formal.out"
cp "$ERR" "$DST/formal.err"

{
  echo "===== GIT ====="
  git rev-parse HEAD
  git branch --show-current
  git status --short
  echo
  echo "===== REMOTES ====="
  git remote -v
} > "$DST/git_meta.txt"

cat > "$DST/README.txt" <<EOF
Low-budget CEM failure autopsy.

Target:
  ALD+TF
  N=30, I=10, K=3
  official full PushT
  100 closed-loop episodes

Diagnosis:
  all failures + up to 12 difficulty-matched successful controls
  physical replay snapshots at CEM iterations 0,1,3,5,9

Primary fields in population_metrics.csv:
  oracle_has_success_candidate
  rho_pred_phys
  elite_overlap_pred_phys
  pred_selected_phys_percentile
  selection_regret
  cem_update_cos_pred_phys
  center_after_phys_cost
  candidate_raw_oob_fraction

Upload every file in this directory together.
EOF

echo
echo "==== FILES TO UPLOAD ===="
find "$DST" -maxdepth 1 -type f -printf '%f\n' | sort
COUNT=$(find "$DST" -maxdepth 1 -type f | wc -l)
echo "count=$COUNT"
if (( COUNT > 20 )); then
  echo "ERROR: more than 20 upload files." >&2
  exit 4
fi
echo
echo "Upload directory:"
echo "$DST"
