#!/usr/bin/env bash
# Collect paired B=3000 failure-analysis outputs into one uncompressed upload dir.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
SRC="$REPO/outputs/b3000_paired_failure_analysis/formal"
DST="$REPO/outputs/b3000_paired_failure_analysis_upload"

REQ=(
  "$SRC/closed_loop_results.json"
  "$SRC/paired_manifest.csv"
  "$SRC/cross_population_metrics.csv"
  "$SRC/case_summary.csv"
  "$SRC/cross_candidate_metrics.npz"
  "$SRC/paired_summary.json"
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

OUT=$(find logs -maxdepth 1 -type f -name 'b3000_paired_formal_*.out' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'b3000_paired_formal_*.err' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "Formal logs not found." >&2
  exit 3
fi

rm -rf "$DST"
mkdir -p "$DST"

cp "$SRC/closed_loop_results.json" "$DST/"
cp "$SRC/paired_manifest.csv" "$DST/"
cp "$SRC/cross_population_metrics.csv" "$DST/"
cp "$SRC/case_summary.csv" "$DST/"
cp "$SRC/cross_candidate_metrics.npz" "$DST/"
cp "$SRC/paired_summary.json" "$DST/"
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
STEP 1 paired B=3000 LeWM vs ALD+TF failure/rescue analysis.

Frozen planner:
  CEM N=300, I=10, K=30, B=3000
  100 identical official PushT starts

Primary paired groups:
  both_success
  lewm_fail_ald_success
  both_fail
  lewm_success_ald_fail

Cross-evaluation:
  For selected critical cases, both models score exactly the same candidate
  population from both LeWM-CEM and ALD+TF-CEM trajectories.
  Physical simulator replay is diagnosis-only.

Key files:
  paired_manifest.csv           exact episode outcome partition
  cross_population_metrics.csv  per-population cross-model metrics
  case_summary.csv              group/source aggregates
  cross_candidate_metrics.npz   candidate-level physical/LeWM/ALD costs
  paired_summary.json           high-level summary
  closed_loop_results.json      paired closed-loop results

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
