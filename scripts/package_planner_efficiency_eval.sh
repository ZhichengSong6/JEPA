#!/usr/bin/env bash
# Create an UNCOMPRESSED upload directory (<=20 files) for analysis.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
cd "$REPO"

CTX_DIR="$REPO/outputs/planner_efficiency_context"
CEM_DIR="$REPO/outputs/planner_efficiency_cem"
SUM_DIR="$REPO/outputs/planner_efficiency_summary"
UPLOAD="$REPO/outputs/planner_efficiency_upload"

REQ=(
  "$CTX_DIR/context_prefix_metrics.csv"
  "$CTX_DIR/candidate_metrics.npz"
  "$CTX_DIR/summary.json"
  "$SUM_DIR/budget_summary.csv"
  "$SUM_DIR/budget_summary.json"
)

echo "==== CHECK RESULTS ===="
for p in "${REQ[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "MISSING: $p" >&2
    exit 2
  fi
  echo "OK: $p"
done

OUT=$(find logs -maxdepth 1 -type f -name 'planner_efficiency_formal_*.out' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'planner_efficiency_formal_*.err' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "Formal logs not found." >&2
  exit 3
fi

rm -rf "$UPLOAD"
mkdir -p "$UPLOAD"

cp "$CTX_DIR/context_prefix_metrics.csv" "$UPLOAD/"
cp "$CTX_DIR/candidate_metrics.npz" "$UPLOAD/context_candidate_metrics.npz"
cp "$CTX_DIR/summary.json" "$UPLOAD/context_summary.json"
cp "$SUM_DIR/budget_summary.csv" "$UPLOAD/"
cp "$SUM_DIR/budget_summary.json" "$UPLOAD/"
cp "$OUT" "$UPLOAD/formal.out"
cp "$ERR" "$UPLOAD/formal.err"

{
  echo "===== GIT ====="
  git rev-parse HEAD
  git branch --show-current
  git status --short
  echo
  echo "===== REMOTES ====="
  git remote -v
} > "$UPLOAD/git_meta.txt"

{
  while IFS= read -r f; do
    echo
    echo "################################################################"
    echo "FILE: $f"
    echo "################################################################"
    cat "$f"
  done < <(find "$CEM_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
} > "$UPLOAD/cem_raw_results.txt"

cat > "$UPLOAD/README.txt" <<EOF
Planner-efficiency evaluation bundle.

Primary target:
  LeWM reference = N=300, I=10, B=N*I=3000.
  >=10x reduction requires B<=300 while matching/exceeding reference success.

Files:
  context_summary.json
  context_prefix_metrics.csv
  context_candidate_metrics.npz
  budget_summary.csv
  budget_summary.json
  cem_raw_results.txt
  formal.out
  formal.err
  git_meta.txt
  README.txt
EOF

echo
echo "==== UPLOAD DIRECTORY ===="
find "$UPLOAD" -maxdepth 1 -type f -printf '%f\n' | sort
COUNT=$(find "$UPLOAD" -maxdepth 1 -type f | wc -l)
echo "count=$COUNT"
if (( COUNT > 20 )); then
  echo "ERROR: upload directory exceeds 20 files." >&2
  exit 4
fi
echo
echo "Upload ALL files in this directory in one batch:"
echo "$UPLOAD"
