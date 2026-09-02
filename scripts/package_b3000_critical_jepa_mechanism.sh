#!/usr/bin/env bash
# Collect Step-2 critical JEPA mechanism outputs into one uncompressed directory.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
SRC="$REPO/outputs/b3000_critical_jepa_mechanism/formal"
DST="$REPO/outputs/b3000_critical_jepa_mechanism_upload"

REQ=(
  "$SRC/closed_loop_audit.json"
  "$SRC/critical_manifest.csv"
  "$SRC/population_mechanism_metrics.csv"
  "$SRC/mean_plan_causal_chain.csv"
  "$SRC/candidate_mechanism_metrics.npz"
  "$SRC/mechanism_summary.json"
)

cd "$REPO"

echo "==== CHECK STEP-2 OUTPUTS ===="
for f in "${REQ[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "MISSING/EMPTY: $f" >&2
    exit 2
  fi
  echo "OK: $f"
done

OUT=$(find logs -maxdepth 1 -type f -name 'b3000_mechanism_formal_*.out' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'b3000_mechanism_formal_*.err' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "Formal logs not found." >&2
  exit 3
fi

rm -rf "$DST"
mkdir -p "$DST"

cp "$SRC/closed_loop_audit.json" "$DST/"
cp "$SRC/critical_manifest.csv" "$DST/"
cp "$SRC/population_mechanism_metrics.csv" "$DST/"
cp "$SRC/mean_plan_causal_chain.csv" "$DST/"
cp "$SRC/candidate_mechanism_metrics.npz" "$DST/"
cp "$SRC/mechanism_summary.json" "$DST/"
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
STEP 2 critical-case JEPA mechanism diagnostic.

Frozen planner:
  CEM N=300, I=10, K=30, B=3000.
  Formal cases are exactly the non-both-success cases from Step 1.

Questions:
  1. Encoder ceiling:
     Does E(real future observation) preserve physical ordering?
  2. Endpoint fidelity:
     How close is predicted terminal latent to E(real terminal observation)?
  3. Causal prefix:
     How do C=1,2,3 affect ranking/tail fidelity and endpoint MSE?
  4. Mean-plan causal chain:
     Do the official policy's actually returned raw actions match the solve-0
     final CEM mean, and how much physical progress does the directly recorded
     closed-loop trajectory make before solve 1?

Files:
  closed_loop_audit.json
  critical_manifest.csv
  population_mechanism_metrics.csv
  mean_plan_causal_chain.csv
  candidate_mechanism_metrics.npz
  mechanism_summary.json
  formal.out
  formal.err
  git_meta.txt
  README.txt

Upload all files together.
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
