#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
SRC="$REPO/outputs/b3000_hardstate_encoder_geometry/formal"
DST="$REPO/outputs/b3000_hardstate_encoder_geometry_upload"

REQ=(
  "$SRC/selected_cases.csv"
  "$SRC/population_component_metrics.csv"
  "$SRC/case_component_summary.csv"
  "$SRC/candidate_components.npz"
  "$SRC/summary.json"
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

OUT=$(find logs -maxdepth 1 -type f -name 'b3000_encoder_geometry_formal_*.out' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'b3000_encoder_geometry_formal_*.err' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)

rm -rf "$DST"
mkdir -p "$DST"

cp "$SRC/selected_cases.csv" "$DST/"
cp "$SRC/population_component_metrics.csv" "$DST/"
cp "$SRC/case_component_summary.csv" "$DST/"
cp "$SRC/candidate_components.npz" "$DST/"
cp "$SRC/summary.json" "$DST/"
[[ -n "${OUT:-}" ]] && cp "$OUT" "$DST/formal.out"
[[ -n "${ERR:-}" ]] && cp "$ERR" "$DST/formal.err"

{
  echo "===== GIT ====="
  git rev-parse HEAD
  git branch --show-current
  git status --short
} > "$DST/git_meta.txt"

cat > "$DST/README.txt" <<EOF
B=3000 hard-state encoder geometry autopsy.

Primary question:
  Why is real-future encoder Euclidean goal geometry weak in eval 27/53?

Cases:
  27,53  both-fail
  23     regression control
  3 hardest rescue controls chosen from prior ALD solve-1 physical cost

Source:
  ALD+TF closed-loop trajectory, solve1, CEM iterations 0/3/9.

Physical decomposition:
  pusher XY
  block XY
  theta
  object_task = block + theta (diagnostic)
  official = pusher + block + theta

Key tests:
  Spearman / top10 / oracle-rank / selected-percentile per component
  partial Spearman controlling other factors
  matched-pair tests:
    object matched -> pusher preference
    pusher matched -> object preference
    block matched -> theta preference
    theta matched -> block preference

Upload every file in this directory.
EOF

echo
echo "==== FILES TO UPLOAD ===="
find "$DST" -maxdepth 1 -type f -printf '%f\n' | sort
echo
echo "Upload directory:"
echo "$DST"
