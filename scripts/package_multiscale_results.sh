#!/usr/bin/env bash
# Package formal multi-scale diagnostic results safely.
#
# Run:
#   bash scripts/package_multiscale_results.sh
#
# If something is missing, only this script exits; the SSH session stays alive.
set -u

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
cd "$REPO" || {
  echo "ERROR: cannot cd to $REPO"
  exit 1
}

BUNDLE="${1:-multiscale_diagnostics_formal_bundle.tar.gz}"
META="outputs/multiscale_diag_bundle_meta"

REQ_PATHS=(
  "outputs/pusht_cem_population_trace_lewm_formal"
  "outputs/pusht_cem_population_trace_ald_formal"
  "outputs/pusht_cem_population_trace_lewm_formal/cem_population_fidelity"
  "outputs/pusht_cem_population_trace_ald_formal/cem_population_fidelity"
  "outputs/pusht_cem_population_trace_lewm_formal/center_value_trajectory"
  "outputs/pusht_cem_population_trace_ald_formal/center_value_trajectory"
  "outputs/pusht_cem_trace_lewm_formal/null_response_decomposition_formal"
  "outputs/pusht_cem_trace_ald_formal/null_response_decomposition_formal"
)

CHECK_FILES=(
  "outputs/pusht_cem_population_trace_lewm_formal/cem_population_fidelity/summary.csv"
  "outputs/pusht_cem_population_trace_ald_formal/cem_population_fidelity/summary.csv"
  "outputs/pusht_cem_population_trace_lewm_formal/center_value_trajectory/center_value_metrics.csv"
  "outputs/pusht_cem_population_trace_lewm_formal/center_value_trajectory/solve_summary.csv"
  "outputs/pusht_cem_population_trace_ald_formal/center_value_trajectory/center_value_metrics.csv"
  "outputs/pusht_cem_population_trace_ald_formal/center_value_trajectory/solve_summary.csv"
  "outputs/pusht_cem_trace_lewm_formal/null_response_decomposition_formal/direction_metrics.csv"
  "outputs/pusht_cem_trace_ald_formal/null_response_decomposition_formal/direction_metrics.csv"
)

echo "==== CHECKING FORMAL RESULT DIRECTORIES ===="
missing=0
for p in "${REQ_PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "OK: $p"
  else
    echo "MISSING: $p"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Formal results are incomplete or have not been run yet."
  echo "Run:"
  echo "  NODE=4090node3 bash scripts/submit_multiscale_diagnostics.sh formal"
  echo
  echo "After that job reaches '=== DONE ===', run this packaging script again."
  exit 2
fi

echo
echo "==== CHECKING KEY RESULT FILES ===="
missing=0
for p in "${CHECK_FILES[@]}"; do
  if [[ -f "$p" ]]; then
    echo "OK: $p"
  else
    echo "MISSING: $p"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Some formal diagnostic outputs are missing. Check the formal .out/.err logs."
  exit 3
fi

FORMAL_OUT=$(find logs -maxdepth 1 -type f -name 'multiscale_diag_formal_*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
FORMAL_ERR=$(find logs -maxdepth 1 -type f -name 'multiscale_diag_formal_*.err' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)

if [[ -z "${FORMAL_OUT:-}" || -z "${FORMAL_ERR:-}" ]]; then
  echo "ERROR: formal multiscale log files were not found."
  exit 4
fi

mkdir -p "$META"
git rev-parse HEAD > "$META/git_commit.txt" 2>/dev/null || true
git branch --show-current > "$META/git_branch.txt" 2>/dev/null || true
git status --short > "$META/git_status.txt" 2>/dev/null || true
git remote -v > "$META/git_remote.txt" 2>/dev/null || true

echo
echo "Using logs:"
echo "  OUT=$FORMAL_OUT"
echo "  ERR=$FORMAL_ERR"

rm -f "$BUNDLE"

echo
echo "==== CREATING BUNDLE ===="
if ! tar -czf "$BUNDLE" \
  "$META" \
  outputs/pusht_cem_population_trace_lewm_formal \
  outputs/pusht_cem_population_trace_ald_formal \
  outputs/pusht_cem_trace_lewm_formal/null_response_decomposition_formal \
  outputs/pusht_cem_trace_ald_formal/null_response_decomposition_formal \
  "$FORMAL_OUT" \
  "$FORMAL_ERR"; then
  echo "ERROR: tar failed. No result bundle should be trusted."
  rm -f "$BUNDLE"
  exit 5
fi

echo
echo "==== VERIFYING BUNDLE ===="
key_count=$(tar -tzf "$BUNDLE" | grep -Ec   'cem_population_fidelity/summary.csv|center_value_trajectory/solve_summary.csv|null_response_decomposition_formal/direction_metrics.csv|multiscale_diag_formal_.*\.(out|err)$' || true)

if [[ "$key_count" -lt 8 ]]; then
  echo "ERROR: bundle verification failed; expected at least 8 key files, found $key_count."
  rm -f "$BUNDLE"
  exit 6
fi

echo "Key files found: $key_count"
echo "Total archive entries: $(tar -tzf "$BUNDLE" | wc -l)"
ls -lh "$BUNDLE"
echo
echo "SUCCESS: $REPO/$BUNDLE"
