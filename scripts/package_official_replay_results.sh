#!/usr/bin/env bash
# Safely package corrected official-execution replay results.
# Run only after official_replay_<JOBID>.out reaches === DONE ===.
set -u

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
cd "$REPO" || exit 1

BUNDLE="${1:-official_replay_diagnostics_bundle.tar.gz}"
META="outputs/official_replay_bundle_meta"

PATHS=(
  "outputs/pusht_cem_population_trace_lewm_formal/cem_population_fidelity_official"
  "outputs/pusht_cem_population_trace_ald_formal/cem_population_fidelity_official"
  "outputs/pusht_cem_population_trace_lewm_formal/center_value_trajectory_official"
  "outputs/pusht_cem_population_trace_ald_formal/center_value_trajectory_official"
)

missing=0
for p in "${PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "OK: $p"
  else
    echo "MISSING: $p"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Corrected replay results are incomplete."
  exit 2
fi

OUT=$(find logs -maxdepth 1 -type f -name 'official_replay_*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'official_replay_*.err' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "ERROR: corrected replay logs not found."
  exit 3
fi

mkdir -p "$META"
git rev-parse HEAD > "$META/git_commit.txt" 2>/dev/null || true
git branch --show-current > "$META/git_branch.txt" 2>/dev/null || true
git status --short > "$META/git_status.txt" 2>/dev/null || true

rm -f "$BUNDLE"
if ! tar -czf "$BUNDLE" \
  "$META" \
  "${PATHS[@]}" \
  "$OUT" "$ERR"; then
  rm -f "$BUNDLE"
  exit 4
fi

echo
ls -lh "$BUNDLE"
echo "SUCCESS: $REPO/$BUNDLE"
