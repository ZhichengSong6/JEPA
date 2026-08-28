#!/usr/bin/env bash
# Safely package Pure-ALD formal results.
# Run only after logs/pure_ald_formal_<JOBID>.out reaches === DONE ===.
set -u

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
cd "$REPO" || exit 1

BUNDLE="${1:-pure_ald_formal_bundle.tar.gz}"
META="$REPO/outputs/pure_ald_bundle_meta"
FORMAL_DIR="$STABLEWM_HOME/pusht_pure_ald_h5_seed3072_ep10_ddp4"
LOCAL_DIR="$REPO/outputs/pure_ald_h5_local_formal"
CEM_DIR="$REPO/outputs/pure_ald_cem_budget"

REQ=(
  "$FORMAL_DIR/config.yaml"
  "$LOCAL_DIR/anchor_horizon_radius_metrics.csv"
  "$LOCAL_DIR/candidate_metrics.npz"
  "$LOCAL_DIR/summary.json"
  "$CEM_DIR/pure_ald_n300_i1_ep100.txt"
  "$CEM_DIR/pure_ald_n300_i3_ep100.txt"
  "$CEM_DIR/pure_ald_n300_i5_ep100.txt"
  "$CEM_DIR/pure_ald_n300_i10_ep100.txt"
  "$CEM_DIR/pure_ald_n300_i30_ep100.txt"
)

echo "==== CHECKING PURE-ALD FORMAL RESULTS ===="
missing=0
for p in "${REQ[@]}"; do
  if [[ -f "$p" ]]; then
    echo "OK: $p"
  else
    echo "MISSING: $p"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Pure-ALD formal results are incomplete."
  echo "Do not package until the formal log reaches === DONE ===."
  exit 2
fi

OUT=$(find logs -maxdepth 1 -type f -name 'pure_ald_formal_*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
ERR=$(find logs -maxdepth 1 -type f -name 'pure_ald_formal_*.err' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)

if [[ -z "${OUT:-}" || -z "${ERR:-}" ]]; then
  echo "ERROR: Pure-ALD formal .out/.err logs not found."
  exit 3
fi

mkdir -p "$META"
git rev-parse HEAD > "$META/git_commit.txt" 2>/dev/null || true
git branch --show-current > "$META/git_branch.txt" 2>/dev/null || true
git status --short > "$META/git_status.txt" 2>/dev/null || true
git remote -v > "$META/git_remote.txt" 2>/dev/null || true

rm -f "$BUNDLE"

# Do not include the large checkpoint. Config + diagnostics + planning results
# + logs + git metadata are sufficient for analysis and reproducibility.
if ! tar -czf "$BUNDLE" \
  "$META" \
  "$FORMAL_DIR/config.yaml" \
  "$LOCAL_DIR" \
  "$CEM_DIR" \
  "$OUT" \
  "$ERR"; then
  echo "ERROR: tar failed."
  rm -f "$BUNDLE"
  exit 4
fi

echo
echo "==== BUNDLE CREATED ===="
ls -lh "$BUNDLE"
echo "Archive entries: $(tar -tzf "$BUNDLE" | wc -l)"
echo
echo "SUCCESS: $REPO/$BUNDLE"
