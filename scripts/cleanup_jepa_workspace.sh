#!/usr/bin/env bash
# Destructive cleanup for the current JEPA/LeWM workspace.
# Keeps source code, the authoritative formal diagnostic needed for later
# hard-state evaluation, PushT HDF5 datasets, and trained models.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"

KEEP_OFFICIAL="formal_20260903T024414Z_3108082"
KEEP_A="A_20260903T083640Z_3219580"
KEEP_B="B_20260903T085721Z_3234166"

[[ -d "$REPO/.git" ]] || { echo "Not a git repo: $REPO" >&2; exit 2; }
[[ -d "$DATA" ]] || { echo "Missing data root: $DATA" >&2; exit 2; }

echo "=== Cleaning LeWM_official generated results ==="

if [[ -d "$REPO/outputs" ]]; then
  find "$REPO/outputs" -mindepth 1 -maxdepth 1 -type d     ! -name "pusht_official_diagnostic"     ! -name "pusht_residual_diagnostics"     -print -exec rm -rf {} +

  if [[ -d "$REPO/outputs/pusht_official_diagnostic" ]]; then
    find "$REPO/outputs/pusht_official_diagnostic"       -mindepth 1 -maxdepth 1 -type d       ! -name "$KEEP_OFFICIAL"       -print -exec rm -rf {} +
    find "$REPO/outputs/pusht_official_diagnostic"       -maxdepth 1 -type f -print -delete
  fi

  if [[ -d "$REPO/outputs/pusht_residual_diagnostics" ]]; then
    find "$REPO/outputs/pusht_residual_diagnostics"       -mindepth 1 -maxdepth 1 -type d       ! -name "$KEEP_A" ! -name "$KEEP_B"       -print -exec rm -rf {} +
  fi
fi

rm -rf "$REPO/logs"/*
rm -rf "$REPO/wandb"/*
rm -rf "$REPO/artifacts"/*
find "$REPO/slurm" -mindepth 1 -maxdepth 1 -type d   -name 'generated_*' -print -exec rm -rf {} + 2>/dev/null || true

find "$REPO" -maxdepth 1 -type f \(   -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) -print -delete
rm -rf "$REPO/analysis_inbox"
rm -rf "$REPO/docs/results"

echo
echo "=== Cleaning LeWM_data while retaining datasets/models ==="

find "$DATA" -maxdepth 1 -type f \(   -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) -print -delete

for d in "$DATA"/*; do
  [[ -d "$d" ]] || continue

  if find "$d" -maxdepth 3 -type f \( -name '*.h5' -o -name '*.hdf5' \)       -print -quit | grep -q .; then
    echo "KEEP dataset dir: $d"
    continue
  fi

  if find "$d" -maxdepth 3 -type f -name '*_object.ckpt' -print -quit | grep -q .; then
    echo "KEEP model dir: $d"

    mapfile -t objs < <(
      find "$d" -maxdepth 1 -type f -name '*_epoch_*_object.ckpt' | sort -V
    )
    if (( ${#objs[@]} > 0 )); then
      last="${objs[${#objs[@]}-1]}"
      for f in "${objs[@]}"; do
        b="$(basename "$f")"
        if [[ "$f" == "$last" || "$b" == *_epoch_10_object.ckpt ]]; then
          echo "  KEEP checkpoint: $b"
        else
          echo "  DELETE checkpoint: $b"
          rm -f "$f"
        fi
      done
      find "$d" -maxdepth 1 -type f -name '*_weights.ckpt' -print -delete
    fi

    rm -rf "$d/lightning_logs" "$d/wandb"
    continue
  fi

  echo "DELETE non-model/non-dataset dir: $d"
  rm -rf "$d"
done

echo
echo "=== Remaining project sizes ==="
du -sh "$REPO" "$DATA"

echo
echo "Kept authoritative diagnostic:"
echo "  $REPO/outputs/pusht_official_diagnostic/$KEEP_OFFICIAL"
echo "Kept residual diagnostics:"
echo "  $REPO/outputs/pusht_residual_diagnostics/$KEEP_A"
echo "  $REPO/outputs/pusht_residual_diagnostics/$KEEP_B"
echo
echo "Cleanup complete."
