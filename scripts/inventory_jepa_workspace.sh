#!/usr/bin/env bash
# Produce a read-only inventory of LeWM_official and LeWM_data before cleanup.
# This script NEVER deletes or moves anything.
#
# Usage:
#   bash scripts/inventory_jepa_workspace.sh
#
# Reports:
#   LeWM_official/cleanup_inventory/

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
OUTDIR="$REPO/cleanup_inventory"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTDIR"

REPO_OUT="$OUTDIR/repo_inventory_${STAMP}.txt"
DATA_OUT="$OUTDIR/data_inventory_${STAMP}.txt"

{
    echo "=== REPO INVENTORY ==="
    echo "path: $REPO"
    echo "generated: $(date -Is)"
    echo
    echo "=== TOTAL SIZE ==="
    du -sh "$REPO" || true
    echo
    echo "=== TOP-LEVEL SIZE ==="
    du -h --max-depth=1 "$REPO" 2>/dev/null | sort -h || true
    echo
    echo "=== GIT STATUS ==="
    cd "$REPO"
    git status --short || true
    echo
    echo "=== TOP-LEVEL ENTRIES ==="
    find "$REPO" -mindepth 1 -maxdepth 1 -printf '%TY-%Tm-%Td %TH:%TM  %y  %p\n' | sort || true
    echo
    echo "=== OUTPUTS / LOGS / GENERATED SLURM ==="
    find "$REPO/outputs" "$REPO/logs" "$REPO/slurm" \
        -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort || true
    echo
    echo "=== CACHE DIRECTORIES ==="
    find "$REPO" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -print 2>/dev/null || true
} > "$REPO_OUT"

{
    echo "=== DATA INVENTORY ==="
    echo "path: $DATA"
    echo "generated: $(date -Is)"
    echo
    echo "=== TOTAL SIZE ==="
    du -sh "$DATA" || true
    echo
    echo "=== TOP-LEVEL SIZE ==="
    du -h --max-depth=1 "$DATA" 2>/dev/null | sort -h || true
    echo
    echo "=== TOP-LEVEL ENTRIES ==="
    find "$DATA" -mindepth 1 -maxdepth 1 -printf '%TY-%Tm-%Td %TH:%TM  %y  %p\n' | sort || true
    echo
    echo "=== CHECKPOINTS ==="
    find "$DATA" -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) \
        -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort || true
    echo
    echo "=== EVAL RESULT FILES ==="
    find "$DATA" -type f \( -name '*.txt' -o -name '*.csv' -o -name '*.json' -o -name '*.npz' \) \
        -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort || true
    echo
    echo "=== VIDEO FILES ==="
    find "$DATA" -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.gif' \) \
        -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort || true
    echo
    echo "=== LARGE FILES >= 100 MB ==="
    find "$DATA" -type f -size +100M -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort || true
} > "$DATA_OUT"

echo "Inventory complete. Nothing was deleted."
echo "Repo report:"
echo "  $REPO_OUT"
echo "Data report:"
echo "  $DATA_OUT"
