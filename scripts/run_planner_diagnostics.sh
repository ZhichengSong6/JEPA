#!/usr/bin/env bash
set -euo pipefail

# Three diagnostics discussed in the project:
#   D1. trace CEM refinement centers
#   D2. re-bin order transition by actual latent goal distance
#   D3. physical oracle landscape/Jacobian at planner-visited centers
#
# Override these from the shell if your checkpoint names differ.
LEWM_POLICY="${LEWM_POLICY:-lewm_epoch_10}"
ALD_POLICY="${ALD_POLICY:-pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10}"
TRACE_DIR="${TRACE_DIR:-outputs/pusht_cem_trace_lewm}"
GOAL_PROX_DIR="${GOAL_PROX_DIR:-${STABLEWM_HOME}/fixed_horizon_goal_proximity}"
NUM_EVAL="${NUM_EVAL:-10}"
MAX_SOLVES="${MAX_SOLVES:-20}"

echo "=== D1: trace official CEM refinement ==="
rm -rf "${TRACE_DIR}"
mkdir -p "${TRACE_DIR}"
JEPA_CEM_TRACE_DIR="${TRACE_DIR}" \
python -u eval.py \
  --config-name=pusht.yaml \
  solver=traced_cem \
  policy="${LEWM_POLICY}" \
  eval.num_eval="${NUM_EVAL}"

echo
echo "=== D2: latent-distance-binned order transition ==="
if [[ ! -f "${GOAL_PROX_DIR}/anchor_goal_metrics.csv" ]]; then
  echo "Missing ${GOAL_PROX_DIR}/anchor_goal_metrics.csv"
  echo "Run eval_pusht_fixed_horizon_goal_proximity.py first, or set GOAL_PROX_DIR."
  exit 2
fi
python -u analyze_pusht_order_transition_bins.py \
  --input "${GOAL_PROX_DIR}/anchor_goal_metrics.csv" \
  --num-bins 5

echo
echo "=== D3: oracle local landscape at CEM centers ==="
python -u eval_pusht_planner_center_landscape.py \
  --trace-dir "${TRACE_DIR}" \
  --policies "${LEWM_POLICY}" "${ALD_POLICY}" \
  --labels lewm ald \
  --iterations 0 1 3 5 10 30 \
  --max-solves "${MAX_SOLVES}" \
  --radius 0.1565 \
  --num-directions 64

echo
echo "Done."
echo "D1 traces: ${TRACE_DIR}/solve_*.npz"
echo "D2: ${GOAL_PROX_DIR}/distance_binned_order_transition/"
echo "D3: ${TRACE_DIR}/planner_center_landscape/"
