#!/usr/bin/env bash
# File: jax_core/meta_adaptive_ctrl/train.sh

set -euo pipefail
SCRIPT="jax_core/meta_adaptive_ctrl/training_diag.py"

# ── parameter grid ──────────────────────────────────────────────────────────
seeds=(3)
Ms=(20)
pens=(5e-3)
# ────────────────────────────────────────────────────────────────────────────

logs_dir="logs"
mkdir -p "${logs_dir}"

for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  M="${Ms[$i]}"
  pen="${pens[$i]}"

  echo "▶︎  seed=${seed}  M=${M}  ctrl_pen=${pen}"
  python "${SCRIPT}" "${seed}" "${M}" --ctrl_pen "${pen}" "$@" \
     | tee "${logs_dir}/seed${seed}_M${M}_pen${pen}.log"
done
