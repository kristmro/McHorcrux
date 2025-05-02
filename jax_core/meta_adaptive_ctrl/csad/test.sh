#!/usr/bin/env bash
# File: jax_core/meta_adaptive_ctrl/test.sh

set -euo pipefail
SCRIPT="jax_core/meta_adaptive_ctrl/csad/test_all.py"
#1 1 2 1 0
#5 10 2 20 20


# ── parameter grid ──────────────────────────────────────────────────────────
seeds=(10 10)
Ms=(10 20)
# ────────────────────────────────────────────────────────────────────────────


for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  M="${Ms[$i]}"

  echo "▶︎  seed=${seed}  M=${M}"
  python "${SCRIPT}" "${seed}" "${M}" --use_cpu --use_x64 
done