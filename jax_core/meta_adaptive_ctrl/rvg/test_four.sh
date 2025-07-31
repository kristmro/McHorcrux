#!/bin/bash

# Load Conda and activate 'tensor' env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensor

# Verify correct Python is being used
echo "Python path in script: $(which python)"
python -c "import jax; print(' JAX version:', jax.__version__)"

# Add project to PYTHONPATH so jax_core can be found
export PYTHONPATH=$PYTHONPATH:/home/kmroen/projects/McHorcrux

# Define base scripts
scripts=(
    "jax_core/meta_adaptive_ctrl/rvg/test_four_corner_full_scale.py"
)

# Define (seed, M) pairs
declare -a seed_M_pairs=(
    "2 5"
)

# Define (hs, tp) pairs
declare -a wave_conditions=(
    "2 8"
    "3 10"
    "4 14"
    "5 16"
)

# Loop through scripts, seed/M pairs, and wave conditions
for script in "${scripts[@]}"; do
  for pair in "${seed_M_pairs[@]}"; do
    for wave in "${wave_conditions[@]}"; do
      read -r seed M <<< "$pair"
      read -r hs tp <<< "$wave"
      echo "Running: python $script $seed $M $hs $tp"
      python "$script" "$seed" "$M" "$hs" "$tp"
    done
  done
done
