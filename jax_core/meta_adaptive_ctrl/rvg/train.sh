#!/bin/bash

# Load Conda and activate 'tensor' env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensor

# Verify correct Python is being used
echo "Python path in script: $(which python)"
python -c "import jax; print(' JAX version:', jax.__version__)"

# Add project to PYTHONPATH so jax_core can be found
export PYTHONPATH=$PYTHONPATH:/home/kmroen/projects/McHorcrux

# Training loop
for seed in {0..2}
do
    for M in 2 5 10 20
    do
        file_path="data/training_results/rvg/model_uncertainty/tanh/act_off/ctrl_pen_6/seed=${seed}_M=${M}.pkl"
        if [ -f "$file_path" ]; then
            echo "File $file_path exists. Skipping."
        else
            echo "seed = $seed, M = $M"
            echo "meta train:"
            python jax_core/meta_adaptive_ctrl/rvg/training_diag.py $seed $M 
        fi
    done
done
