# TODO description.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for seed in {0..2}
do
    for M in 2 5 10 20
    do
        echo "seed = $seed, M = $M"

        # Train the model
        python jax_core/meta_adaptive_ctrl/training_diag.py $seed $M 

    done
done
