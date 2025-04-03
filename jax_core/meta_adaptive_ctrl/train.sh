# TODO description.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for seed in {6..9}
do
    for M in 20 30 40 50
    do
        echo "seed = $seed, M = $M"

        echo "Meta-ridge-regression:"
        # python train_lstsq.py $seed $M

        python jax_core/meta_adaptive_ctrl/training.py $seed $M 

    done
done
