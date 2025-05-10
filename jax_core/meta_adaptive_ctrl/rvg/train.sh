#!/bin/bash

# TODO description.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)



for seed in {5..7}
do
    for M in 2 5 10 20
    do
        echo "seed = $seed, M = $M"
        echo "meta train:"
        python jax_core/meta_adaptive_ctrl/rvg/training_diag.py $seed $M --ctrl_pen 1e-7
    done
done