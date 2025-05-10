#!/bin/bash

# TODO description.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

seed=5
for M in 2 5 10 20
do
    echo "seed = $seed, M = $M"

    echo "testing all:"
    python jax_core/meta_adaptive_ctrl/rvg/test_all.py $seed $M 
done
