for seed in {0..2}
do
    for M in 2 5 10 20
    do
        echo "seed = $seed, M = $M"
        python jax_core/meta_adaptive_ctrl/test_all.py $seed $M --use_x64 --use_cpu
    done
done
