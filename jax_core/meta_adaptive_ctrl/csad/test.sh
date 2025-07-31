
# Load Conda and activate 'tensor' env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensor

# Verify correct Python is being used
echo "Python path in script: $(which python)"
python -c "import jax; print(' JAX version:', jax.__version__)"

# Add project to PYTHONPATH so jax_core can be found
export PYTHONPATH=$PYTHONPATH:/home/kmroen/projects/McHorcrux
for seed in {0..2}
do
    for M in 2 5 10 20
    do
        echo "seed = $seed, M = $M"

        echo "testing all:"
        python jax_core/meta_adaptive_ctrl/csad/test_all.py $seed $M 
    done
done