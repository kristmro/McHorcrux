# McHorcrux

This repository hosts **three distinct “cores”** for simulating and controlling the **C/S Arctic Drillship** (or similar vessels) under wave disturbances:

1. **`jax_core/`** – A JAX-based pipeline specialized for **high-performance** simulation and **differentiable** meta-learning or HPC. Great for large-scale or repeated simulations where speed is crucial. It can be used for all types of machine learning (ML) and has the most memory- and speed-efficient simulators (soon voyager) and controllers. 

2. **`numpy_core/`** – Straightforward reference controllers in plain NumPy, plus a **highly modular and up-to-date Gym** environment suitable for **MPC**, **standardized testing**, or your own self-defined tasks.

3. **`torch_core/`** – A PyTorch-based environment and code, focusing on **reinforcement learning (RL)** and advanced ML-driven controllers. Its Gym environment conceptually mirrors the NumPy version but is oriented more toward policy training and neural-based approaches.


Within each core, you’ll find relevant sub-directories such as `simulator/`, `controllers/`, and so on. For more details, consult each core’s individual `README.md`.

---

## Why Three Cores?

- **NumPy Core**  
  - **Language Simplicity**: Standard Python + NumPy.  
  - **Comprehensive & Modular Gym**: Facilitates model-predictive control (MPC), advanced testing, or custom tasks within a **modular environment**.  
  - **Lower Learning Curve**: Ideal if you just need a *simple environment* or want to quickly spin up a tested controller.

- **PyTorch Core**  
  - **RL & Machine Learning**: Tailored for policy training (e.g., PPO, DDPG), gradient-based controllers, or model-based RL in **Torch**.  
  - **Gym-Like Environment**: Structured similarly to NumPy’s environment but aiming for direct integration with PyTorch RL frameworks.  
  - **Deep Learning Ecosystem**: Integrate with Torch libraries, stable baselines, or other external ML toolkits.

- **JAX Core**  
  - **High-Performance + Differentiability**: JAX’s `jit`, `vmap`, `scan` accelerate repeated or large-scale simulations.  
  - **Meta-Learning**: Perfect for backprop through an entire simulation (including wave models and vessel ODEs).  
  - **Speed Gains**: Observably faster than naive Python for large wave sets (`N`) or small `dt`.

---

## Demonstration Scripts (JAX)

There are **two** JAX demos to illustrate performance differences between **pure JAX** and more **traditional** simulation:

1. **`demos/demo_jax/notebook_demo_jax.py`**  
   - Loads vessel parameters in JAX, sets up a JONSWAP wave environment, and runs a **pure JAX** RK4 integration loop.  
   - Times the simulation, plots positions/headings, and highlights how JAX remains fast even with large wave discretizations or small time-steps.

2. **`demos/demo_jax/notebook_demo_jax_mclsimpy.py`**  
   - Similar scenario but uses **`mclsimpy`** (pure Python object-oriented style).  
   - Demonstrates how “vanilla” Python code can slow down for large `N` or finer `dt`.  
   - Compare speed with the first script to see JAX’s performance advantage.

> **Tip**: **Run both** demos and observe the difference in simulation times.

---

## Installation & Requirements

You will need:
1. **Python 3.7+**  
2. **Package Dependencies**  
   - **Base**:
     ```bash
     pip install numpy matplotlib tqdm
     ```
   - **Torch Core**:
     ```bash
     pip install torch torchvision
     ```
   - **JAX Core**:
     ```bash
     # CPU-only JAX
     pip install --upgrade "jax[cpu]"
     # or GPU version (see https://github.com/google/jax#installation)
     ```
   - **mclsimpy** (used in the second JAX demo):
     ```bash
     pip install git+https://github.com/NTNU-MCS/mclsimpy.git@master
     ```
3. **Optional HPC** or GPU libraries for advanced setups.

*(Check each core’s own README for any additional requirements or recommended RL libraries.)*

---

## Running the JAX Demos

1. **Activate your Python environment** (conda, venv, etc.).
2. **Change to the demos directory** and run the pure-JAX script:
   ```bash
   cd demos/demo_jax
   python notebook_demo_jax.py
   ```
   - Observe timing output, final states, etc.  
   - Plots appear for position/time or XY path.

3. **Run the mclsimpy-based script**:
   ```bash
   python notebook_demo_jax_mclsimpy.py
   ```
   - Same wave and vessel parameters but in Python OOP style.
   - Compare speeds against the pure-JAX version.

---

## Directory Structure (High Level)

```
.
├── numpy_core/
│   ├── allocations/
│   ├── controllers/
│   ├── gym/                # Highly tested gym for MPC or standardized tasks
│   └── ...
├── torch_core/
│   ├── allocations/
│   ├── controllers/
│   ├── gym/                # RL-oriented environment
│   └── ...
├── jax_core/
│   ├── simulator/
│   │   ├── waves/
│   │   ├── vessels/
│   │   └── ...
│   ├── meta_adaptive_ctrl/
│   └── ...
├── demos/
│   └── demo_jax/
│       ├── notebook_demo_jax.py
│       ├── notebook_demo_jax_mclsimpy.py
│       └── ...
├── data/                   # Potential training_data/ & testing_results/
├── figures/                # Output for plots
├── environment_tensor.txt  # Example environment / pip freeze references
└── README.md               # (This file)
```

---

## Further Reading & Next Steps

- **`numpy_core/README.md`** – Detailed info on the simplest controllers, advanced MPC or standardized testing, and the most stable Gym environment.  
- **`torch_core/README.md`** – Explanation of RL or advanced ML usage with PyTorch.  
- **`jax_core/README.md`** – HPC techniques, differentiable wave modeling, meta-learning architecture.  

---

## Contributing

Pull requests and issues are welcome! If you have new wave models, RL pipelines, or advanced adaptive controllers, we’d love to integrate them.

---

## License

Distributed under the [MIT License](LICENSE).  
See the repository root for details.
```
