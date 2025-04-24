# JAX Core

This directory contains **high-performance**, **JAX-based** simulation and meta-adaptive control code for the **C/S Arctic Drillship** (Greatly inspired by a rotorcraft example in Spencer Richards’s [*Adaptive-Control-Oriented Meta-Learning*](https://arxiv.org/pdf/2103.04490)) The code has been **adapted for maritime vessel dynamics**, preserving the key principle that a **fully differentiable** and **highly efficient** simulator can greatly accelerate training advanced controllers offline.

---

## Overview & Motivation

1. **Rooted in Adaptive-Control-Oriented Meta-Learning**  
   - Code inherits ideas from Spencer Richards’s work on meta-learning for control, which views the adaptation problem as a **bi-level** optimization.  
   - Instead of typical “regression-oriented” meta-learning, the approach is **control-oriented**: we optimize neural-network-based features and the controller gains **directly for robust closed-loop performance**, not just to fit input/output data.

2. **JAX for Speed & Differentiability**  
   - Vessel dynamics, wave loads, and training loops are written in **JAX** to exploit **XLA** compilation, automatic differentiation, vectorization (`vmap`), and parallel scanning (`scan`).  
   - Large-scale or repetitive simulations can thus run orders of magnitude faster than naive Python or NumPy – especially beneficial when meta-optimizing controllers over many reference trajectories.

3. **Maritime Adaptation**  
   - While Spencer’s original setup used a planar rotorcraft, I have **refactored** many of those ideas for a **marine vessel** (the C/S Arctic Drillship and hopefully C/S Voyager).  
   - The external forces come from wave models (`wave_load_jax_jit.py`), referencing advanced JONSWAP spectra. The vessel’s dynamic equations are carefully structured as a JAX-friendly, fully differentiable system.

4. **In-Progress Gym**  
   - A “gym-like” environment exists but is not a primary focus here. JAX’s function-oriented style makes a purely object-oriented gym class less ideal.  
   - Instead, we rely on direct ODE simulations (coupled with neural nets, wave loads, and adaptation laws) for offline training. If you need a standard `env.step()` interface, see [`gym_jax`](./gym) for a prototype that may require further refinement.

---

## Key Scripts

1. **`data_gen.py`**  
   - Generates offline training data by simulating different wave conditions, references, or initial configurations.  
   - Saves states, controls, references, etc. in a `.pkl` for further processing.

2. **`dynamics.py`**  
   - Sets up the ship’s dynamics and any relevant adaptation laws or state-space equations (in JAX).  
   - Reflects the Lagrangian form but specialized to maritime forces, not rotorcraft.

3. **`training.py`**  
   - Implements the *meta-adaptive control* strategy described by Spencer Richards, specifically reworked for the drillship model.  
   - Uses JAX’s transformations (`jit`, `vmap`, `scan`) to run repeated rollouts with different wave loads or references, collecting gradients for the final controller parameters.

4. **`test_single.py`, `test_four_corner.py`**  
   - Testing / evaluation scripts to confirm that the learned adaptive controllers track various reference trajectories (e.g., a single path, a four-corner route).  
   - Loads the `.pkl` results of `training.py` and executes JAX-based ODE simulations.  
   - Useful for quick validation or generating figures.

5. **`plot.py`**  
   - Creates various performance plots (RMS error, control effort, etc.) from stored test results.  
   - Provides a quick visual summary of how effectively the meta-trained controllers adapt to wave disturbances.

6. **`train.sh`**  
   - Simple shell script that loops over multiple seeds or hyperparameters, calling `training.py` systematically for batch experiments.

---

## High-Performance JAX Features

- **JIT Compilation**: All major components (vessel ODE solvers, wave load routines) are wrapped with `@jax.jit` for speed.  
- **Vectorized Simulation**: We often use `jax.vmap` to run many simulations at once, e.g., different wave seeds or references.  
- **Auto-Diff for Gradients**: The entire simulator pipeline is differentiable. In principle, we can compute partial derivatives wrt. wave parameters, vessel inertia, or neural net features with minimal overhead.

---

## Relationship to Spencer’s Paper

Spencer’s paper [*Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems*](https://arxiv.org/pdf/2103.04490) showcased meta-learning on a planar rotorcraft subject to wind. Here, we have:

- Replaced rotorcraft equations with the C/S Arctic Drillship equations – a ship with **wave-induced** disturbances.  
- Maintained the **bi-level optimization** approach: an *inner loop* for real-time adaptation, an *outer loop* that offline “learns to learn” by back-propagating through the entire simulation.  
- Preserved sample-efficiency: we only need a moderate set of “training conditions” (wave heights, wave periods, wave directions) to robustly learn how to adapt.  
- Demonstrated HPC speed-ups from JAX that make large-scale repeated simulations feasible.

---

## Incomplete or Experimental Components

- **JAX Gym**: A partial environment class is present (`jax_core/gym/…`) but not fully integrated with `gym.Env` due to JAX’s pure-functional style.  
- **Advanced Wave Models**: The wave modeling (JONSWAP, second-order effects) is still evolving. Some features may need more real-world calibration.  
- **Real-World Deployment**: While HPC in JAX accelerates training offline, real-time inference can still be done with standard Python or exported as XLA-compiled code. This is an ongoing area of research.

---

## Installation & Dependencies

1. **JAX**: Install with CPU or GPU support per [JAX documentation](https://github.com/google/jax#installation).
2. **Standard Python Packages**:  
   ```bash
   pip install numpy matplotlib tqdm
   ```
3. **(Optional)** Additional HPC libraries can help if your system is large.

---

## Typical Workflow

1. **Generate Data**  
   ```bash
   python data_gen.py
   ```
   - E.g., vary wave heights, directions, or initial states. Saves `.pkl`.

2. **Train**  
   ```bash
   bash train.sh
   ```
   - Or directly:
     ```bash
     python training.py --seed=0 --M=10
     ```
   - Produces meta-trained parameters for adaptive control in `data/training_results/`.

3. **Test**  
   ```bash
   python test_single.py
   ```
   - Runs new references / wave loads with the learned adaptive controller. Saves results to `data/testing_results/`.

4. **Plot**  
   ```bash
   python plot.py
   ```
   - Generates `.png` charts (e.g., tracking errors, control efforts) in `figures/`.

---

## Contributing

If you wish to modify or extend the code for other maritime vessels, advanced wave spectra, or new adaptive laws, please open issues or pull requests. We welcome suggestions for finalizing the JAX-based gym environment or for improved HPC parallelization strategies.

---

*For an overview of the entire repository (including `numpy_core/` or `torch_core/`), see the main [README.md](../README.md) at the project root.* 
```
