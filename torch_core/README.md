# Torch Core

This directory provides a **PyTorch-based** simulation and control pipeline for the **C/S Arctic Drillship (CSAD)** model from Marine Cybernetics (NTNU). It replicates the logic of the NumPy-based MC-Gym but uses **PyTorch** to enable end-to-end differentiability and potentially gradient-based learning or optimization methods.

Example of use with meta-learning: [*Online-Meta-Adaptive-Control*](https://github.com/GuanyaShi/Online-Meta-Adaptive-Control/tree/main)

---

## Key Features

1. **Differentiable MC-Gym Environment**  
   - Defined in [`gym/mc_gym_csad_torch.py`](./gym/mc_gym_csad_torch.py).  
   - Uses a **6-DOF** vessel model (`CSAD_6DOF`) and **wave load** modules implemented in PyTorch.  
   - Follows a typical “gym-like” API (`step()`, `reset()`, etc.) for integration with RL or custom control loops.

2. **PyTorch Modules for Dynamics & Waves**  
   - **Vessels**: [`simulator/vessels/csad_torch.py`](./simulator/vessels/csad_torch.py) implements the vessel as an `nn.Module` (or uses `torch.tensor` operations).  
   - **Wave Forces**: [`simulator/waves/wave_load_torch.py`](./simulator/waves) and [`wave_spectra_torch.py`](./simulator/waves) define JONSWAP-based wave loads entirely in PyTorch, allowing partial backprop through wave computations.

3. **Controllers & Thruster Allocation**  
   - Found in [`controllers/`](./controllers) and [`allocations/`](./allocations).  
   - **PID** or **model-based** controllers can be fully or partially differentiable if implemented in Torch.  
   - Thruster allocation routines convert high-level forces to individual thruster commands.

4. **Reference Filtering & Utility Functions**  
   - A **third-order reference filter** (`ThrdOrderRefFilter`) resides in [`ref_gen/reference_filter.py`](./ref_gen/reference_filter.py).  
   - Various Torch-based or bridging utilities in [`simulator/utils.py`](./simulator/utils.py) and [`torch_core/utils.py`](./utils.py) handle coordinate transforms, wave calculations, etc.

5. **NumPy Dependencies**  
   - **IMPORTANT**: Some code paths (e.g., wave parameter generation, random phases, final plotting, `pygame` rendering) still rely on **NumPy** arrays.  
   - The environment mixes Torch tensors with NumPy logic, particularly for convenience in `matplotlib` or `pygame`.  
   - Future work could replace these sections with pure Torch or better GPU support.

6. **Device Management**  
   - The simulator is set up to default to **CPU** usage.  
   - **TODO**: Adapt references (e.g., `torch.device('cpu')`) to allow GPU usage, and ensure wave-libraries or parameter generation also move to the same device.

---

## Directory Structure

```
torch_core/
├── allocations/
│   ├── allocation_psudo.py          # Thruster allocation in Torch or bridging code
│   └── __init__.py
├── controllers/
│   ├── model_based.py               # Example differentiable model-based controller
│   ├── pid.py                       # Torch-friendly PID controller
│   └── __init__.py
├── gym/
│   ├── mc_gym_csad_torch.py         # Differentiable MC-Gym environment (step/reset)
│   └── __init__.py
├── main/
│   ├── model_pd_main.py            # Demo: model-based PD control
│   ├── pid_main.py                  # Demo: simple PID loop
│   └── __init__.py
├── ref_gen/
│   ├── reference_filter.py          # ThrdOrderRefFilter in Torch
│   └── __init__.py
├── simulator/
│   ├── __init__.py
│   ├── utils.py                     # Shared transforms, wave logic
│   ├── vessels/
│   │   ├── csad_torch.py           # Torch-based 6DOF vessel model
│   │   ├── vessel_torch.py         # Possibly a base vessel class
│   │   └── __init__.py
│   └── waves/
│       ├── wave_load_torch.py      # Wave forcing in Torch
│       ├── wave_spectra_torch.py   # Torch-based JONSWAP, etc.
│       └── __init__.py
├── thruster/
│   ├── thruster_data.py            # Thruster performance definitions
│   ├── thruster_dynamics.py        # Torch-based thruster dynamics
│   └── thruster.py
├── utils.py                         # Additional bridging utilities (NumPy ↔ Torch)
└── __init__.py
```

---

## Usage

1. **Install Requirements**  
   - **PyTorch**:  
     ```bash
     pip install torch torchvision torchaudio
     ```  
   - **NumPy**, **matplotlib**, **pygame** (for wave parameters, plotting, or real-time rendering):
     ```bash
     pip install numpy matplotlib pygame
     ```

2. **Run a Demo**  
   For instance, run the **model-based PD** example:
   ```bash
   python main/model_pd_main.py
   ```
   - Initializes the Torch-based **MC-Gym** environment (`mc_gym_csad_torch.py`).  
   - Applies a PD controller (`controllers/model_based.py`) under wave disturbances.  
   - Optionally renders via `pygame` and plots final trajectories with `matplotlib`.

3. **Gym-Like API**  
   ```python
   from torch_core.gym.mc_gym_csad_torch import McGym

   env = McGym(dt=0.1, grid_width=15, grid_height=6, render_on=True)
   obs = env.reset()
   done = False
   while not done:
       action = [0.0, 0.0, 0.0]  # e.g. zero control or from RL policy
       obs, done, info, reward = env.step(action)
   ```
   - Returned `obs` is a dictionary with `'eta'`, `'nu'`, `'goal'`, etc.  
   - **Note**: Some arrays remain NumPy-based after `.cpu().numpy()` conversions.

4. **Differentiability Caveats**  
   - The main simulation loop (vessel dynamics, wave forcing) is mostly Torch-based. However, random wave seeds, JONSWAP parameter creation, or PyGame visualization use NumPy.  
   - **TODO**: If you need full GPU acceleration or gradient-based wave design, unify the wave generation code with Torch.  
   - Keep an eye out for `.detach()` or `.numpy()` calls that break the computational graph.

5. **Plotting & Visualization**  
   - Use `render_on=True` for **pygame** real-time rendering.  
   - The environment logs boat state/velocity in Python lists, which are converted to NumPy for **matplotlib** after simulation.  
   - Call `env.plot_trajectory()` (if implemented) once `done` to visualize the final results.

---

## Known Limitations & TODO

- **Hardcoded CPU Usage**: The environment does not automatically detect GPU devices. Adjust the code to move Tensors or modules to `cuda` if desired.  
- **NumPy Dependencies**: Large portions of wave initialization and randomization are still in NumPy. For end-to-end backprop or GPU usage, consider rewriting them in Torch.  
- **Action/Observation Spaces**: Not fully specified in a standard Gym `Box` format. Extend as needed for RL libraries.  

---

## License

This code is distributed under the [MIT License](../LICENSE). Refer to the repository’s root LICENSE file for more details.

---

## Contributing

Contributions and suggestions are welcome!  
Please open an issue or pull request if you encounter bugs or have feature ideas.

---

*For more context on the entire repository (including `numpy_core/` or `jax_core/`), see the main [README.md](../README.md) at the project root.*
```
