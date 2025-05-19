# Torch Core

This directory provides a **PyTorch-based** simulation and control pipeline for the **C/S Arctic Drillship (CSAD)**, **R/V Gunnerus** and **C/S Voyager** models from Marine Cybernetics (NTNU). It replicates the logic of the NumPy-based MC-Gym but uses **PyTorch** to enable end-to-end differentiability and potentially gradient-based learning or optimization methods.

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

## Usage

1. **Install Requirements**  
   - **PyTorch**:  
     ```bash
     pip install torch torchvision 
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
   import torch
   # -----------------------------
   #  Constants and Scaling Factors
   # -----------------------------
   LAMBDA = 1 / 90  # Model scaling factor
   TIME_SCALE = np.sqrt(LAMBDA)  # Time scaling
   WAVE_HEIGHT_SCALE = LAMBDA  # Scale wave height accordingly

   # -----------------------------
   #  Simulation Parameters
   # -----------------------------
   dt = 0.01   # Time step 
   simtime = 450   # Total simulation
   start_pos = (2.0, 2.0, 0.0)  # (north, east, heading)

   # Define wave conditions
   wave_conditions = (2.0 * WAVE_HEIGHT_SCALE, 8.0 * TIME_SCALE, 180.0)  # Scaled wave properties

   env = McGym(dt=dt, final_plot=True, render_on=True)
   env.set_task(start_pos=start_pos,  
            wave_conditions=wave_conditions, 
            four_corner_test=True, 
            simtime=simtime,
            ref_omega=[0.3, 0.3, 0.15] #reference gains
            )
   done = False
   ctrl = torch.zeros[6]
   while not done:
      # Get reference trajectory from four-corner test
      eta_d, nu_d, eta_d_ddot, eta_d_dot = env.get_four_corner_nd(step_count) #2 speed ref, BODY/NED
      state, done, info, reward = env.step(ctrl) #can also use env.get_state()


   ```
   - Returned `state` is a dictionary with `'eta'`, `'nu'`, `'goal'`, etc.  
   - **Note**: Some arrays remain NumPy-based after `.cpu().numpy()` conversions.

4. **Differentiability Caveats**  
   - The main simulation loop (vessel dynamics, wave forcing) is mostly Torch-based. However, random wave seeds, JONSWAP parameter creation, or PyGame visualization use NumPy.  
   - **TODO**: If you need full GPU acceleration or gradient-based wave design, unify the wave generation code with Torch.  
   - Keep an eye out for `.detach()` or `.numpy()` calls that break the computational graph.

5. **Plotting & Visualization**  
   - Use `render_on=True` for **pygame** real-time rendering.  
   - The environment logs boat state/velocity in Python lists, which are converted to NumPy for **matplotlib** after simulation.  
   - Use `final_plot=True`

---

## Known Limitations & TODO

- **Hardcoded CPU Usage**: The environment does not automatically detect GPU devices. Adjust the code to move Tensors. 
- **NumPy Dependencies**: Large portions of wave initialization and randomization are still in NumPy. For end-to-end backprop or GPU usage, consider rewriting them in Torch or Cuda.  
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
