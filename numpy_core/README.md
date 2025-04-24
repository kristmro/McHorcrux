# NumPy Core

This directory hosts a **NumPy-based** motion-control pipeline for the **C/S Arctic Drillship (CSAD)** model from Marine Cybernetics (NTNU). It implements a gym-like simulation environment—called **MC-Gym**—designed for traditional control, **reinforcement learning (RL)**, or **model predictive control (MPC)**. While earlier versions explored meta-learning and model-based RL, **this current focus** is on standard controllers, observers, and wave simulation.

---

## Key Features

1. **MC-Gym Environment**  
   - Defined in [`gym/mc_gym_csad_numpy.py`](./gym/mc_gym_csad_numpy.py).  
   - Uses **mclsimpy** for a 6-DOF vessel simulator and JONSWAP wave loads.  
   - Offers a “step/reset” API similar to OpenAI Gym, with optional real-time rendering (via `pygame`) and post-simulation plotting (via `matplotlib`).

2. **Adaptive Controllers**  
   - Available in [`controllers/`](./controllers).  
   - Handles surge-speed and heading control under wave disturbances (e.g., backstepping-like approaches, MRAC-based control).

3. **Thruster Allocation**  
   - In [`allocations/allocator_psudo.py`](./allocations/allocator_psudo.py).  
   - Maps high-level control commands (e.g., desired surge or yaw moments) into individual thruster signals using a pseudo-inverse approach.

4. **Observers & Reference Filtering**  
   - **Observers:** [`observers/`](./observers) includes Extended Kalman Filters (`ekf.py`), linear time-varying Kalman filters (`ltv_kf.py`), etc.  
   - **Third-Order Ref Filter:** [`ref_gen/reference_filter.py`](./ref_gen/reference_filter.py) generates smooth trajectories for tracking.

5. **Minimal Dependencies**  
   - Primarily **NumPy** + standard Python libraries.  
   - Uses `mclsimpy` for vessel and wave simulation, and optionally `pygame` + `matplotlib` for rendering/plots.

---

## Installation

1. **Python Environment**  
   - Requires **Python 3.7+** and a virtual environment is recommended.

2. **Required Packages**  
   ```bash
   pip install numpy matplotlib pygame
   ```
   (Pygame is optional but needed for real-time rendering.)

3. **mclsimpy**  
   This environment relies on [mclsimpy](https://github.com/NTNU-MCS/mclsimpy). Install it directly from GitHub:
   ```bash
   pip install git+https://github.com/NTNU-MCS/mclsimpy.git@master
   ```

---

## Directory Overview

```
numpy_core/
├── allocations/
│   ├── allocator_psudo.py       # Thruster allocation using pseudo-inverse
│   └── __init__.py
├── controllers/
│   ├── adaptive_fs_controller.py # Adaptive backstepping-like approach (wave force estimation)
│   ├── adaptive_seakeeping.py    # MRAC-based heading & surge control
│   └── __init__.py
├── gym/
│   ├── mc_gym_csad_numpy.py      # "MC-Gym": 6-DOF vessel + wave loads + RL/MPC interface
│   └── __init__.py
├── main/
│   ├── adap_fs_main.py           # Demo script: run the adaptive FS controller in MC-Gym
│   ├── ekfmain.py                # Demo script: run environment + EKF observer
│   ├── seekeeping_main.py        # Demo script: run MRAC-based seakeeping controller
│   └── __init__.py
├── observers/
│   ├── ekf.py                    # Extended Kalman Filter
│   ├── ltv_kf.py                 # Linear time-varying Kalman Filter
│   ├── nonlinobs.py              # Nonlinear observer (placeholder)
│   └── __init__.py
├── ref_gen/
│   ├── reference_filter.py       # 3rd-order reference filter (ThrdOrderRefFilter)
│   └── __init__.py
├── thruster/
│   ├── thruster.py               # Simplified thruster model
│   ├── thruster_data.py          # Thruster performance curves
│   ├── thruster_dynamics.py      # Basic thruster dynamics
│   └── __init__.py
├── utils.py                      # Helper utilities for rotation, wave modeling, etc.
└── __init__.py
```

---

## Usage

1. **Run a Demo**  
   Try the adaptive FS controller:
   ```bash
   python main/adap_fs_main.py
   ```
   - Initializes the **MC-Gym** environment (`mc_gym_csad_numpy.py`).  
   - Applies the adaptive FS controller to manage surge and heading under wave loads.  
   - Optionally renders the simulation in a pygame window and plots final trajectories.

2. **Integrate with RL or MPC**  
   The `gym` environment provides the typical `step(action) -> obs, done, info, reward` cycle. You can integrate it with RL libraries or design your own control loops that expect a “gym-like” interface.

3. **Modify or Create Controllers**  
   - Add your own controller scripts in [`controllers/`](./controllers).  
   - Check the `main/` folder for examples of how to instantiate the environment and run a simulation loop.

---

## About MC-Gym

Below is a **high-level summary** of the MC-Gym environment:

- **6-DOF Vessel Simulation**: Powered by `mclsimpy.simulator.csad.CSAD_DP_6DOF`, which uses RK4 integration and responds to thruster forces plus wave disturbances.  
- **Wave Modeling**: Implements wave loads from a JONSWAP spectrum, discretizing over frequency ranges.  
- **Grid-Based Domain**: Defaults to an (N×E) rectangle. The boat’s position is updated in “north” and “east” coordinates, with optional obstacles and dynamic tasks.  
- **Reward Function & Termination**: Provides placeholders that can be extended for custom tasks.  
- **Optional Four-Corner Test**: A built-in mode for quickly evaluating maneuverability by stepping through predefined waypoints.

Feel free to explore the docstrings in [`mc_gym_csad_numpy.py`](./gym/mc_gym_csad_numpy.py) for a complete breakdown of parameters, wave load generation, obstacle definitions, and other features.

---

## Future Development

- **Meta-Learning & MBRL**: Legacy code for advanced learning approaches has been removed from this directory. The environment remains open to future RL, MPC, or adaptive control methods.  
- **Extended Observers & State Estimation**: Additional observer designs (e.g., UKF, Particle Filters) can be added for better state estimation in rough seas.  
- **Task/Reward Design**: The environment is ready for custom tasks (docking, collision avoidance, etc.). Expand the reward function, obstacles, or dynamic goals to suit your project needs.

---

## License

Distributed under the [MIT License](../LICENSE).  
Refer to the repository’s root `LICENSE` file for more details.

---

## Contributing

Contributions, suggestions, and bug reports are welcome!  
Please open an issue or submit a pull request with your improvement or fix.

---

*For an overview of the entire repository (including `torch_core/` or `jax_core/`), see the main [README.md](../README.md) in the project root.*
```

