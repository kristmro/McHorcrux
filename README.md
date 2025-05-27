![upscalemedia-transformed (1) (1)](https://github.com/user-attachments/assets/2da3aa74-f585-4854-91dd-c32cb1bad24b)
**OBS OBS UNDER DEVELOPMENT**

This repository contains **three distinct simulation and control frameworks** designed for simulating and controlling the three 6 DOF models, **C/S Arctic Drillship**, **R/V Gunnerus** and **C/S Voyager**, under wave disturbances. Developed for Kristian Magnus Roen's master's thesis in Marine Cybernetics (2025), the repository includes:

### 1. `jax_core/`
A high-performance, differentiable pipeline based on [JAX](https://github.com/jax-ml/jax) and adapted from the [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator. It provides **significantly enhanced computational speed** and memory efficiency, making it ideal for large-scale or repeated simulations and extensive machine learning applications like reinforcement learning. This core also features an adapted marine-focused version of the meta-trained adaptive controller from [Richards et al. (2021)](https://github.com/StanfordASL/Adaptive-Control-Oriented-Meta-Learning/tree/master) and adding model-uncertainty and training only the diagonal gains.


### 2. `numpy_core/`
Features a highly modular Gym environment called **mcGym**, complete with live visualization capabilities with [pygame](https://github.com/pygame/), built around the standard [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator. mcGym follows an API structure similar to [OpenAI's Gym](https://github.com/openai/gym), offering standardized tasks and the flexibility to define custom scenarios, including dynamic or static obstacles influenced by wave motions. Ideal for visual testing of custom controllers and benchmark testing. 

### 3. `torch_core/`
A [PyTorch](https://github.com/pytorch/pytorch)-powered variant of the [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator designed explicitly for machine learning integration. Leveraging PyTorch’s extensive ML modules, this core excels in **deep learning** and RL and sophisticated ML-driven controller implementations, fully harnessing mcGym's capabilities.

---
## Overview
![mcHorcrux (1)](https://github.com/user-attachments/assets/30c36695-e6f3-4d72-9f48-0b9582509c5b)

---
## Demonstrations

Below are example runs showcasing the capabilities of our different mcGym cores:

#### Static Obstacle and Goal in NumPy mcGym  
Here the **numpy_core** mcGym runs a static scenario: the vessel must navigate from its start point to a fixed goal while avoiding a stationary circular obstacle. The controller plans a path around the obstacle and holds final heading within the specified tolerance.  

![Static obstacle and goal mcGym](figures/demo_gifs/static_goal_obstical_mcgym.gif)

---

#### Dynamic Obstacles and Moving Goal in NumPy mcGym  
In this dynamic scenario (also using **numpy_core**), both the goal and obstacles move over time: the goal “wiggles” sinusoidally, and two circular obstacles follow independent trajectories. The controller continuously replans to track the moving goal however, it hits one of the obstacles, and the loop is then terminated.  

![Dynamical obstacles and goal mcGym](figures/demo_gifs/dynamic_goal_obstical_mcgym.gif)

---

#### Four-corner DP in Torch mcGym  
This GIF uses the **torch_core** version of mcGym together with a custom `ModelController(nn.Module)`—a pure PyTorch controller that implements a PD law augmented by full model compensation. Because it’s built on `torch.nn`, you can seamlessly swap in learned networks, fine-tune gains via backpropagation, or integrate any other PyTorch module.

![Four-corner DP in Torch mcGym](figures/demo_gifs/4corner_dp_torch.gif)

---
## Simple API
![mcGym](https://github.com/user-attachments/assets/0528284b-9b3b-475e-8b2f-1467843702de)

## The waves and response of R/V Gunnerus
<p float="left">
  <img src="figures/demo_gifs/vessel_motion3d__rvg_waveangle_270.gif" width="50%" />
  <img src="https://github.com/user-attachments/assets/6a84c799-70f6-4aa6-961d-9042fd690af8" width="49%" />
</p>
</p>
<p float="left">
<img src="figures/demo_gifs/wave_motion1d_2.gif" width="50%" />
</p>

---

## Installation & Requirements

You will need:

1. **Python 3.7+**
2. **Package Dependencies**
    - **Base**:
        
        ```bash
        pip install numpy matplotlib tqdm pickle pygame json scipy
        
        ```
        
    - **Torch Core**:
        
        ```bash
        pip install torch torchvision
        
        ```
        
    - **JAX Core**:
        
        ```bash
        pip install -U "jax[cuda12]"
  
        ```
        
    - **mclsimpy** (Simulator for Numpy core):
        
        ```bash
        pip install git+https://github.com/NTNU-MCS/mcsimpy.git@master
        
        ```   

## Contributing

Pull requests and issues are welcome! 

---

## License

Distributed under the [MIT License](LICENSE).  
See the repository root for details.
```
