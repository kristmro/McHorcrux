# McHorcrux
**OBS OBS UNDER DEVELOPMENT**

This repository contains **three distinct simulation and control frameworks** designed for simulating and controlling the **C/S Arctic Drillship** (and similar marine vessels) under wave disturbances. Developed for Kristian Magnus Roen's master's thesis in Marine Cybernetics (2025), the repository includes:

### 1. `jax_core/`
A high-performance, differentiable pipeline based on [JAX](https://github.com/jax-ml/jax) and adapted from the [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator. It provides **significantly enhanced computational speed** and memory efficiency, making it ideal for large-scale or repeated simulations and extensive machine learning applications. This core also features an adapted marine-focused version of the meta-trained adaptive controller from [Richards et al. (2021)](https://github.com/StanfordASL/Adaptive-Control-Oriented-Meta-Learning/tree/master).

### 2. `numpy_core/`
Features a highly modular Gym environment called **McGym**, complete with live visualization capabilities, built around the standard [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator. McGym follows an API structure similar to [OpenAI's Gym](https://github.com/openai/gym), offering standardized tasks and the flexibility to define custom scenarios, including dynamic or static obstacles influenced by wave motions. Ideal for **Monte Carlo simulations** or for real-time testing of custom controllers.

### 3. `torch_core/`
A [PyTorch](https://github.com/pytorch/pytorch)-powered variant of the [mcsimpy](https://github.com/NTNU-MCS/mcsimpy) simulator designed explicitly for machine learning integration. Leveraging PyTorch’s extensive ML modules, this core excels in **reinforcement learning (RL)** and sophisticated ML-driven controller implementations, fully harnessing McGym's capabilities.

Each core contains specialized subdirectories such as `simulator/`, `controllers/`, and others. For detailed information and instructions, please refer to the individual `README.md` files within each core.
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
        # CPU-only JAX
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
