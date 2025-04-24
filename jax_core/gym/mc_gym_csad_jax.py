#!/usr/bin/env python3
"""
------------------------------------------------------------------------------------------------------------------------------------------------
6DOF Ship Navigation Simulation Gym (MC-GYM-JAX)
------------------------------------------------------------------------------------------------------------------------------------------------
Author: Kristian Magnus Roen
Date:   2025-03-25

Description:
    This JAX‑based gym environment simulates a 6DOF vessel subject to differentiable wave loads.
    Vessel dynamics are computed functionally using csad_x_dot and integrated via an RK4 integrator.
    Vessel parameters are loaded via load_csad_parameters (init_csad style) and the state is stored as a
    12-element vector (η and ν). Wave loads are computed using init_wave_load and wave_load.
    
    The environment supports user‑defined tasks (or premade ones) and returns a zero reward.
    
    Optional real‑time rendering (pygame) and post‑simulation plotting (matplotlib) are provided.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pygame

# Functional vessel simulation routines:
from jax_core.simulator.vessels.csad_jax import load_csad_parameters, csad_x_dot
from jax_core.utils import rk4_step
# Wave spectrum & load routines:
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, wave_load, WaveLoad
# Conversion utilities:
from jax_core.utils import three2sixDOF, six2threeDOF
# Reference trajectory filter (for four-corner test)
from jax_core.ref_gen.reference_filter import build_filter_matrices, simulate_filter_rk4

# Default config file (adjust the path as needed)
DEFAULT_CONFIG_FILE = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
