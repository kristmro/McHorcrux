#!/usr/bin/env python3
"""
compare.py

This script compares the QTF matrices computed by the NumPy-based and JAX-based
wave load implementations. It initializes both versions with the same wave
parameters and vessel configuration, extracts the QTF matrices, computes the
difference, and then displays basic statistics along with plots for a visual
comparison of a representative slice.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time

# --- Import the NumPy version of wave load ---
from mclsimpy.waves import WaveLoad as NumpyWaveLoad

# --- Import the JAX version of wave load and required packages ---
import jax
import jax.numpy as jnp
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load as init_jax_wave_load
from jax_core.simulator.waves.wave_load_jax_jit import first_order_loads as jax_first_order_loads

# ------------------------------
# Set up identical wave parameters
# ------------------------------

# Simulation parameters for wave properties
hs    = 5.0 / 90.0  # Significant wave height [m]
tp    = 9.0 * np.sqrt(1 / 90.0)  # Peak period [s]
gamma = 3.3  # JONSWAP peak factor

wp   = 2 * np.pi / tp   # Peak frequency [rad/s]
wmin = 0.5 * wp
wmax = 3.0 * wp
N    = 100             # Number of wave components

# Generate frequency arrays for NumPy and JAX
freqs_np = np.linspace(wmin, wmax, N)
freqs_jax = jnp.linspace(wmin, wmax, N)

# Dummy wave spectrum data is used (actual spectrum not needed for QTF interpolation)
dw = (wmax - wmin) / N
# Using a dummy spectrum (ones) so that the wave amplitudes are computed similarly;
# note: the QTF computation does not use the amplitudes.
wave_spectrum_np = np.ones(N)
wave_spectrum_jax = jnp.ones(N)
wave_amps_np = np.sqrt(2.0 * wave_spectrum_np * dw)
wave_amps_jax = jnp.sqrt(2.0 * wave_spectrum_jax * dw)

# Random phases
rand_phase_np = np.random.uniform(0, 2 * np.pi, size=N)
rand_phase_jax = jax.random.uniform(jax.random.PRNGKey(0), shape=(N,), minval=0.0, maxval=2 * jnp.pi)

# Constant incident wave angles (e.g. 180°)
wave_angles_np = np.ones(N) * (np.pi )
wave_angles_jax = jnp.ones(N) * (jnp.pi)

# Configuration file path (update this path as needed)
config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"  # Ensure this file exists and is compatible with both versions

# ------------------------------
# Initialize WaveLoad objects from both implementations
# ------------------------------

# Create NumPy-based wave load instance
start_time = time.time()
numpy_wl_geo = NumpyWaveLoad(
    wave_amps=wave_amps_np,
    freqs=freqs_np,
    eps=rand_phase_np,
    angles=wave_angles_np,
    config_file=config_file,
    interpolate=True,
    qtf_method="geo-mean",
    deep_water=True
)
print(f"NumPy-based wave load init time: {time.time() - start_time:.2f} seconds")

# Create JAX-based wave load instance
start_time = time.time()
jax_wl_geo = init_jax_wave_load(
    wave_amps=wave_amps_jax,
    freqs=freqs_jax,
    eps=rand_phase_jax,
    angles=wave_angles_jax,
    config_file=config_file,
    rho=1025,
    g=9.81,
    dof=6,
    depth=100,
    deep_water=True,
    qtf_method="geo-mean",
    qtf_interp_angles=True,
    interpolate=True
)
print(f"JAX-based wave load init time: {time.time() - start_time:.2f} seconds")

# =============================================================================
# Compare RAO Interpolation Outputs
# =============================================================================

print("\n=== Comparing RAO Interpolation Outputs ===")

# For both implementations, we need a relative incident angle array.
# In the NumPy implementation, _relative_incident_angle is a method.
# Use a representative heading, for example, heading = 0 (radians).
heading_for_test = 2

# ----------------------------
# NumPy version: use the internal _relative_incident_angle and _rao_interp.
# ----------------------------
rel_angle_np = numpy_wl_geo._relative_incident_angle(heading_for_test)  
# This returns a vector of relative incident angles (one per wave component)
rao_amp_np_interp, rao_phase_np_interp = numpy_wl_geo._rao_interp(rel_angle_np)

# ----------------------------
# JAX version: use the functional equivalents.
# We assume that in jax_core/simulator/waves/wave_load_jax_jit.py, you have:
#   - relative_incident_angle(angles, heading)
#   - rao_interp(forceRAOamp, forceRAOphase, qtf_angles, rel_angle)
# Import them as follows:
# (If not already imported at the top, add these imports)
from jax_core.simulator.waves.wave_load_jax_jit import relative_incident_angle as jax_relative_incident_angle
from jax_core.simulator.waves.wave_load_jax_jit import rao_interp as jax_rao_interp

# Compute the relative incident angle for the JAX implementation.
# Here, jax_wl.angles is a JAX array of the original wave incident angles.
rel_angle_jax = jax_relative_incident_angle(jax_wl_geo.angles, heading_for_test)
# Compute the RAO interpolation using the JAX function.
rao_amp_jax_interp, rao_phase_jax_interp = jax_rao_interp(jax_wl_geo.forceRAOamp,
                                                          jax_wl_geo.forceRAOphase,
                                                          jax_wl_geo.qtf_angles,
                                                          rel_angle_jax)
# Convert the JAX outputs to NumPy arrays for comparison.
rao_amp_jax_interp_np = np.array(rao_amp_jax_interp)
rao_phase_jax_interp_np = np.array(rao_phase_jax_interp)

# ----------------------------
# Compare the shapes.
# ----------------------------
print("RAO Interpolation Output Shapes:")
print("  NumPy rao_amp:", rao_amp_np_interp.shape, "rao_phase:", rao_phase_np_interp.shape)
print("  JAX   rao_amp:", rao_amp_jax_interp_np.shape, "rao_phase:", rao_phase_jax_interp_np.shape)

# ----------------------------
# Compute and print difference statistics.
# ----------------------------
rao_amp_diff = np.abs(rao_amp_np_interp - rao_amp_jax_interp_np)
rao_phase_diff = np.abs(rao_phase_np_interp - rao_phase_jax_interp_np)
print("RAO Amplitude Interpolation - Max difference:", np.max(rao_amp_diff),
      "Mean difference:", np.mean(rao_amp_diff))
print("RAO Phase Interpolation - Max difference:", np.max(rao_phase_diff),
      "Mean difference:", np.mean(rao_phase_diff))

# ----------------------------
# Plot a representative comparison for one DOF.
# ----------------------------
# Here, we plot the RAO amplitude as a function of the QTF angles (in radians)
# for DOF 0. (If _qtf_angles is stored in the NumPy wave load, we use that for x-axis).
qtf_angles_deg = np.rad2deg(numpy_wl_geo._qtf_angles)

# Ensure the x-axis data matches the shape of the y-axis data
# Interpolate qtf_angles_deg to match the shape of the RAO amplitude arrays
from scipy.interpolate import interp1d

# Interpolate qtf_angles_deg to 100 points to match the RAO amplitude arrays
interp_func = interp1d(np.linspace(0, 360, len(qtf_angles_deg)), qtf_angles_deg, kind="linear")
qtf_angles_deg_interp = interp_func(np.linspace(0, 1, rao_amp_np_interp.shape[1]))

plt.figure(figsize=(10, 6))
plt.plot(qtf_angles_deg_interp, rao_amp_np_interp[0, :], label="NumPy RAO Amp (DOF 0)", marker="o")
plt.plot(qtf_angles_deg_interp, rao_amp_jax_interp_np[0, :], label="JAX RAO Amp (DOF 0)", marker="x", linestyle="--")
plt.xlabel("QTF Angle (deg)")
plt.ylabel("Interpolated RAO Amplitude")
plt.title("Comparison of RAO Amplitude Interpolation for DOF 0")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot the phase difference
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(qtf_angles_deg_interp, rao_phase_np_interp[0, :], label="NumPy RAO Phase (DOF 0)", marker="o")
plt.plot(qtf_angles_deg_interp, rao_phase_jax_interp_np[0, :], label="JAX RAO Phase (DOF 0)", marker="x", linestyle="--")
plt.xlabel("QTF Angle (deg)")
plt.ylabel("Interpolated RAO Phase (rad)")
plt.title("Comparison of RAO Phase Interpolation for DOF 0")
plt.legend()
plt.tight_layout()
plt.show()

