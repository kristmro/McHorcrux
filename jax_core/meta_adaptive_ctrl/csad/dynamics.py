
import jax
import jax.numpy as jnp
from jax_core.utils import six2threeDOF, Rz, J, three2sixDOF
from jax_core.simulator.vessels.csad_jax import load_csad_parameters
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
# Updated import: now use the new jit-compatible module
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, WaveLoad, wave_load

# --------------------------------------------------------------------------
# Load vessel parameters and set up initial state (functional style)
# --------------------------------------------------------------------------
config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
params_jit = load_csad_parameters(config_file)
M = six2threeDOF(params_jit["M"])
D = six2threeDOF(params_jit["D"])
G = six2threeDOF(params_jit["G"])

def prior_3dof(q, dq, M=M, D=D, G=G):
    return M, D, G, Rz(q[-1])

def prior_6dof(q, dq):
    config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
    params_jit = load_csad_parameters(config_file)
    M = params_jit["M"]
    D = params_jit["D"]
    G = params_jit["G"]    
    return M, D, G, Rz(q[-1]), J(q)

def plant(q, dq, u, f_ext, prior=prior_3dof):
    M, D, G, R = prior(q, dq)
    ddq = jax.scipy.linalg.solve(M, six2threeDOF(f_ext) + R @ u - D @ dq - G @ q, assume_a='pos')
    dq = R @ dq
    return dq, ddq

# def plant_6dof(q, dq, u, f_ext, prior=prior_6dof):
#     M, D, G, R, J = prior(q, dq)
#     jnp.where(u.shape[0] == 6,  ddq = jax.scipy.linalg.solve(M, f_ext + J @ u - D @ dq - G @ q, assume_a='pos'))
#     return ddq

def disturbance(wave_parm, key, N=15,
                config_file="/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"):
    hs, tp, wave_dir = wave_parm
    print(hs, tp, wave_dir)
    wp = 2 * jnp.pi / tp       # Peak frequency
    wmin = 0.5 * wp
    wmax = 3.0 * wp
    wave_freq = jnp.linspace(wmin, wmax, N)  # Frequencies in rad/s
    omega, wave_spectrum = jonswap_spectrum(wave_freq, hs, tp, gamma=3.3)
    dw = (wmax - wmin) / N
    wave_amps = jnp.sqrt(2.0 * wave_spectrum * dw)
    rand_phase = jax.random.uniform(key, shape=(N,), minval=0, maxval=2 * jnp.pi)
    wave_dir = jnp.deg2rad(wave_dir)
    wave_angles = jnp.ones(N) * wave_dir

    # Initialize wave load using jit-compatible init function.
    wl = init_wave_load(
        wave_amps=wave_amps,
        freqs=omega,
        eps=rand_phase,
        angles=wave_angles,
        config_file=config_file,
        rho=1025,
        g=9.81,
        dof=6,
        depth=100,
        deep_water=True,
        qtf_method="Newman",
        qtf_interp_angles=True,
        interpolate=True
    )
    return wl



# # ------------------------------------------------------------------------------
# # Simulation of Wave Loads using wave_load
# # ------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# # Simulation time parameters
# T = 60.0         # Total simulation time in seconds
# dt = 0.01        # Time step in seconds
# t = jnp.arange(0.0, T, dt)

# # Define example wave parameters:
# #   hs: significant wave height [m]
# #   tp: peak wave period [s]
# #   wave_dir_deg: wave direction in degrees
# hs = 5.0/90
# tp = 18.0*jnp.sqrt(1/90)  # Peak period in seconds
# wave_dir_deg = 12.0

# # Number of frequency components to consider in the spectrum
# N = 15

# # Compute the peak frequency and define the frequency range
# wp = 2 * jnp.pi / tp       # peak frequency [rad/s]
# wmin = 0.5 * wp
# wmax = 3.0 * wp
# wave_freq = jnp.linspace(wmin, wmax, N)

# # Compute the JONSWAP spectrum for the given frequencies
# omega, wave_spectrum = jonswap_spectrum(wave_freq, hs, tp, gamma=3.3)
# dw = (wmax - wmin) / N
# wave_amps = jnp.sqrt(2.0 * wave_spectrum * dw)

# # Create random phases for each frequency component
# key = jax.random.PRNGKey(0)
# rand_phase = jax.random.uniform(key, shape=(N,), minval=0, maxval=2 * jnp.pi)

# # Convert wave direction to radians and create an array of identical wave angles
# wave_dir_rad = jnp.deg2rad(wave_dir_deg)
# wave_angles = jnp.ones(N) * wave_dir_rad

# # Define the configuration file (adjust the path if needed)
# config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"

# # Initialize the wave load object using the jit-compatible init function.
# wl = init_wave_load(
#     wave_amps=wave_amps,
#     freqs=omega,
#     eps=rand_phase,
#     angles=wave_angles,
#     config_file=config_file,
#     rho=1025,           # water density in kg/m^3
#     g=9.81,             # gravitational acceleration in m/s^2
#     dof=6,              # degrees of freedom (6-DOF vessel model)
#     depth=100,          # water depth in meters
#     deep_water=True,    # flag for deep water conditions
#     qtf_method="Newman",
#     qtf_interp_angles=True,
#     interpolate=True
# )

# # Evaluate the wave load over time.
# # Here we use the vectorized 'wave_load' function to compute the load at each time instance.
# # It is assumed that 'wave_load' accepts a wave load object and a time argument and returns a force vector.
# f_wave = jax.vmap(lambda ti: wave_load(t=ti,eta=jnp.zeros(6),wl=wl))(t)

# # Convert the result to NumPy arrays for plotting.
# t_np = np.array(t)
# f_wave_np = np.array(f_wave)

# # ------------------------------------------------------------------------------
# # Plot the Wave Load Components versus Time
# # ------------------------------------------------------------------------------

# # For a 6-DOF load, we typically have:
# #   Component 0: Surge load (x-direction)
# #   Component 1: Sway load (y-direction)
# #   Component 2: Heave load (z-direction)
# # Adjust the components as necessary for your application.
# plt.figure(figsize=(10, 6))
# plt.plot(t_np, f_wave_np[:, 0], label="Surge Load", linewidth=2)
# plt.plot(t_np, f_wave_np[:, 1], label="Sway Load", linewidth=2)
# plt.plot(t_np, f_wave_np[:, 5], label="Yaw Load", linewidth=2)
# plt.xlabel("Time [s]")
# plt.ylabel("Wave Load Force [N]")
# plt.title("Wave Load Components over Time")
# plt.legend()
# plt.grid(True)
# plt.show()
