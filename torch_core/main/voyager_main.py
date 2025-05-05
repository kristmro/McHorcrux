import time
# Imports Gunnerus model
from torch_core.simulator.vessels.voyager_torch import VOYAGER


# Imports waves
from torch_core.simulator.waves.wave_load_torch import WaveLoad
from torch_core.simulator.waves.wave_spectra_torch import JONSWAP

# Imports utilities
import math
import torch
import matplotlib.pyplot as plt


import cProfile, pstats, io 

start = time.time()

dt = 0.01
simtime = 100
t = torch.arange(0, simtime, dt)

vessel = VOYAGER(dt=dt, method='RK4')
print(f"Init boat took {time.time() - start:.2f} seconds")

eta = torch.zeros((6, len(t)))
nu = torch.zeros((6, len(t)))
wave_load = torch.zeros((6, len(t)))

Uc = 0.0
beta_c = 0

start = time.time()
hs = 3.0/34 # Significant wave height
tp = 14.0*math.sqrt(1/35) # Peak period (be careful using math.sqrt. Use that were tensor is not needed, but only the number)
gamma = 3.3 # Peak factor
N = 100 

# Discretize freq range
wp = 2.*math.pi / tp
wmin, wmax = wp/2., 3.*wp
dw = (wmax - wmin)/N
w = torch.linspace(wmin, wmax, N)


# JONSWAP in PyTorch
jonswap = JONSWAP(w) #accept numpy or torch tensor
freq, spec = jonswap(hs, tp, gamma)  # freq, spec both Tensors
wave_amps = torch.sqrt(2.0 * spec * dw)
eps = torch.rand(N) * (2 * math.pi)
wave_dir = torch.ones(N) * (math.pi / 4.0)

# DiffWaveLoad expects wave_amps, freqs, eps, angles as NumPy or Torch Tensors
waveload = WaveLoad(
    wave_amps,
    freq, #accepts as numpy or torch tensor
    eps,
    wave_dir,
    config_file=vessel._config_file,
    interpolate=True,
    qtf_method="Newman",
    deep_water=True
)
print(f"Init the everything related to the waves took {time.time() - start:.2f} seconds")
print(vessel._eta.device)

# Define initial state explicitly for this cell's runs
eta_init = torch.zeros(6, dtype=torch.float32)
nu_init = torch.zeros(6, dtype=torch.float32)

# Define the simulation loop as a function
def run_simulation_loop(vessel, waveload_obj, time_vec, Uc_val, beta_c_val, tau_ctrl):
    num_steps = len(time_vec)
    # Pre-allocate tensors on the same device as vessel state if possible
    # Assuming vessel state (eta, nu) is on CPU by default unless moved
    device = vessel._eta.device # Get device from vessel's tensor (Use _eta)
    eta_hist = torch.zeros((6, num_steps), dtype=torch.float32, device=device)
    nu_hist = torch.zeros((6, num_steps), dtype=torch.float32, device=device)
    wave_load_hist = torch.zeros((6, num_steps), dtype=torch.float32, device=device)

    # Ensure tau_ctrl is on the correct device and dtype
    tau_ctrl = tau_ctrl.to(device=device, dtype=torch.float32)

    for i in range(num_steps):
        current_eta = vessel.get_eta()
        eta_hist[:, i] = current_eta
        nu_hist[:, i] = vessel.get_nu()

        # Calculate wave load - assuming waveload_obj returns a tensor
        tau_wave = waveload_obj(time_vec[i], current_eta)
        # Ensure tau_wave is a tensor on the correct device/dtype
        # (Add assertion or explicit conversion if waveload_obj might return numpy/list)
        if not isinstance(tau_wave, torch.Tensor):
             # Example: Convert if it's numpy, adjust as needed
             tau_wave = torch.tensor(tau_wave, dtype=torch.float32, device=device)
        else:
             tau_wave = tau_wave.to(device=device, dtype=torch.float32)

        wave_load_hist[:, i] = tau_wave

        # Calculate total tau - should result in a float32 tensor
        tau = tau_ctrl + tau_wave

        # Integrate - vessel.integrate should handle the tensor type internally
        vessel.integrate(Uc_val, beta_c_val, tau)

    return eta_hist, nu_hist, wave_load_hist

# Compiled version
print("\nStarting compiled simulation")
vessel_compiled = VOYAGER(dt=dt, method='RK4')
vessel_compiled.set_eta(eta_init.clone())
vessel_compiled.set_nu(nu_init.clone())
tau_control_compiled = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)

# Compile the entire loop function using 'reduce-overhead' mode
print("Compiling simulation loop function (mode='reduce-overhead')...")
compiled_run_simulation_loop = torch.compile(run_simulation_loop, mode='reduce-overhead')

# Run the compiled function (first run includes compile time)
print("Running compiled function (includes compile time)...")
profiler_compile_run = cProfile.Profile()
start_compile_run = time.time()
profiler_compile_run.enable()
eta_compiled, nu_compiled, wave_load_compiled = compiled_run_simulation_loop(
    vessel_compiled, waveload, t, Uc, beta_c, tau_control_compiled
)
profiler_compile_run.disable()
end_compile_run = time.time()
compile_time = end_compile_run - start_compile_run
print(f"Compiled simulation (including compile time) took {compile_time:.2f} seconds")
print("\n--- Compiled Profile (First Run) --- ")
s_compile_run = io.StringIO()
ps_compile_run = pstats.Stats(profiler_compile_run, stream=s_compile_run).sort_stats('cumulative')
ps_compile_run.print_stats(20) # Print top 20 functions
print(s_compile_run.getvalue())


# --- Plotting ---

# Convert time tensor to numpy for plotting if needed, or keep as tensor if matplotlib handles it
t_np = t.cpu().numpy() # Assuming t is on CPU, convert for safety
eta_np = eta_compiled.cpu().numpy()
nu_np = nu_compiled.cpu().numpy()
wave_load_np = wave_load_compiled.cpu().numpy()

# 1. Position Plot (Separate Subplots for Translation and Rotation)
fig_pos, axes_pos = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig_pos.suptitle('Vessel Position and Orientation Over Time', fontsize=16)

# Translational Position (x, y, z)
axes_pos[0].plot(t_np, eta_np[0, :], label='x (Surge)')
axes_pos[0].plot(t_np, eta_np[1, :], label='y (Sway)')
axes_pos[0].plot(t_np, eta_np[2, :], label='z (Heave)')
axes_pos[0].set_ylabel('Position [m]')
axes_pos[0].set_title('Translational Motion')
axes_pos[0].legend()
axes_pos[0].grid(True)

# Rotational Position (phi, theta, psi)
axes_pos[1].plot(t_np, eta_np[3, :], label=r'$\phi$ (Roll)')
axes_pos[1].plot(t_np, eta_np[4, :], label=r'$\theta$ (Pitch)')
axes_pos[1].plot(t_np, eta_np[5, :], label=r'$\psi$ (Yaw)')
axes_pos[1].set_ylabel('Angle [rad]')
axes_pos[1].set_title('Rotational Motion')
axes_pos[1].legend()
axes_pos[1].grid(True)

axes_pos[1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()


# 2. Velocity Plot (Separate Subplots for Linear and Angular Velocity)
fig_vel, axes_vel = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig_vel.suptitle('Vessel Velocity Over Time', fontsize=16)

# Linear Velocity (u, v, w)
axes_vel[0].plot(t_np, nu_np[0, :], label='u (Surge Velocity)')
axes_vel[0].plot(t_np, nu_np[1, :], label='v (Sway Velocity)')
axes_vel[0].plot(t_np, nu_np[2, :], label='w (Heave Velocity)')
axes_vel[0].set_ylabel('Velocity [m/s]')
axes_vel[0].set_title('Linear Velocity')
axes_vel[0].legend()
axes_vel[0].grid(True)

# Angular Velocity (p, q, r)
axes_vel[1].plot(t_np, nu_np[3, :], label='p (Roll Rate)')
axes_vel[1].plot(t_np, nu_np[4, :], label='q (Pitch Rate)')
axes_vel[1].plot(t_np, nu_np[5, :], label='r (Yaw Rate)')
axes_vel[1].set_ylabel('Angular Velocity [rad/s]')
axes_vel[1].set_title('Angular Velocity')
axes_vel[1].legend()
axes_vel[1].grid(True)

axes_vel[1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.show()


# 3. XY Plot (Top-down view of trajectory)
plt.figure(figsize=(8, 8))
plt.plot(eta_np[1, :], eta_np[0, :]) # Plotting Y vs X (North vs East convention often used)
plt.xlabel('East Position (y) [m]')
plt.ylabel('North Position (x) [m]')
plt.title('Vessel Trajectory (XY Plane)')
plt.grid(True)
plt.axis('equal') # Ensure aspect ratio is equal
plt.show()


# 4. Wave Loads Plot (Individual DOF per subplot)
fig_wave, axes_wave = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
fig_wave.suptitle('Wave Loads Over Time per Degree of Freedom (DOF)', fontsize=16)
dof_labels = ['Surge Force', 'Sway Force', 'Heave Force', 'Roll Moment', 'Pitch Moment', 'Yaw Moment']
units = ['[N]', '[N]', '[N]', '[Nm]', '[Nm]', '[Nm]']

for i in range(6):
    axes_wave[i].plot(t_np, wave_load_np[i, :], label=f'{dof_labels[i]}')
    axes_wave[i].set_ylabel(f'Load {units[i]}')
    axes_wave[i].legend(loc='upper right')
    axes_wave[i].grid(True)
    if i == 0:
        axes_wave[i].set_title('Wave Forces and Moments') # Add title to the first subplot

axes_wave[-1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
plt.show()