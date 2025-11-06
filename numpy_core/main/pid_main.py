
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from mcsimpy.simulator import CSAD_DP_6DOF
    from mcsimpy.waves import JONSWAP, WaveLoad
except ImportError as e:
    raise ImportError("This script requires the 'mclsimpy' package. "
                      "Please install it in your Python environment before running.") from e

# ---------------- Simulation setup ----------------
dt = 0.001          # Time step [s]
simtime = 10.0      # Total simulation time [s]
t = np.arange(0.0, simtime, dt)
N = 12              # Number of wave components

# Instantiate vessel model
vessel = CSAD_DP_6DOF(dt=dt, method="RK4")

# ---------------- Wave load setup -----------------
hs = 5.0            # Significant wave height [m]
tp = 9.0            # Peak period [s]
gamma = 3.3         # Peak enhancement factor

wp = 2.0 * np.pi / tp
wmin = 0.5 * wp
wmax = 3.0 * wp
freqs = np.linspace(wmin, wmax, N)

# JONSWAP spectral density
jonswap = JONSWAP(freqs)
omega, wave_spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)
dw = freqs[1] - freqs[0]
wave_amps = np.sqrt(2.0 * wave_spectrum * dw)
rand_phase = np.random.uniform(0.0, 2.0 * np.pi, size=N)
wave_angles = np.ones(N) * np.pi / 4.0

waveload = WaveLoad(
    wave_amps=wave_amps,
    freqs=freqs,
    eps=rand_phase,
    angles=wave_angles,
    config_file=vessel._config_file,
    interpolate=True,
    qtf_method="Newman",
    deep_water=True,
)

# ---------------- PID controller setup ------------
# Tuning gains (simple example – tune for your vessel!)
Kp = np.diag([1.5e3, 1.5e3, 0.0, 0.0, 0.0, 8.0e2])   # surge, sway, yaw
Ki = np.diag([5.0, 5.0, 0.0, 0.0, 0.0, 1.0])
Kd = np.diag([4.0e2, 4.0e2, 0.0, 0.0, 0.0, 2.0e2])

int_err = np.zeros(6)
eta_ref = np.zeros(6)   # Desired pose (at origin)

# ---------------- Data storage --------------------
eta_hist = np.zeros((6, len(t)))
nu_hist = np.zeros((6, len(t)))
tau_wave_hist = np.zeros((6, len(t)))
tau_control_hist = np.zeros((6, len(t)))

# ---------------- Main simulation loop ------------
start = time.time()
for i, ti in enumerate(t):
    eta = vessel.get_eta()
    nu = vessel.get_nu()

    err = eta_ref - eta
    if i == 0:
        int_err = err * dt
    else:
        int_err += (dt / 2.0) * (err + prev_err)
    prev_err = err
    der = -nu         # derivative of error since eta_ref is constant

    tau_control = (Kp @ err) + (Ki @ int_err) + (Kd @ der)
    # Zero-out heave, roll & pitch control channels (index 2,3,4)
    tau_control[[2, 3, 4]] = 0.0

    # Environmental loads
    tau_wave = waveload(ti, eta)

    # Total generalized forces / moments
    tau = tau_control + tau_wave

    # Integrate vessel state
    vessel.integrate(Uc=0.0, beta_c=0.0, tau=tau)

    # Store histories
    eta_hist[:, i] = eta
    nu_hist[:, i] = nu
    tau_wave_hist[:, i] = tau_wave
    tau_control_hist[:, i] = tau_control

elapsed = time.time() - start
print(f"Simulation completed in {elapsed:.2f} seconds (real‑time factor ≈ {simtime/elapsed:.2f})")

# ---------------- Plot results --------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, eta_hist[0, :], label="x")
ax[0].plot(t, eta_hist[1, :], label="y")
ax[0].set_ylabel("Position [m]")
ax[0].set_title("Surge & Sway")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(t, eta_hist[5, :], label="yaw (ψ)")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Yaw [rad]")
ax[1].set_title("Heading")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()
