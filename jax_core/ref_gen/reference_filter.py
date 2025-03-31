import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from jax_core.utils import rk4_step
# --- Provided RK4 integrator ---
# def rk4_step_impl(x, dt, f, *args):
#     k1 = f(x, *args)
#     k2 = f(x + 0.5 * dt * k1, *args)
#     k3 = f(x + 0.5 * dt * k2, *args)
#     k4 = f(x + dt * k3, *args)
#     return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# rk4_step = jax.jit(rk4_step_impl, static_argnums=(2,))

# --- Filter matrices and dynamics ---
def build_filter_matrices(dt, omega=jnp.array([0.2, 0.15, 0.2])):
    """
    Build the system matrices (Ad, Bd) for the third-order filter.
    The state is defined as x = [η, η_dot, η_ddot].
    """
    delta = jnp.eye(3)
    w = jnp.diag(omega)
    O3 = jnp.zeros((3, 3))
    Ad = jnp.block([
        [O3,           jnp.eye(3), O3],
        [O3,           O3,         jnp.eye(3)],
        [-w**3, -(2*delta + jnp.eye(3)) @ (w**2), -(2*delta + jnp.eye(3)) @ w]
    ])
    Bd = jnp.block([
        [O3],
        [O3],
        [w**3]
    ])
    return Ad, Bd

def make_filter_dynamics(Ad, Bd):
    """
    Returns a function computing the filter dynamics:
      f(x, η_r) = Ad @ x + Bd @ η_r
    """
    def dynamics(x, eta_r):
        return Ad @ x + Bd @ eta_r
    return dynamics

# --- RK4-based simulation ---
@jax.jit
def simulate_filter_rk4(x0, eta_r_traj, dt, Ad, Bd):
    """
    Simulate the third-order reference filter using RK4 integration.

    Args:
        x0: initial state vector.
        eta_r_traj: reference trajectory; shape [steps, 3]
        dt: time step.
        Ad, Bd: system matrices.

    Returns:
        final state and outputs tuple: (η_history, η_dot_history, η_ddot_history)
    """
    filter_dynamics = make_filter_dynamics(Ad, Bd)
    
    def step_fn(x, eta_r):
        new_x = rk4_step(x, dt, filter_dynamics, eta_r)
        # Record position, velocity, and acceleration.
        output = (new_x[:3], new_x[3:6], new_x[6:])
        return new_x, output

    final_state, outputs = jax.lax.scan(step_fn, x0, eta_r_traj)
    return final_state, outputs

# --- Simulation setup ---
dt = 0.01
T_sim = 400.0  # total simulation time in seconds
steps = int(T_sim / dt)

# Define the set points for the trajectory.
points = jnp.array([
    [2.0, 2.0, 0.0],
    [4.0, 2.0, 0.0],
    [4.0, 4.0, 0.0],
    [4.0, 4.0, -jnp.pi/4],
    [2.0, 4.0, -jnp.pi/4],
    [2.0, 2.0, 0.0]
])

# Create a piecewise constant reference trajectory.
num_points = points.shape[0]
seg_steps = steps // num_points
eta_r_traj = jnp.repeat(points, seg_steps, axis=0)
# Pad with the final set point if necessary.
if eta_r_traj.shape[0] < steps:
    pad = jnp.tile(points[-1][None, :], (steps - eta_r_traj.shape[0], 1))
    eta_r_traj = jnp.concatenate([eta_r_traj, pad], axis=0)

# Build filter system matrices.
Ad, Bd = build_filter_matrices(dt)
# Set the initial state at the first set point (with zero velocity and acceleration).
x0 = jnp.concatenate([points[0], jnp.zeros(6)])

# Run the simulation using the RK4 integrator.
final_state, outputs = simulate_filter_rk4(x0, eta_r_traj, dt, Ad, Bd)
eta_d_hist, eta_d_dot_hist, eta_d_ddot_hist = outputs

# --- Plotting ---
time = jnp.linspace(0, T_sim, steps)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, eta_d_hist[:, 0], label='x')
plt.plot(time, eta_d_hist[:, 1], label='y')
plt.plot(time, eta_d_hist[:, 2], label='ψ')
plt.title('Filtered Position (η)')
plt.ylabel('Position [m, rad]')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, eta_d_dot_hist[:, 0], label='x_dot')
plt.plot(time, eta_d_dot_hist[:, 1], label='y_dot')
plt.plot(time, eta_d_dot_hist[:, 2], label='ψ_dot')
plt.title('Filtered Velocity (η_dot)')
plt.ylabel('Velocity [m/s, rad/s]')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, eta_d_ddot_hist[:, 0], label='x_ddot')
plt.plot(time, eta_d_ddot_hist[:, 1], label='y_ddot')
plt.plot(time, eta_d_ddot_hist[:, 2], label='ψ_ddot')
plt.title('Filtered Acceleration (η_ddot)')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s², rad/s²]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
