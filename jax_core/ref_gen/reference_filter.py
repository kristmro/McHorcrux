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
def build_filter_matrices(dt, omega=jnp.array([0.8, 0.35, 0.5])):
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

if __name__ == "__main__":
    # --- Simulation setup ---
    dt = 0.01
    # Define the set points for the trajectory.
    points = jnp.array([
        [2.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 4.0, 0.0],
        [4.0, 4.0, -jnp.pi/4],
        [2.0, 4.0, -jnp.pi/4],
        [2.0, 2.0, 0.0]
    ])

    # --- Define segments with specific durations and types ---
    segments = [
        # 1. [2.0,2.0,0.0] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([2.0, 2.0, 0.0]), 'time': 5.0},
        # 2. [2.0,2.0,0.0] to [4.0,2.0,0.0] for 30 seconds.
        {'type': 'transition', 'start': jnp.array([2.0, 2.0, 0.0]), 'end': jnp.array([4.0, 2.0, 0.0]), 'time': 30.0},
        # 3. [4.0,2.0,0.0] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([4.0, 2.0, 0.0]), 'time': 15.0},
        # 4. [4.0,2.0,0.0] to [4.0,4.0,0.0] for 40 seconds.
        {'type': 'transition', 'start': jnp.array([4.0, 2.0, 0.0]), 'end': jnp.array([4.0, 4.0, 0.0]), 'time': 40.0},
        # 5. [4.0,4.0,0.0] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([4.0, 4.0, 0.0]), 'time': 23.0},
        # 6. [4.0,4.0,0.0] to [4.0,4.0,-jnp.pi/4] for 15 seconds.
        {'type': 'transition', 'start': jnp.array([4.0, 4.0, 0.0]), 'end': jnp.array([4.0, 4.0, -jnp.pi/4]), 'time': 15.0},
        # 7. [4.0,4.0,-jnp.pi/4] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([4.0, 4.0, -jnp.pi/4]), 'time': 17.0},
        # 8. [4.0,4.0,-jnp.pi/4] to [2.0,4.0,-jnp.pi/4] for 40 seconds.
        {'type': 'transition', 'start': jnp.array([4.0, 4.0, -jnp.pi/4]), 'end': jnp.array([2.0, 4.0, -jnp.pi/4]), 'time': 40.0},
        # 9. [2.0,4.0,-jnp.pi/4] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([2.0, 4.0, -jnp.pi/4]), 'time': 15.0},
        # 10. [2.0,4.0,-jnp.pi/4] to [2.0,2.0,0.0] for 50 seconds.
        {'type': 'transition', 'start': jnp.array([2.0, 4.0, -jnp.pi/4]), 'end': jnp.array([2.0, 2.0, 0.0]), 'time': 70.0},
        # 11. [2.0,2.0,0.0] for 5 seconds.
        {'type': 'dwell', 'point': jnp.array([2.0, 2.0, 0.0]), 'time': 20.0},
    ]

    # --- Generate the reference trajectory based on these segments ---
    traj_segments = []
    for seg in segments:
        seg_steps = int(seg['time'] / dt)
        if seg['type'] == 'dwell':
            # Repeat the point for the dwell duration.
            traj_segments.append(jnp.tile(seg['point'][None, :], (seg_steps, 1)))
        elif seg['type'] == 'transition':
            # Linearly interpolate from start to end.
            t = jnp.linspace(0, 1, seg_steps)
            transition = (1 - t[:, None]) * seg['start'] + t[:, None] * seg['end']
            traj_segments.append(transition)

    # Concatenate all segments.
    eta_r_traj = jnp.concatenate(traj_segments, axis=0)

    # --- Update simulation parameters based on trajectory length ---
    steps = eta_r_traj.shape[0]
    T_sim = steps * dt
    print(f"Total simulation time: {T_sim} seconds")
    time = jnp.linspace(0, T_sim, steps)


    # Build filter system matrices.
    Ad, Bd = build_filter_matrices(dt)
    # Set the initial state at the first set point (with zero velocity and acceleration).
    x0 = jnp.concatenate([points[0], jnp.zeros(6)])

    # Run the simulation using the RK4 integrator.
    final_state, outputs = simulate_filter_rk4(x0, eta_r_traj, dt, Ad, Bd)
    eta_d_hist, eta_d_dot_hist, eta_d_ddot_hist = outputs


    plt.figure(figsize=(15, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time, eta_d_hist[:, 0], label='x')
    plt.plot(time, eta_d_hist[:, 1], label='y')
    plt.plot(time, eta_d_hist[:, 2], label='ψ')
    plt.title('Filtered Position (η)')
    plt.ylabel('Position [m, rad]')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(time, eta_d_dot_hist[:, 0], label='x_dot')
    plt.plot(time, eta_d_dot_hist[:, 1], label='y_dot')
    plt.plot(time, eta_d_dot_hist[:, 2], label='ψ_dot')
    plt.title('Filtered Velocity (η_dot)')
    plt.ylabel('Velocity [m/s, rad/s]')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(time, eta_d_ddot_hist[:, 0], label='x_ddot')
    plt.plot(time, eta_d_ddot_hist[:, 1], label='y_ddot')
    plt.plot(time, eta_d_ddot_hist[:, 2], label='ψ_ddot')
    plt.title('Filtered Acceleration (η_ddot)')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s², rad/s²]')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(eta_d_hist[:,1], eta_d_hist[:, 0], label='Trajectory')
    plt.title('Filtered Trajectory in XY Plane')
    plt.xlabel('Y Position [m]')
    plt.ylabel('X Position [m]')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
