import jax
import jax.numpy as jnp
import interpax
import matplotlib.pyplot as plt

# Define waypoints and times explicitly
t_waypoints = jnp.array([0, 10, 80, 150, 220, 290, 360, 400])
points = jnp.array([
    [2.0, 2.0, 0.0],
    [2.0, 2.0, 0.0],
    [4.0, 2.0, 0.0],
    [4.0, 4.0, 0.0],
    [4.0, 4.0, -jnp.pi/4],
    [2.0, 4.0, -jnp.pi/4],
    [2.0, 2.0, 0.0],
    [2.0, 2.0, 0.0],
])

# Initialize cubic spline interpolator
splines = [interpax.CubicSpline(t_waypoints, points[:, i], bc_type='clamped') for i in range(3)]

def reference(t):
    """
    Generate a smooth nonlinear reference trajectory for the boat using cubic spline interpolation.

    Args:
        t: Time (seconds)
        
    Returns:
        r: Desired position [x, y, φ]
        r_dot: Desired velocity [x_dot, y_dot, φ_dot]
        r_ddot: Desired acceleration [x_ddot, y_ddot, φ_ddot]
    """
    r = jnp.array([spl(t) for spl in splines])
    r_dot = jnp.array([spl(t, nu=1) for spl in splines])
    r_ddot = jnp.array([spl(t, nu=2) for spl in splines])

    return r, r_dot, r_ddot


# Plotting the trajectory
t_plot = jnp.linspace(0, 400, 1000)
r_plot = jnp.array([reference(t)[0] for t in t_plot])

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(t_plot, r_plot[:, 0], label='x')
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.title('X Position vs Time')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(t_plot, r_plot[:, 1], label='y')
plt.xlabel('Time [s]')
plt.ylabel('y [m]')
plt.title('Y Position vs Time')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(t_plot, r_plot[:, 2], label='φ')
plt.xlabel('Time [s]')
plt.ylabel('φ [rad]')
plt.title('Orientation vs Time')
plt.grid()

plt.tight_layout()
plt.show()
