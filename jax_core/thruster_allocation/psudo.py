import jax
import jax.numpy as jnp
from functools import partial

# Default thruster configuration constants
DEFAULT_POS_X = jnp.array([1.0678, 0.9344, 0.9344, -1.1644, -0.9911, -0.9911])
DEFAULT_POS_Y = jnp.array([0.0, 0.11, -0.11, 0.0, -0.1644, 0.1644])
DEFAULT_K = jnp.array([1.491, 1.491, 1.491, 1.491, 1.491, 1.491])
DEFAULT_THRUST_MAX = 2.0
DEFAULT_THRUST_MIN = -0.85
DEFAULT_DT = 0.01
DEFAULT_N_DOT_MAX = 3.0
DEFAULT_ALPHA_DOT_MAX = 2.0

def create_thruster_config(pos_x=DEFAULT_POS_X, pos_y=DEFAULT_POS_Y, K=DEFAULT_K):
    """
    Creates a pre-computed thruster configuration for efficient allocation.
    
    Returns a dictionary containing pre-computed matrices for allocation.
    """
    n_thrusters = pos_x.shape[0]
    DOFS = 3
    n_problem = 2 * n_thrusters
    
    # Construct allocation matrix
    T_e = jnp.zeros((DOFS, n_problem))
    indices_x = jnp.arange(0, n_problem, 2)
    indices_y = jnp.arange(1, n_problem, 2)
    
    T_e = T_e.at[0, indices_x].set(jnp.ones_like(indices_x, dtype=T_e.dtype))
    T_e = T_e.at[1, indices_y].set(jnp.ones_like(indices_y, dtype=T_e.dtype))
    T_e = T_e.at[2, indices_x].set(-pos_y)
    T_e = T_e.at[2, indices_y].set(pos_x)
    
    # Pre-compute pseudo-inverse
    T_e_pinv = jnp.linalg.pinv(T_e)
    
    # Build the gain vector
    K_vec = jnp.repeat(K, 2)
    
    return {
        'T_e': T_e,
        'T_e_pinv': T_e_pinv,
        'K_vec': K_vec,
        'pos_x': pos_x,
        'pos_y': pos_y,
        'K': K,
        'n_thrusters': n_thrusters
    }

# Create a JIT-compiled allocation function that uses pre-computed config
@partial(jax.jit, static_argnums=(2, 3, 4))  # Added n_thrusters as static
def allocate_with_config(tau_d, config, thrust_max, thruster_min, n_thrusters=6):
    """
    Allocates thruster commands using pre-computed configuration for better performance.
    
    Parameters:
      tau_d: jnp.ndarray of shape (3,), desired [F_x, F_y, M]
      config: dict, pre-computed thruster configuration from create_thruster_config()
      thrust_max: scalar, maximum thrust (forward limit) [N]
      thruster_min: scalar, minimum thrust (reverse limit) [N]
      n_thrusters: int or None, number of thrusters (static value for JIT)
    
    Returns:
      u_sat: jnp.ndarray of shape (n_thrusters,), saturated thrust magnitudes
      alpha: jnp.ndarray of shape (n_thrusters,), corresponding azimuth angles [rad]
    """
    # Use static n_thrusters if provided, otherwise try to get from config
    if n_thrusters is None:
        n_thrusters = config['n_thrusters']
        
    # Solve for the extended thruster command and "normalize" by gain
    u_e = (config['T_e_pinv'] @ tau_d) / config['K_vec']
    
    # Reshape and convert to polar coordinates using static shape value
    u_e = u_e.reshape(n_thrusters, 2)
    u = jnp.linalg.norm(u_e, axis=1)
    alpha = jnp.arctan2(u_e[:, 1], u_e[:, 0])
    
    # Saturate the thrust magnitudes
    u_sat = jnp.clip(u, 0, thrust_max)
    
    return u_sat, alpha

@jax.jit
def saturate_rate(u, alpha, u_prev, alpha_prev, dt, n_dot_max, alpha_dot_max):
    """
    Saturates the rate of change for thruster commands.
    
    Parameters:
      u: jnp.ndarray, thrust magnitudes
      alpha: jnp.ndarray, steering angles [rad]
      u_prev: jnp.ndarray, previous thrust magnitudes
      alpha_prev: jnp.ndarray, previous steering angles [rad]
      dt: scalar, time step [s]
      n_dot_max: scalar, maximum change in thrust magnitude [N/s]
      alpha_dot_max: scalar, maximum change in steering angle [rad/s]
    
    Returns:
      u_rate_sat: jnp.ndarray, rate-limited thrust magnitudes
      alpha_rate_sat: jnp.ndarray, rate-limited steering angles [rad]
    """
    # Compute maximum allowable change per time step
    u_rate_limit = n_dot_max * dt
    alpha_rate_limit = alpha_dot_max * dt
    
    # Apply rate saturation in a single operation
    u_rate_sat = jnp.clip(u, u_prev - u_rate_limit, u_prev + u_rate_limit)
    
    # Handle angle wrapping for rate saturation
    # Calculate the change in angle, taking into account the circular nature of angles
    alpha_diff = (alpha - alpha_prev + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Apply rate limit to the change
    alpha_diff_sat = jnp.clip(alpha_diff, -alpha_rate_limit, alpha_rate_limit)
    
    # Apply the limited change to the previous angle
    alpha_rate_sat = (alpha_prev + alpha_diff_sat + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    return u_rate_sat, alpha_rate_sat

@jax.jit
def map_to_3dof(u_sat, alpha, config):
    """
    Maps thruster commands back to the effective 3-DOF global force/moment.
    
    Parameters:
      u_sat: jnp.ndarray, thrust magnitudes
      alpha: jnp.ndarray, steering angles [rad]
      config: dict, pre-computed thruster configuration from create_thruster_config()
    
    Returns:
      tau_eff: jnp.ndarray of shape (3,), effective global force/moment [F_x, F_y, M]
    """
    # Calculate thruster components (x,y) for each thruster
    u_x = u_sat * jnp.cos(alpha)
    u_y = u_sat * jnp.sin(alpha)
    
    # Weight by thruster gains
    u_x_weighted = u_x * config['K']
    u_y_weighted = u_y * config['K']
    
    # Compute force components
    F_x = jnp.sum(u_x_weighted)
    F_y = jnp.sum(u_y_weighted)
    
    # Compute moment: cross product of position and force
    M = jnp.sum(config['pos_x'] * u_y_weighted - config['pos_y'] * u_x_weighted)
    
    tau_eff = jnp.array([F_x, F_y, M])
    return tau_eff

# Original functions maintained for backward compatibility, but using optimized versions
@partial(jax.jit, static_argnums=(4, 5))
def allocate_jax(tau_d, pos_x, pos_y, K, thrust_max, thruster_min):
    """
    Legacy function maintained for backward compatibility.
    Uses the optimized implementation internally.
    """
    config = create_thruster_config(pos_x, pos_y, K)
    return allocate_with_config(tau_d, config, thrust_max, thruster_min)

# Default parameter wrapper functions
def get_default_config():
    """Returns a pre-computed configuration using default parameters."""
    return create_thruster_config(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_K)

def allocate_default(tau_d, thrust_max=DEFAULT_THRUST_MAX, thruster_min=DEFAULT_THRUST_MIN):
    """Allocates thruster commands using default parameters."""
    config = get_default_config()
    return allocate_with_config(tau_d, config, thrust_max, thruster_min)

def saturate_rate_default(u, alpha, u_prev, alpha_prev):
    """Saturates rates using default parameters."""
    return saturate_rate(u, alpha, u_prev, alpha_prev, DEFAULT_DT, 
                         DEFAULT_N_DOT_MAX, DEFAULT_ALPHA_DOT_MAX)

def map_to_3dof_default(u_sat, alpha):
    """Maps commands to 3DOF forces/moments using default parameters."""
    config = get_default_config()
    return map_to_3dof(u_sat, alpha, config)