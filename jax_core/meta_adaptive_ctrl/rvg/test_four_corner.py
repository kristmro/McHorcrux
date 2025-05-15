"""
This script simulates the four-corner test for a 6-DOF system using JAX.

author: Kristian Magnus Roen

"""

import pickle
import os
import argparse
import time
import numpy as np
from functools import partial
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--use_x64', help='use 64-bit precision',
                    action='store_true')
args = parser.parse_args()

# Set precision
if args.use_x64:
    os.environ['JAX_ENABLE_X64'] = 'True'

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402
from jax.experimental.ode import odeint         # noqa: E402
from jax_core.utils import params_to_posdef      # noqa: E402
from jax_core.meta_adaptive_ctrl.rvg.dynamics import prior_3dof_nom as prior_3dof, plant_6 as plant, disturbance  # noqa: E402
from jax_core.simulator.waves.wave_load_jax_jit import wave_load  # noqa: E402
from jax_core.thruster_allocation.psudo import (
    create_thruster_config, allocate_with_config, saturate_rate, map_to_3dof,
    get_default_config, DEFAULT_THRUST_MAX, DEFAULT_THRUST_MIN, DEFAULT_DT,
    DEFAULT_N_DOT_MAX, DEFAULT_ALPHA_DOT_MAX
) 
from jax_core.utils import mat_to_svec_dim

def diag_chol_indices(n):
    """
    Return the indices in the length‐d Cholesky‐parameter vector
    corresponding to the diagonal of an n×n L.

    For n=3, this returns [0,2,5], since the param order is
        [L00, L10, L11, L20, L21, L22].
    """
    return jnp.array([i * (i+1)//2 + i for i in range(n)], dtype=int)

def vec_to_posdef_diag_cholesky(v):
    """
    Build a *purely diagonal* PD matrix X from an unconstrained vector v∈ℝⁿ
    by embedding it into the Cholesky‐parametrization and reusing params_to_posdef.

    Internally:
      • full = zeros(d) with d = n(n+1)/2
      • full[idxs] = v/2    # see note below
      • L = params_to_cholesky(full)  # exponentiates diag(full)
      • X = L @ L.T

    Because full[i*(i+1)/2 + i] = v_i/2, L_ii = exp(v_i/2), so X_ii = exp(v_i).

    Off‐diagonal slots of full remain zero ⇒ L_ij=0 for i≠j ⇒ X is exactly diagonal.
    """
    v = jnp.atleast_1d(v)
    n = v.shape[-1]
    d = mat_to_svec_dim(n)                # = n(n+1)/2
    idxs = diag_chol_indices(n)           # which param‑vector slots are diag
    full = jnp.zeros(v.shape[:-1] + (d,), dtype=v.dtype)
    # scatter half‐logs so that X_ii = exp(v_i)
    full = full.at[..., idxs].set(v/2)
    # now call your existing reparam → PD
    return params_to_posdef(full)


# Import the reference filter functions.
from jax_core.ref_gen.reference_filter import build_filter_matrices, simulate_filter_rk4


#-----------------------------------------------------------------
# Updated reference trajectory: Four-Corner Test (MC-GYM style)
#-----------------------------------------------------------------

# --- dt ---
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
    {'type': 'transition', 'start': jnp.array([2.0, 4.0, -jnp.pi/4]), 'end': jnp.array([2.0, 2.0, 0.0]), 'time': 50.0},
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


# Build filter system matrices.
Ad, Bd = build_filter_matrices(dt)
# Set the initial state at the first set point (with zero velocity and acceleration).
x0 = jnp.concatenate([points[0], jnp.zeros(6)])

# Run the simulation using the RK4 integrator.
final_state, outputs = simulate_filter_rk4(x0, eta_r_traj, dt, Ad, Bd)
eta_d_hist, eta_d_dot_hist, eta_d_ddot_hist = outputs


def ref(t):
    """
    Four corner test trajectory using a reference filter.
    
    Given time t (in seconds), returns a tuple:
      (r, dr, ddr)
    where r is the filtered position, dr is the filtered velocity, and 
    ddr is the filtered acceleration.
    """
    # Compute the corresponding index based on the time step.
    index = jnp.minimum(jnp.floor_divide(t, dt).astype(jnp.int32), eta_d_hist.shape[0] - 1)
    r = eta_d_hist[index]
    dr = eta_d_dot_hist[index]
    ddr = eta_d_ddot_hist[index]
    return r, dr, ddr

# The remainder of the simulation code remains unchanged.
if __name__ == "__main__":
    print('Testing ... ', flush=True)
    start = time.time()
    seed, M, ctrl_pen, act, test_act = 0, 2, 6, 'off', 'off'
    integral_abs_limit_val = 1e+4  # Configurable integral clamp limit

    # Sampled-time simulator
    @partial(jax.jit, static_argnums=(3,))
    def simulate(ts, w, params, ref=ref,
                 plant=plant, prior=prior_3dof, disturbance=wave_load):
        """TODO: docstring."""
        thruster_config = get_default_config()

        # Adaptation law
        def adaptation_law(q, dq, r, dr, params=params):
            # Regressor features
            y = jnp.concatenate((q, dq))
            for W, b in zip(params['W'], params['b']):
                y = jnp.tanh(W @ y + b)

            # Auxiliary signals
            Λ, P = params['Λ'], params['P']
            e, de = q - r, dq - dr
            s = de + Λ @ e

            dA = P @ jnp.outer(s, y)
            return dA, y

        # Controller
        def controller(q, dq, r, dr, ddr, f_hat, params=params):
            # Auxiliary signals
            Λ, K = params['Λ'], params['K']
            e, de = q - r, dq - dr
            s = de + Λ @ e
            v, dv = dr - Λ @ e, ddr - Λ @ de

            # Control input and adaptation law
            def sat(s):
                """Saturation function."""
                phi = 20
                return jnp.where(jnp.abs(s/phi) > 1, jnp.sign(s), s/phi)
            M_mat, D, G, R = prior(q, dq)
            τ = M_mat @ dv + D @ v + G @ e - f_hat - K @ sat(s)
            u = jnp.linalg.solve(R, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on the controller
        def ode(x, t, u, w=w):
            q, dq = x
            f_ext = disturbance(t, q, w)
            dq,ddq = plant(q, dq, u, f_ext)
            dx = (dq, ddq)
            return dx
        
        # Simulation loop
        def loop(carry, input_slice):
            t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev, alpha_prev, u_f_prev = carry
            t = input_slice
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref(t)

            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            A = A_prev + (t - t_prev) * (dA_prev + dA) / 2

            # Compute force estimate and control input
            f_hat = A @ y
            u, τ = controller(q, dq, r, dr, ddr, f_hat)

            # Thrust saturation
            u_sat, alpha = allocate_with_config(
                u, 
                thruster_config, 
                DEFAULT_THRUST_MAX, 
                DEFAULT_THRUST_MIN
            )
            
            u_rate_sat, alpha_rate_sat = saturate_rate(
                u_sat, alpha, u_f_prev, alpha_prev, 
                DEFAULT_DT, DEFAULT_N_DOT_MAX, DEFAULT_ALPHA_DOT_MAX
            )
            
            u_aft = map_to_3dof(u_rate_sat, alpha_rate_sat, thruster_config)
                
            carry = (t, q, dq, u, A, dA, alpha, u_rate_sat)
            output_slice = (q, dq, u, τ, r, dr)
            return carry, (output_slice, f_hat)

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref(t0)
        q0, dq0 = r0, dr0
        dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
        A0 = jnp.zeros((q0.size, y0.size))
        f0 = A0 @ y0
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0, f0)
        alpha0 = jnp.zeros(6)
        u_f0 = jnp.zeros(6)
        # Run simulation loop
        carry = (t0, q0, dq0, u0, A0, dA0, alpha0, u_f0)
        carry, (output, f_hat) = jax.lax.scan(loop, carry, ts[1:])
        q, dq, u, τ, r, dr = output

        # Prepend initial conditions
        q = jnp.vstack((q0, q))
        dq = jnp.vstack((dq0, dq))
        u = jnp.vstack((u0, u))
        τ = jnp.vstack((τ0, τ))
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))
        f_hat = jnp.vstack((f0, f_hat))

        return q, dq, u, τ, r, dr, f_hat
    
    # ---------------------------------------------------------------------#
    #  Pure PID simulator                                                  #
    # ---------------------------------------------------------------------#
    @partial(jax.jit, static_argnums=(3, 4, 5)) # num_dof_param (idx 3), current_integral_abs_limit (idx 4), ref_fn (idx 5)
    def simulate_pid(ts, w, params, num_dof_param, current_integral_abs_limit, ref_fn=ref,
                     plant_fn=plant, disturbance_fn=wave_load, prior_fn=prior_3dof):
        """Simulate closed-loop system with a vector PID controller."""

        Kp_mat = jnp.diag(params['Kp'])
        Ki_mat = jnp.diag(params['Ki'])
        Kd_mat = jnp.diag(params['Kd'])

        # Define limits for the integral term I (for anti-windup via clamping)
        I_min_limits = jnp.full((num_dof_param,), -current_integral_abs_limit)
        I_max_limits = jnp.full((num_dof_param,), current_integral_abs_limit)

        # ---- plant ODE w/ ZOH ----------------------------------------------
        def ode(state, t, u):
            q, dq = state
            f_ext = disturbance_fn(t, q, w)
            dq, ddq = plant_fn(q, dq, u, f_ext)
            return (dq, ddq)

        # ---- scan step ------------------------------------------------------
        def step(carry, t):
            t_prev, q_prev, dq_prev, u_prev, I_prev = carry
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
            q, dq = qs[-1], dqs[-1]

            r, dr, _ = ref_fn(t)
            e, de = q - r, dq - dr
            dt = t - t_prev
            # Using the trapezoidal rule
            I = I_prev + 0.5 * (e + (q_prev - r)) * dt
            I = jnp.clip(I, I_min_limits, I_max_limits)

            M_mat, D, G, R = prior_fn(q, dq)
            
            τ = -(Kp_mat @ e + Ki_mat @ I + Kd_mat @ de)
            u = jnp.linalg.solve(R, τ)  

            new_carry = (t, q, dq, u, I)
            out_slice = (q, dq, u, τ, r, dr)
            return new_carry, out_slice

        # ---- initial conditions --------------------------------------------
        t0 = ts[0]
        r0, dr0, _ = ref_fn(t0)
        q0, dq0 = r0, dr0
        I0 = jnp.zeros_like(q0)
        u0 = jnp.zeros_like(q0)

        carry, outputs = jax.lax.scan(step, (t0, q0, dq0, u0, I0), ts[1:])
        q, dq, u, τ, r, dr = outputs

        q  = jnp.vstack((q0, q))
        dq = jnp.vstack((dq0, dq))
        u  = jnp.vstack((u0, u))
        τ  = jnp.vstack((u0, τ))   # τ==u at t0
        r  = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))
        return q, dq, u, τ, r, dr

    # Choose wave parameters, fixed control gains, and simulation times
    num_dof = 3
    key = jax.random.PRNGKey(seed)
    w = disturbance(jnp.array((3.0, 15.0, 0)), key)
    λ, k, p = 10.0, 50.0, 50.0
    T, dt = T_sim, dt
    ts = jnp.arange(0, T + dt, dt)


    # Simulate tracking for each method
    test_results = {
        'w': w,
        'gains': (λ, k, p),
    }

    # Our method with meta-learned gains
    print('meta trained adaptive ctrl ...', flush=True)
    filename = os.path.join('data', 'training_results','rvg','model_uncertainty','sat','act_{}'.format(act), 'ctrl_pen_{}'.format(ctrl_pen),'seed={}_M={}.pkl'.format(seed, M))
    with open(filename, 'rb') as file:
        train_results = pickle.load(file)
    
    if train_results['controller']['Λ'].shape[-1] == 2*num_dof: 
            params = {
                'W': train_results['model']['W'],
                'b': train_results['model']['b'],
                'Λ': params_to_posdef(train_results['controller']['Λ']),
                'K': params_to_posdef(train_results['controller']['K']),
                'P': params_to_posdef(train_results['controller']['P']),
            }
    else:
        # If the controller parameters are not in the correct shape, reshape them
        params = {
            'W': train_results['model']['W'],
            'b': train_results['model']['b'],
            'Λ': vec_to_posdef_diag_cholesky(train_results['controller']['Λ']),
            'K': vec_to_posdef_diag_cholesky(train_results['controller']['K']),
            'P': vec_to_posdef_diag_cholesky(train_results['controller']['P']),
        }
    
    print('The meta adaptive controller parameters are \n', params['Λ'], '\n', params['K'], '\n',params['P'], flush=True)
    print('The model parameters are \n', params['W'], '\n', params['b'], flush=True)
    q, dq, u, τ, r, dr, f_hat = simulate(ts, w, params, ref)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    test_results['meta_adap_ctrl'] = {
        'params': params,
        't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
        'u': u, 'τ': τ, 'e': e,
    }


    f_ext_hist = jax.vmap(lambda ti, qi: wave_load(ti, qi, w))(ts, q)   # (n_steps, 6)
    f_ext_hist_np = np.asarray(f_ext_hist).T                            # (DOF, n_steps)

    import matplotlib.pyplot as plt
    import numpy as np
    # ----------------------------------------------------------------------------
    # Plot the wave loads vs time for each degree of freedom (DOF)
    # ----------------------------------------------------------------------------
    # Note: wave_loads is an array with shape (n_steps, 6) if tau_wave is a 6-element vector.
    wave_loads_np = np.array(f_hat).T  # Transpose to shape (DOF, n_steps) to match t_array shape
    t_array = np.asarray(ts)  # Time array, shape: (n_steps,)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    dof_labels = ['Surge', 'Sway', 'Yaw']

    for i in range(3):
        axes[i].plot(t_array, f_ext_hist_np[i], 'k--', label='f_ext (true)')
        axes[i].plot(t_array, wave_loads_np[i],  'r',   label='f_hat')
        axes[i].legend()
        axes[i].grid()

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Wave Loads vs Time for Each DOF', y=0.95)  # Adjusted y position
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    

    for method in ('pid', 'adaptive_ctrl'):
        if method == 'pid':
            print('PID Ctrl...', flush=True)
            params_pid = {
                'Kp': jnp.array([7.642e+04, 6.0000e+05, 9.9626e+06]),
                'Ki': jnp.array([2.1456e+00, 2.4660e+00, 9.9990e+01]),
                'Kd': jnp.array([1.0500e+05, 1.000e+04, 3.0931e+02]),
            }
            integral_abs_limit_val = 10   
            q, dq, u, τ, r, dr = simulate_pid(ts, w, params_pid, num_dof, integral_abs_limit_val)
            e = np.concatenate((q - r, dq - dr), axis=-1)
            test_results[method] = {
                'params': params_pid,
                't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
                'u': u, 'τ': τ, 'e': e,
            }

        else:
            with open(filename, 'rb') as file:
                train_results = pickle.load(file)
            params = {
                'W': train_results['model']['W'],
                'b': train_results['model']['b'],
            }
            print('Adaptive ctrl self tuned...', flush=True)
            params['Λ'] = λ * jnp.eye(num_dof)
            params['K'] = k * jnp.eye(num_dof)
            params['P'] = p * jnp.eye(num_dof)
            q, dq, u, τ, r, dr, f_hat = simulate(ts, w, params, ref)
            e = np.concatenate((q - r, dq - dr), axis=-1)
            test_results[method] = {
                'params': params,
                't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
                'u': u, 'τ': τ, 'e': e,
            }

    # Save the test results.
    output_path = os.path.join('data', 'testing_results','rvg','model_uncertainty','train_act_{}'.format(act),'four_corner','test_act_{}'.format(test_act),'ctrl_pen_{}'.format(ctrl_pen),'seed={}_M={}.pkl'.format(seed, M))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save
    with open(output_path, 'wb') as file:
        pickle.dump(test_results, file)

    end = time.time()
    print('done! ({:.2f} s)'.format(end - start))



    # ----------------------------------------------------------------------------
    # Plot the wave loads vs time for each degree of freedom (DOF)
    # ----------------------------------------------------------------------------
    # Note: wave_loads is an array with shape (n_steps, 6) if tau_wave is a 6-element vector.
    wave_loads_np = np.array(f_hat).T  # Transpose to shape (DOF, n_steps) to match t_array shape
    t_array = np.asarray(ts)  # Time array, shape: (n_steps,)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    dof_labels = ['Surge', 'Sway', 'Yaw']

    for i in range(3):
        axes[i].plot(t_array, f_ext_hist_np[i], 'k--', label='f_ext (true)')
        axes[i].plot(t_array, wave_loads_np[i, :], label=f'{dof_labels[i]}')
        axes[i].set_ylabel('Wave load')
        axes[i].legend()
        axes[i].grid()

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Wave Loads vs Time for Each DOF', y=0.95)  # Adjusted y position
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    #--------------------------------------------------------------------
    # Plotting
    #--------------------------------------------------------------------
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the test results
    which_test = 'four_corner'
    print("Loading test results...")
    with open('data/testing_results/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}.pkl'.format(act,which_test,test_act,ctrl_pen,seed,M), 'rb') as file:
        results = pickle.load(file)

    # Create figures directory if it doesn't exist
    os.makedirs('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}'.format(act,which_test,test_act,ctrl_pen, seed, M), exist_ok=True)

    # Check what keys are actually available in the results
    print("Available methods:", [key for key in results.keys() if key not in ['w', 'gains']])

    # Define methods based on what's available
    available_methods = [key for key in results.keys() if key not in ['w', 'gains']]
    if not available_methods:
        print("Error: No method results found in the data file!")
        exit(1)

    # Define styling based on available methods
    if 'ours_meta' in available_methods:
        if len(available_methods) > 1:
            methods = available_methods
            labels = ['Meta-learned' if m == 'ours_meta' else m.replace('_', ' ').title() for m in methods]
        else:
            methods = ['ours_meta']
            labels = ['Meta-learned']
    else:
        methods = available_methods
        labels = [m.replace('_', ' ').title() for m in methods]

    # Generate enough distinct colors
    colors = ['b', 'r', 'g', 'm', 'c', 'y'] + ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(max(0, len(methods)-6))]
    coord_labels = ['x', 'y', 'φ']

    print(f"Plotting data for methods: {methods}")

    # Get time data
    t = results[methods[0]]['t']

    # Figure 1: Position tracking
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle('Position Tracking Performance', fontsize=16)

    for i in range(3):
        ax = axes1[i]
        # Plot reference
        ax.plot(t, results[methods[0]]['r'][:, i], 'k--', label='Reference')
        
        # Plot each method
        for j, method in enumerate(methods):
            ax.plot(t, results[method]['q'][:, i], colors[j], label=labels[j])
        
        ax.set_ylabel(f'{coord_labels[i]} position')
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes1[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/position_tracking.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    # Continue with the rest of your plotting code using the dynamically determined methods
    # Figure 2: 2D trajectory plot
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(results[methods[0]]['r'][:, 0], results[methods[0]]['r'][:, 1], 'k--', label='Reference')

    for j, method in enumerate(methods):
        ax2.plot(results[method]['q'][:, 0], results[method]['q'][:, 1], colors[j], label=labels[j])

    ax2.set_xlabel('x position (m)')
    ax2.set_ylabel('y position (m)')
    ax2.set_title('2D Trajectory')
    ax2.grid(True)
    ax2.legend()
    ax2.set_aspect('equal')
    plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/trajectory_2d.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    # Rest of your plotting code with the dynamic methods list...
    # Figure 3: Tracking errors
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig3.suptitle('Tracking Errors', fontsize=16)

    for i in range(3):
        ax = axes3[i]
        
        for j, method in enumerate(methods):
            # Extract position error (first 3 components of the error vector)
            ax.plot(t, results[method]['e'][:, i], colors[j], label=labels[j])
        
        ax.set_ylabel(f'{coord_labels[i]} error')
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes3[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/tracking_errors.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    # Figure 4: Control efforts
    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig4.suptitle('Control Efforts cmd', fontsize=16)

    for i in range(3):
        ax = axes4[i]
        
        for j, method in enumerate(methods):
            ax.plot(t, results[method]['τ'][:, i], colors[j], label=labels[j])
        
        ax.set_ylabel(f'τ_{coord_labels[i]}')
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes4[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_cmd.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    # Figure 4: Control efforts
    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig4.suptitle('Control Efforts body after saturation', fontsize=16)

    for i in range(3):
        ax = axes4[i]
        
        for j, method in enumerate(methods):
            ax.plot(t, results[method]['u'][:, i], colors[j], label=labels[j])
        
        ax.set_ylabel(f'u_{coord_labels[i]}')
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes4[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_u_after.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    # Figure 5: RMS error comparison
    if len(methods) > 1:  # Only make comparison if we have multiple methods
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        bar_width = 0.25
        index = np.arange(3)

        for j, method in enumerate(methods):
            rms_errors = [np.sqrt(np.mean(results[method]['e'][:, i]**2)) for i in range(3)]
            ax5.bar(index + j*bar_width, rms_errors, bar_width, label=labels[j], color=colors[j])

        ax5.set_xlabel('Coordinate')
        ax5.set_ylabel('RMS Error')
        ax5.set_title('RMS Tracking Error Comparison')
        ax5.set_xticks(index + bar_width/2)
        ax5.set_xticklabels(coord_labels)
        ax5.legend()
        plt.tight_layout()
        plt.savefig('figures/rvg/model_uncertainty/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/rms_error_comparison.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    print("Plots saved")
