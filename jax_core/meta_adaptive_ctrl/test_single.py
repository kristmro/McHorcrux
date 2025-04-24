"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
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
from jax_core.utils import params_to_posdef              # noqa: E402
from jax_core.meta_adaptive_ctrl.dynamics import prior_3dof, plant, disturbance  # noqa: E402
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
# Uncomment this line to force using the CPU
jax.config.update('jax_platform_name', 'cpu')  # TODO: keep or remove?

if __name__ == "__main__":
    print('Testing ... ', flush=True)
    start = time.time()
    seed, M, ctrl_pen, act, test_act = 7, 10, 3, 'off', 'off'

    # Sampled-time simulator
    @partial(jax.jit, static_argnums=(3,))
    def simulate(ts, w, params, reference,
                 plant=plant, prior=prior_3dof, disturbance=wave_load):
        """TODO: docstring."""
        # thruster_config = get_default_config()
        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
            ddr = jnp.nan_to_num(ddr, nan=0.)
            return r, dr, ddr

        # Adaptation law
        def adaptation_law(q, dq, r, dr, params=params):
            # Regressor features
            y = jnp.concatenate((q, dq))
            for W, b in zip(params['W'], params['b']):
                y = jnp.tanh(W@y + b)

            # Auxiliary signals
            Λ, P = params['Λ'], params['P']
            e, de = q - r, dq - dr
            s = de + Λ@e

            dA = P @ jnp.outer(s, y)
            return dA, y

        # Controller
        def controller(q, dq, r, dr, ddr, f_hat, params=params):
            # Auxiliary signals
            Λ, K = params['Λ'], params['K']
            e, de = q - r, dq - dr
            s = de + Λ@e
            v, dv = dr - Λ@e, ddr - Λ@de

            # Control input and adaptation law
            M_mat, D, G, R = prior(q, dq)
            τ = M_mat @ dv + D @ v + G @ q - f_hat - K @ s
            u = jnp.linalg.solve(R, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on the controller
        def ode(x, t, u, w=w):
            q, dq = x
            f_ext = disturbance(t, q, w)
            dq, ddq = plant(q, dq, u, f_ext)
            dx = (dq, ddq)
            return dx

        # Simulation loop
        def loop(carry, input_slice):
            t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev = carry
            t = input_slice
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]),
                             u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref_derivatives(t)

            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            A = A_prev + (t - t_prev)*(dA_prev + dA)/2

            # Compute force estimate and control input
            f_hat = A @ y
            u, τ = controller(q, dq, r, dr, ddr, f_hat)

            # # Thuster saturationD
            # u_sat, alpha = allocate_with_config(
            #     u, 
            #     thruster_config, 
            #     DEFAULT_THRUST_MAX, 
            #     DEFAULT_THRUST_MIN
            # )
            
            # u_rate_sat, alpha_rate_sat = saturate_rate(
            #     u_sat, alpha, u_f_prev, alpha_prev, 
            #     DEFAULT_DT, DEFAULT_N_DOT_MAX, DEFAULT_ALPHA_DOT_MAX
            # )
            
            # u_aft = map_to_3dof(u_rate_sat, alpha_rate_sat, thruster_config)
                

            carry = (t, q, dq, u, A, dA)
            output_slice = (q, dq, u, τ, r, dr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        q0, dq0 = r0, dr0
        dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
        A0 = jnp.zeros((q0.size, y0.size))
        f0 = A0 @ y0
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0, f0)
        # alpha0 = jnp.zeros(6)
        # u_f0 = jnp.zeros(6)
        # Run simulation loop
        carry = (t0, q0, dq0, u0, A0, dA0)
        carry, output = jax.lax.scan(loop, carry, ts[1:])
        q, dq, u, τ, r, dr = output

        # Prepend initial conditions
        q = jnp.vstack((q0, q))
        dq = jnp.vstack((dq0, dq))
        u = jnp.vstack((u0, u))
        τ = jnp.vstack((τ0, τ))
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))

        return q, dq, u, τ, r, dr

    # Construct a trajectory
    def reference(t):
        """Generate a reference trajectory for the boat to follow.
        
        The trajectory consists of one loop over the entire simulation period.
        
        Args:
            t: Time (seconds)
            
        Returns:
            r: Reference position [x, y, φ]
        """
        T = 400.            # loop period (changed from 10 to 60 seconds)
        d = 4.             # displacement along `x` from `t=0` to `t=T`
        w = 6.             # loop width
        h = 4.             # loop height
        ϕ_max = jnp.pi/3   # maximum yaw1 angle (achieved at top of loop)
    
        x = (w/2)*jnp.sin(2*jnp.pi * t/T) + d*(t/T)
        y = (h/2)*(1 - jnp.cos(2*jnp.pi * t/T))
        ϕ = 4*ϕ_max*(t/T)*(1-t/T)
        r = jnp.array([x, y, ϕ])
        return r

    # Choose a wind velocity, fixed control gains, and simulation times
    num_dof = 3
    key = jax.random.PRNGKey(seed)
    w = disturbance(jnp.array((6*(1/90), 16*(1/90)**0.5, 0)),key)
    λ, k, p = 1.0, 10.0, 10.0
    T, dt = 400., 0.01
    ts = jnp.arange(0, T + dt, dt)

    # Simulate tracking for each method
    test_results = {
        'w': w,
        'gains': (λ, k, p),
    }

    # Our method with meta-learned gains
    print('meta trained adaptive ctrl ...', flush=True)
    filename = os.path.join('data', 'training_results','act_{}'.format(act),'ctrl_pen_{}'.format(ctrl_pen),'seed={}_M={}.pkl'.format(seed, M))
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
    q, dq, u, τ, r, dr = simulate(ts, w, params, reference)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    test_results['meta_adap_ctrl'] = {
        'params': params,
        't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
        'u': u, 'τ': τ, 'e': e,
    }
    
    for method in ('pid', 'adaptive_ctrl'):
        if method == 'pid':
            print('PID Ctrl...', flush=True)
            params = {
                'W': [jnp.zeros((1, 2*num_dof)), ],
                'b': [jnp.inf * jnp.ones((1,)), ],
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
        q, dq, u, τ, r, dr = simulate(ts, w, params, reference)
        e = np.concatenate((q - r, dq - dr), axis=-1)
        test_results[method] = {
            'params': params,
            't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
            'u': u, 'τ': τ, 'e': e,
        }

    output_path = os.path.join('data', 'testing_results','train_act_{}'.format(act),'loop','test_act_{}'.format(test_act),'ctrl_pen_{}'.format(ctrl_pen),'seed={}_M={}.pkl'.format(seed, M))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save
    with open(output_path, 'wb') as file:
        pickle.dump(test_results, file)

    end = time.time()
    print('done! ({:.2f} s)'.format(end - start))
    #--------------------------------------------------------------------
    # Plotting
    #--------------------------------------------------------------------
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    which_test = 'loop'
    # Load the test results
    print("Loading test results...")
    with open('data/testing_results/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}.pkl'.format(act,which_test,test_act,ctrl_pen,seed,M), 'rb') as file:
        results = pickle.load(file)

    # Create figures directory if it doesn't exist
    os.makedirs('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}'.format(act,which_test,test_act,ctrl_pen, seed, M), exist_ok=True)

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
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/position_tracking.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

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
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/trajectory_2d.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

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
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/tracking_errors.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

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
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_cmd.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

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
    plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/control_efforts_u_after.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

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
        plt.savefig('figures/train_act_{}/{}/test_act_{}/ctrl_pen_{}/seed={}_M={}/rms_error_comparison.png'.format(act,which_test,test_act,ctrl_pen,seed,M), dpi=300)

    print("Plots saved")
    plt.show()