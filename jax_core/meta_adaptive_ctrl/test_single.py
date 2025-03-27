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

# Uncomment this line to force using the CPU
jax.config.update('jax_platform_name', 'cpu')  # TODO: keep or remove?

if __name__ == "__main__":
    print('Testing ... ', flush=True)
    start = time.time()
    seed, M = 0, 10

    # Sampled-time simulator
    @jax.tree_util.Partial(jax.jit, static_argnums=(3,))
    def simulate(ts, w, params, reference,
                 plant=plant, prior=prior_3dof, disturbance=wave_load):
        """TODO: docstring."""
        thruster_config = get_default_config()
        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
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
            M, D, G, R = prior(q, dq)
            τ = M@dv + D@v + G@q - f_hat - K@s
            u = jnp.linalg.solve(R, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on
        # the controller
        def ode(x, t, u, w=w):
            q, dq = x
            f_ext = disturbance(t, q, w)
            ddq = plant(q, dq, u, f_ext)
            dx = (dq, ddq)
            return dx

        # Simulation loop
        def loop(carry, input_slice):
            t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev, alpha_prev, u_f_prev = carry
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

            # Thuster saturationD
            u_sat, alpha = allocate_with_config(
                τ, 
                thruster_config, 
                DEFAULT_THRUST_MAX, 
                DEFAULT_THRUST_MIN
            )
            
            u_rate_sat, alpha_rate_sat = saturate_rate(
                u_sat, alpha, u_f_prev, alpha_prev, 
                DEFAULT_DT, DEFAULT_N_DOT_MAX, DEFAULT_ALPHA_DOT_MAX
            )
            
            tau_aft = map_to_3dof(u_rate_sat, alpha_rate_sat, thruster_config)
                

            carry = (t, q, dq, u, A, dA, alpha, u_rate_sat)
            output_slice = (q, dq, u, tau_aft, r, dr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        q0, dq0 = r0, dr0
        dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
        A0 = jnp.zeros((q0.size, y0.size))
        f0 = A0 @ y0
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0, f0)
        alpha0 = jnp.zeros(6)
        u_f0 = jnp.zeros(6)
        # Run simulation loop
        carry = (t0, q0, dq0, u0, A0, dA0, alpha0, u_f0)
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
        ϕ_max = jnp.pi/3   # maximum roll angle (achieved at top of loop)
    
        x = (w/2)*jnp.sin(2*jnp.pi * t/T) + d*(t/T)
        y = (h/2)*(1 - jnp.cos(2*jnp.pi * t/T))
        ϕ = 4*ϕ_max*(t/T)*(1-t/T)
        r = jnp.array([x, y, ϕ])
        return r

    # Choose a wind velocity, fixed control gains, and simulation times
    num_dof = 3
    key = jax.random.PRNGKey(seed)
    w = disturbance(jnp.array((10*(1/90), 20*(1/90)**0.5, 270)),key)
    λ, k, p = 0.1, 1., 1.
    T, dt = 400., 0.01
    ts = jnp.arange(0, T + dt, dt)

    # Simulate tracking for each method
    test_results = {
        'w': w,
        'gains': (λ, k, p),
    }

    # Our method with meta-learned gains
    print('  ours (meta) ...', flush=True)
    filename = os.path.join('data', 'training_results','seed={}_M={}.pkl'.format(seed, M))
    with open(filename, 'rb') as file:
        train_results = pickle.load(file)
    params = {
        'W': train_results['model']['W'],
        'b': train_results['model']['b'],
        'Λ': params_to_posdef(train_results['controller']['Λ']),
        'K': params_to_posdef(train_results['controller']['K']),
        'P': params_to_posdef(train_results['controller']['P']),
    }
    q, dq, u, τ, r, dr = simulate(ts, w, params, reference)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    test_results['meta_adap_ctrl'] = {
        'params': params,
        't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
        'u': u, 'τ': τ, 'e': e,
    }
    filename = os.path.join(
        'data','training_results',
        'seed={}_M={}.pkl'.format(seed, M)
    )
    with open(filename, 'rb') as file:
        train_results = pickle.load(file)
    params = {
        'W': train_results['model']['W'],
        'b': train_results['model']['b'],
    }
    params['Λ'] = λ * jnp.eye(num_dof)
    params['K'] = k * jnp.eye(num_dof)
    params['P'] = p * jnp.eye(num_dof)
    q, dq, u, τ, r, dr = simulate(ts, w, params, reference)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    test_results['adaptive_ctrl'] = {
        'params': params,
        't': ts, 'q': q, 'dq': dq, 'r': r, 'dr': dr,
        'u': u, 'τ': τ, 'e': e,
    }

    # Save
    with open('data/testing_results/test_results_single.pkl', 'wb') as file:
        pickle.dump(test_results, file)

    end = time.time()
    print('done! ({:.2f} s)'.format(end - start))