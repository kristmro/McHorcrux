"""
Test learned models of the PFAR system with adaptive closed-loop feedback.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
import pickle
import time
from itertools import product
from functools import partial
from math import inf, pi

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jax.experimental.ode import odeint


import numpy as np

from tqdm.auto import tqdm

from jax_core.meta_adaptive_ctrl.dynamics import disturbance, plant, prior_3dof as prior
from jax_core.utils import params_to_posdef, random_ragged_spline, spline, vec_to_posdef_diag_cholesky
from jax_core.simulator.waves.wave_load_jax_jit import wave_load


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for pseudo-random number generation',
                    type=int)
parser.add_argument('M', help='number of trajectories to sub-sample',
                    type=int)
parser.add_argument('--use_x64', help='use 64-bit precision',
                    action='store_true')
parser.add_argument('--use_cpu', help='use CPU only',
                    action='store_true')
args = parser.parse_args()

# Set precision
if args.use_x64:
    jax.config.update('jax_enable_x64', True)
if args.use_cpu:
    jax.config.update('jax_platform_name', 'cpu')

# Initialize PRNG key (with offset from original seed to make sure we do not
# sample the same reference trajectories in the training set)
test_seed = 42
key = jax.random.PRNGKey(test_seed)

hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories sub-sampled

    'hs_min': 0.5,     # minimum hs
    'hs_max': 7.,    # maximum hs
    'w_dir': 45.,     # wave direction
    'tp_min': 7.,     # minimum tp
    'tp_max': 20.,    # maximum tp
    'a': 6.,         # shape parameter `a` for beta distribution
    'b': 3.,         # shape parameter `b` for beta distribution

    # Reference trajectory generation
    'T':            10.,                  # time horizon for each reference
    'dt':           1e-3,                 # numerical integration time step
    'num_refs':     200,                  # reference trajectories to generate
    'num_knots':    6,                    # knot points per reference spline
    'poly_orders':  (9, 9, 6),            # spline orders for each DOF
    'deriv_orders': (4, 4, 2),            # smoothness objective for each DOF
    'min_step':     (-0.4, -0.4, -pi/8),  #
    'max_step':     (0.4, 0.4, pi/8),     #
    'min_ref':      (-inf, -inf, -pi/3),  #
    'max_ref':      (inf, inf, pi/3),     #
}


def enumerated_product(*args):
    """Generate an enumeration over all possible combinations."""
    yield from zip(
        product(*(range(len(x)) for x in args)),
        product(*args)
    )


if __name__ == '__main__':
    print('Testing ... ', flush=True)
    start = time.time()

    # Generate reference trajectories
    key, *subkeys = jax.random.split(key, 1 + hparams['num_refs'])
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    min_ref = jnp.asarray(hparams['min_ref'])
    max_ref = jnp.asarray(hparams['max_ref'])
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys,
        hparams['T'],
        hparams['num_knots'],
        hparams['poly_orders'],
        hparams['deriv_orders'],
        jnp.asarray(hparams['min_step']),
        jnp.asarray(hparams['max_step']),
        0.7*min_ref,
        0.7*max_ref,
    )
    r_knots = jnp.dstack(knots)
    num_dof = 3

    # Sample wind velocities from the test distribution
    # Define wave parameters
    scale = 1 / 90
    sqrt_scale = jnp.sqrt(scale)
    a = hparams['a']  # shape parameter `a` for beta distribution
    b = hparams['b']  # shape parameter `b` for beta distribution
    hs_min = hparams['hs_min'] * scale #hs_min
    hs_max = hparams['hs_max'] * scale #hs_max
    tp_min = hparams['tp_min'] * sqrt_scale #tp_min
    tp_max = hparams['tp_max'] * sqrt_scale #tp_max
    wave_dir = hparams['w_dir']  # wave direction
    num_traj = hparams['num_refs']
    key, subkey = jax.random.split(key, 2)
    hs = hs_min + (hs_max - hs_min) * jax.random.beta(subkey, a, b, (num_traj,))
    #wave_dir = jnp.zeros((num_traj,), dtype=int)#making the wave_dir zero
    #wave_dir = jnp.rint(jax.random.uniform(key, (num_traj,), minval=0, maxval=360)).astype(int)
    wave_dir = jnp.rint(jax.random.uniform(key, (num_traj,), minval=-wave_dir, maxval=wave_dir)).astype(int)
    tp = tp_min + (tp_max - tp_min) * jax.random.beta(subkey, a, b, (num_traj,))
    wave_parm = (hs, tp, wave_dir)
    # Initialize wave loads for each trajectory.
    wl_list = []
    # Cannot use jax.vmap here because generation of the wave load params
    # requires large QTF matrices

    for i in range(num_traj):
        print( f"Making init waves for number {i}...")
        wave_parm_single = (hs[i], tp[i], wave_dir[i])
        wl = disturbance(wave_parm_single, key)
        wl_list.append(wl)

    # Stack list of WaveLoad PyTrees into a batched WaveLoad using tree_map.
    wl_batched = tu.tree_map(lambda *x: jnp.stack(x), *wl_list)

    # Sampled-time simulator
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def simulate(ts, wl, t_knots, coefs, params,
                 min_ref=min_ref, max_ref=max_ref,
                 plant=plant, prior=prior, disturbance=wave_load):
        """Simulate the PFAR system with adaptive closed-loop feedback."""
        # Construct spline reference trajectory
        def reference(t):
            r = jnp.array([spline(t, t_knots, c) for c in coefs])
            r = jnp.clip(r, min_ref, max_ref)
            return r

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
            M, D, G, R = prior(q, dq)
            τ = M@dv + D@v + G@q - f_hat - K@s
            u = jnp.linalg.solve(R, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on
        # the controller
        def ode(x, t, u, wl=wl):
            q, dq = x
            f_ext = disturbance(t, q, wl)
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

    # Simulate tracking for each `w`
    T, dt = hparams['T'], hparams['dt']
    ts = jnp.arange(0, T + dt, dt)  # same times for each trajectory

    # Try out different gains
    test_results = {
        'hs': hs, 'hs_min': hs_min, 'hs_max': hs_max,
        'beta_params': (a, b),
        'gains': {
            'Λ': (1.,),
            'K': (1., 10.),
            'P': (1., 10.),
        }
    }
    grid_shape = (len(test_results['gains']['Λ']),
                  len(test_results['gains']['K']),
                  len(test_results['gains']['P']))

    # Our method with meta-learned gains
    print('meta trained adaptive ctrl ...', flush=True)
    filename = os.path.join('data', 'training_results', 'act_off','ctrl_pen_1',
                            'seed={}_M={}.pkl'.format(hparams['seed'],
                                                      hparams['num_subtraj']))
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
    q, dq, u, τ, r, dr = simulate(ts, wl_batched, t_knots, coefs, params)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
    rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
    test_results['meta_adaptive_ctrl'] = {
        'params':    params,
        'rms_error': rms_e,
        'rms_ctrl':  rms_u,
    }

    for method in ('pid', 'adaptive_ctrl'):
        test_results[method] = np.empty(grid_shape, dtype=object)
        print('  {} ...'.format(method), flush=True)
        if method == 'pid':
            print('PID Ctrl...', flush=True)
            params = {
                'W': [jnp.zeros((1, 2*num_dof)), ],
                'b': [jnp.inf * jnp.ones((1,)), ],
            }
        else:
            print('Adaptive ctrl self tuned...', flush=True)
            with open(filename, 'rb') as file:
                train_results = pickle.load(file)
            params = {
                'W': train_results['model']['W'],
                'b': train_results['model']['b'],
            }

        for (i, j, l), (λ, k, p) in tqdm(enumerated_product(
            test_results['gains']['Λ'],
            test_results['gains']['K'],
            test_results['gains']['P']), total=np.prod(grid_shape)
        ):
            params['Λ'] = λ * jnp.eye(num_dof)
            params['K'] = k * jnp.eye(num_dof)
            params['P'] = p * jnp.eye(num_dof)
            q, dq, u, τ, r, dr = simulate(ts, wl_batched, t_knots, coefs, params)
            e = np.concatenate((q - r, dq - dr), axis=-1)
            rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
            rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
            test_results[method][i, j, l] = {
                'params':    params,
                'rms_error': rms_e,
                'rms_ctrl':  rms_u,
            }

    # Save
    # Make sure the output directory exists
    os.makedirs('data/testing_results/all/act_off/ctrl_pen_1', exist_ok=True)
    output_filename = os.path.join(
        'data', 'testing_results', 'all', 'act_off', 'ctrl_pen_1',
        'seed={:d}_M={:d}.pkl'.format(hparams['seed'], hparams['num_subtraj'])
    )
    with open(output_filename, 'wb') as file:
        pickle.dump(test_results, file)

    end = time.time()
    print('done! ({:.2f} s)'.format(end - start))
