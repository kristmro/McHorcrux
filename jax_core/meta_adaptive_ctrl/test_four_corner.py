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

# Uncomment this line to force using the CPU
jax.config.update('jax_platform_name', 'cpu')  # TODO: keep or remove?

#-----------------------------------------------------------------
# Updated reference trajectory: Four-Corner Test (MC-GYM style)
#-----------------------------------------------------------------
def reference(t):
    """
    Generate a reference trajectory for the boat using a four-corner test.

    The trajectory transitions through the following set points:
      [2.0, 2.0, 0.0] -> [4.0, 2.0, 0.0] -> [4.0, 4.0, 0.0] ->
      [4.0, 4.0, -π/4] -> [2.0, 4.0, -π/4] -> [2.0, 2.0, 0.0]

    The first 10 seconds hold the initial point; afterwards, the function
    linearly interpolates between points over the simulation time (T = 400 seconds).

    Args:
        t: Time (seconds)
        
    Returns:
        r: Reference position [x, y, φ] as a 1D array of shape (3,)
    """
    T = 400.0  # Total simulation time

    # Convert the set points to a JAX array (shape: [6, 3])
    points = jnp.array([
        [2.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 4.0, 0.0],
        [4.0, 4.0, -jnp.pi/4],
        [2.0, 4.0, -jnp.pi/4],
        [2.0, 2.0, 0.0]
    ])

    # If t < 10.0, hold the initial point.
    def hold_initial(_):
        return points[0]

    # If shifted time exceeds available time, return the last point.
    def too_far(_):
        return points[-1]

    # Compute the segment index and perform linear interpolation.
    def within_time(_):
        shifted_t = t - 10.0
        total_segment_time = T - 10.0
        segment_duration = total_segment_time / 5.0  # 5 segments between 6 points
        
        seg_idx = jnp.floor(shifted_t / segment_duration).astype(jnp.int32)
        # Ensure seg_idx does not exceed the maximum valid index (4)
        seg_idx = jnp.minimum(seg_idx, 4)
        
        frac = (shifted_t - seg_idx * segment_duration) / segment_duration
        
        # Use dynamic indexing with keepdims=True, then squeeze to get a (3,) array.
        p0 = jax.lax.dynamic_index_in_dim(points, seg_idx, axis=0, keepdims=True)
        p1 = jax.lax.dynamic_index_in_dim(points, seg_idx + 1, axis=0, keepdims=True)
        p0 = jnp.squeeze(p0, axis=0)
        p1 = jnp.squeeze(p1, axis=0)
        return (1 - frac) * p0 + frac * p1

    def interpolate(_):
        shifted_t = t - 10.0
        total_segment_time = T - 10.0
        return jax.lax.cond(shifted_t >= total_segment_time, too_far, within_time, operand=None)

    return jax.lax.cond(t < 10.0, hold_initial, interpolate, operand=None)



# The remainder of the code remains unchanged

if __name__ == "__main__":
    print('Testing ... ', flush=True)
    start = time.time()
    seed, M = 0, 10

    # Sampled-time simulator
    @jax.tree_util.Partial(jax.jit, static_argnums=(3,))
    def simulate(ts, w, params, reference,
                 plant=plant, prior=prior_3dof, disturbance=wave_load):
        """TODO: docstring."""
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
            M, D, G, R = prior(q, dq)
            τ = M @ dv + D @ v + G @ q - f_hat - K @ s
            u = jnp.linalg.solve(R, τ)
            return u, τ

        # Closed-loop ODE for `x = (q, dq)`, with a zero-order hold on the controller
        def ode(x, t, u, w=w):
            q, dq = x
            f_ext = disturbance(t, q, w)
            ddq = plant(q, dq, u, f_ext)
            dx = (dq, ddq)
            return dx

        # Simulation loop
        def loop(carry, input_slice):
            t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev = carry
            t = input_slice
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref_derivatives(t)

            # Integrate adaptation law via trapezoidal rule
            dA, y = adaptation_law(q, dq, r, dr)
            A = A_prev + (t - t_prev) * (dA_prev + dA) / 2

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

    # Choose a wind velocity, fixed control gains, and simulation times
    num_dof = 3
    key = jax.random.PRNGKey(seed)
    w = disturbance(jnp.array((10*(1/90), 20*(1/90)**0.5, 270)), key)
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
    filename = os.path.join('data', 'training_results', 'seed={}_M={}.pkl'.format(seed, M))
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
    filename = os.path.join('data', 'training_results', 'seed={}_M={}.pkl'.format(seed, M))
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
