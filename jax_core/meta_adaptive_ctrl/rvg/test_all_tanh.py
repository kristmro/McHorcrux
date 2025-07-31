"""
Test learned models of the adaptive controller on the 6-DOF plant with
wave loads - plus a pure PID benchmark.

Author: Kristian Magnus Røen
"""

# ---------------------------------------------------------------------#
#  Imports                                                            #
# ---------------------------------------------------------------------#
import argparse
import os
import pickle
import time
from functools import partial
from itertools import product
from math import inf, pi

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jax.experimental.ode import odeint
import numpy as np
from tqdm.auto import tqdm

# Project-local helpers
from jax_core.meta_adaptive_ctrl.rvg.dynamics import (
    disturbance, plant_6 as plant, prior_3dof_nom as prior,
)
from jax_core.utils import (
    params_to_posdef,
    random_ragged_spline,
    spline,
    vec_to_posdef_diag_cholesky,
)
from jax_core.simulator.waves.wave_load_jax_jit import wave_load

# ---------------------------------------------------------------------#
#  CLI & precision flags                                               #
# ---------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help="seed for pseudo-random number generation")
parser.add_argument("M",    type=int, help="number of trajectories to sub-sample")
parser.add_argument("--use_x64", action="store_true", help="use 64-bit precision")
parser.add_argument("--use_cpu", action="store_true", help="force CPU backend")
args = parser.parse_args()

if args.use_x64:
    jax.config.update("jax_enable_x64", True)
if args.use_cpu:
    jax.config.update("jax_platform_name", "cpu")

# ---------------------------------------------------------------------#
#  Hyper-parameters                                                    #
# ---------------------------------------------------------------------#
hparams = {
    "seed":        args.seed,
    "use_x64":     args.use_x64,
    "num_subtraj": args.M,

    # Wave distribution
    "hs_min": 0.5,
    "hs_max": 7.0,
    "w_dir": 45.0,          # ±deg around x-axis
    "tp_min": 7.0,
    "tp_max": 18.0,
    "a": 6.0,               # beta distribution (a, b)
    "b": 3.0,

    # Reference trajectories
    "T":         10.0,
    "dt":        1e-3,
    "num_refs":  200,
    "num_knots": 6,
    "poly_orders":  (9, 9, 6),
    "deriv_orders": (4, 4, 2),
    "min_step":  (-0.4, -0.4, -pi/8),
    "max_step":  (0.4,  0.4,  pi/8),
    "min_ref":   (-inf, -inf, -pi/3),
    "max_ref":   (inf,  inf,  pi/3),
}

num_dof = 3          # 3 controlled DOF (X,Y,Yaw)
test_seed = 42
key = jax.random.PRNGKey(test_seed)

# ---------------------------------------------------------------------#
#  Utility: enumerated cartesian product                               #
# ---------------------------------------------------------------------#
def enumerated_product(*args):
    """Yield ((i,j,k,…), (a,b,c,…)) across a grid of tuples."""
    yield from zip(product(*(range(len(x)) for x in args)),
                   product(*args))

# ---------------------------------------------------------------------#
#  Reference spline generation                                         #
# ---------------------------------------------------------------------#
def make_reference_splines(key):
    key, *subkeys = jax.random.split(key, 1 + hparams["num_refs"])
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
    return key, t_knots, coefs, min_ref, max_ref

# ---------------------------------------------------------------------#
#  Wave-load batch generation                                          #
# ---------------------------------------------------------------------#
def make_wave_batch(key):
    a, b = hparams["a"], hparams["b"]
    hs_min, hs_max = hparams["hs_min"], hparams["hs_max"]
    tp_min, tp_max = hparams["tp_min"], hparams["tp_max"]
    dir_span = hparams["w_dir"]
    num_traj = hparams["num_refs"]

    key, subkey = jax.random.split(key, 2)
    hs  = hs_min + (hs_max - hs_min) * jax.random.beta(subkey, a, b, (num_traj,))
    tp  = tp_min + (tp_max - tp_min) * jax.random.beta(subkey, a, b, (num_traj,))
    wdir = jnp.rint(jax.random.uniform(key, (num_traj,), minval=-dir_span, maxval=dir_span))

    wl_list = []
    for i in range(num_traj):
        wl_list.append(disturbance((hs[i], tp[i], wdir[i]), key))

    wl_batched = tu.tree_map(lambda *xs: jnp.stack(xs), *wl_list)
    return key, wl_batched, hs, hs_min, hs_max, a, b

# ---------------------------------------------------------------------#
#  Adaptive controllers simulator (original)                           #
# ---------------------------------------------------------------------#
min_ref = jnp.asarray(hparams['min_ref'])
max_ref = jnp.asarray(hparams['max_ref'])
@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def simulate_adaptive(ts, wl, t_knots, coefs, params, min_ref=min_ref, max_ref=max_ref,
                      plant=plant, prior=prior, disturbance=wave_load):
    """Simulate boat with (meta-)adaptive controller."""

    # ---- reference spline ------------------------------------------------
    def reference(t):
        r = jnp.array([spline(t, t_knots, c) for c in coefs])
        return jnp.clip(r, min_ref, max_ref)

    def ref_derivatives(t):
        vel_fn = jax.jacfwd(reference)
        acc_fn = jax.jacfwd(vel_fn)
        r  = reference(t)
        dr = vel_fn(t)
        ddr = jnp.nan_to_num(acc_fn(t), nan=0.)
        return r, dr, ddr

    # ---- adaptation law --------------------------------------------------
    def adaptation_law(q, dq, r, dr):
        y = jnp.concatenate((q, dq))
        for W, b in zip(params['W'], params['b']):
            y = jnp.tanh(W @ y + b)
        Λ, P = params['Λ'], params['P']
        e, de = q - r, dq - dr
        s = de + Λ @ e
        dA = P @ jnp.outer(s, y)
        return dA, y

    # ---- controller ------------------------------------------------------
    def controller(q, dq, r, dr, ddr, f_hat):
        Λ, K = params['Λ'], params['K']
        e, de = q - r, dq - dr
        s = de + Λ @ e
        v, dv = dr - Λ @ e, ddr - Λ @ de
        # def sat(s):
        #     """Saturation function."""
        #     phi = 7
        #     return jnp.where(jnp.abs(s/phi) > 1, jnp.sign(s), s/phi)
        def sat(s):
            """Saturation function."""
            espilon_1=5
            espilon_2=7
            return jnp.tanh(s/espilon_1) * (espilon_2+1)
        # Controller and adaptation law
        M, D, G, R = prior(q, dq)
        τ = M@dv + D@v + G@e - f_hat - K @ sat(s)
        u = jnp.linalg.solve(R, τ)
        return u, τ

    # ---- plant ODE w/ zero-order hold ------------------------------------
    def ode(state, t, u):
        q, dq = state
        f_ext = disturbance(t, q, wl)
        dq, ddq = plant(q, dq, u, f_ext)
        return (dq, ddq)

    # ---- scan step -------------------------------------------------------
    def step(carry, t):
        t_prev, q_prev, dq_prev, u_prev, A_prev, dA_prev = carry
        qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
        q, dq = qs[-1], dqs[-1]

        r, dr, ddr = ref_derivatives(t)
        dA, y  = adaptation_law(q, dq, r, dr)
        A      = A_prev + 0.5 * (t - t_prev) * (dA_prev + dA)
        f_hat  = A @ y
        u, τ   = controller(q, dq, r, dr, ddr, f_hat)
        new_carry = (t, q, dq, u, A, dA)
        out_slice = (q, dq, u, τ, r, dr)
        return new_carry, out_slice

    # ---- initial conditions ---------------------------------------------
    t0 = ts[0]
    r0, dr0, ddr0 = ref_derivatives(t0)
    q0, dq0 = r0, dr0
    dA0, y0 = adaptation_law(q0, dq0, r0, dr0)
    A0      = jnp.zeros((q0.size, y0.size))
    f0      = A0 @ y0
    u0, τ0  = controller(q0, dq0, r0, dr0, ddr0, f0)

    #Run simulation loop
    carry = (t0, q0, dq0, u0, A0, dA0)
    carry, outputs = jax.lax.scan(step,carry,ts[1:])
    q, dq, u, τ, r, dr = outputs

    # prepend t0 values for full length tensors
    q  = jnp.vstack((q0, q))
    dq = jnp.vstack((dq0, dq))
    u  = jnp.vstack((u0, u))
    τ  = jnp.vstack((τ0, τ))
    r  = jnp.vstack((r0, r))
    dr = jnp.vstack((dr0, dr))
    return q, dq, u, τ, r, dr

# ---------------------------------------------------------------------#
#  Pure PID simulator                                                  #
# ---------------------------------------------------------------------#
@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def simulate_pid(ts, wl, t_knots, coefs, params,
                 min_ref=min_ref, max_ref=max_ref,prior=prior,
                 plant=plant, disturbance=wave_load):
    """Simulate closed-loop system with a vector PID controller."""

    Kp_mat = jnp.diag(params['Kp'])
    Ki_mat = jnp.diag(params['Ki'])
    Kd_mat = jnp.diag(params['Kd'])

    # ---- reference spline ----------------------------------------------
    def reference(t):
        r = jnp.array([spline(t, t_knots, c) for c in coefs])
        r = r.at[2].set(jnp.mod(r[2] + jnp.pi, 2 * jnp.pi) - jnp.pi)
        return jnp.clip(r, min_ref, max_ref)

    def ref_derivatives(t):
        vel_fn = jax.jacfwd(reference)
        acc_fn = jax.jacfwd(vel_fn)
        r  = reference(t)
        dr = vel_fn(t)
        ddr = jnp.nan_to_num(acc_fn(t), nan=0.0)
        return r, dr, ddr

    # ---- plant ODE w/ ZOH ----------------------------------------------
    def ode(state, t, u):
        q, dq = state
        f_ext = disturbance(t, q, wl)
        dq, ddq = plant(q, dq, u, f_ext)
        return (dq, ddq)

    # ---- scan step ------------------------------------------------------
    def step(carry, t):
        t_prev, q_prev, dq_prev, u_prev, I_prev = carry
        qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
        q, dq = qs[-1], dqs[-1]

        r_prev,_,_ = ref_derivatives(t_prev)
        r, dr, _ = ref_derivatives(t)
        e, de = q - r, dq - dr
        dt = t - t_prev
        # Using the trapezoidal rule
        I = I_prev + 0.5 * (e + (q_prev - r_prev)) * dt
        
        τ = -(Kp_mat @ e + Ki_mat @ I + Kd_mat @ de)
        _, _, _, R = prior(q, dq)
        u = jnp.linalg.solve(R, τ)

        new_carry = (t, q, dq, u, I)
        out_slice = (q, dq, u, τ, r, dr)
        return new_carry, out_slice

    # ---- initial conditions --------------------------------------------
    t0 = ts[0]
    r0, dr0, _ = ref_derivatives(t0)
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

# ---------------------------------------------------------------------#
#  Entry point                                                         #
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    print("Testing …", flush=True)
    t_start = time.time()

    # ---- reference + waves (shared across all controllers) -------------
    key, t_knots, coefs, min_ref, max_ref = make_reference_splines(key)
    key, wl_batched, hs, hs_min, hs_max, a, b = make_wave_batch(key)

    # ---- common simulation clock ---------------------------------------
    T, dt = hparams["T"], hparams["dt"]
    ts = jnp.arange(0.0, T + dt, dt)

    # -----------------------------------------------------------------#
    #  Result container                                                #
    # -----------------------------------------------------------------#
    test_results = {
        'hs' : hs, 'hs_min' : hs_min, 'hs_max' : hs_max,
        'beta_params' : (a, b),
        "gains": {
            # adaptive controllers: Λ (Kd), K (Kp), P (Ki) – three scalars
            "Λ": (10.0,),
            "K": (50.0, 100.0),
            "P": (50.0, 100.0),
            # PID-only grid 
            "Kp": (jnp.array([1293452.  ,   135905.16, 8.8149e+06]),),
            "Ki": (jnp.array([1.4856e+04, 2.5148e+01, 1.6869e+00]),jnp.array([6.5168441e+04, 3.9588146e+01, 1.7311e+00])),
            "Kd": (jnp.array([1.0000e+04, 1.0000e+05, 9.8118e+03]), jnp.array([6.1398863e+04, 2.8914574e+05, 1.3911e+04])),
        },
    }
    # grid shapes (adaptive and PID use different sets but same length tuples)
    grid_shape_adapt = (
        len(test_results["gains"]["Λ"]),
        len(test_results["gains"]["K"]),
        len(test_results["gains"]["P"]),
    )
    grid_shape_pid = (
        len(test_results["gains"]["Kp"]),
        len(test_results["gains"]["Ki"]),
        len(test_results["gains"]["Kd"]),
    )

    # -----------------------------------------------------------------#
    #  1) Meta-adaptive controller (single run)                        #
    # -----------------------------------------------------------------#
    print("Meta-trained adaptive controller …", flush=True)
    train_path = os.path.join(
        "data",
        "training_results",
        "rvg",
        "model_uncertainty",
        "tanh",
        "act_off",
        "ctrl_pen_6",
        f"seed={hparams['seed']}_M={hparams['num_subtraj']}.pkl",
    )
    with open(train_path, "rb") as f:
        train_results = pickle.load(f)

    if train_results["controller"]["Λ"].shape[-1] == 2 * num_dof:
        params_meta = {
            "W": train_results["model"]["W"],
            "b": train_results["model"]["b"],
            "Λ": params_to_posdef(train_results["controller"]["Λ"]),
            "K": params_to_posdef(train_results["controller"]["K"]),
            "P": params_to_posdef(train_results["controller"]["P"]),
        }
    else:  # diagonal reps
        params_meta = {
            "W": train_results["model"]["W"],
            "b": train_results["model"]["b"],
            "Λ": vec_to_posdef_diag_cholesky(train_results["controller"]["Λ"]),
            "K": vec_to_posdef_diag_cholesky(train_results["controller"]["K"]),
            "P": vec_to_posdef_diag_cholesky(train_results["controller"]["P"]),
        }

    q, dq, u, τ, r, dr = simulate_adaptive(ts, wl_batched, t_knots, coefs, params_meta)
    e = np.concatenate((q - r, dq - dr), axis=-1)
    rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
    rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
    test_results['meta_adaptive_ctrl'] = {
        'params':    params_meta,
        'rms_error': rms_e,
        'rms_ctrl':  rms_u,
    }

    # -----------------------------------------------------------------#
    #  2) Adaptive controller – grid sweep                             #
    # -----------------------------------------------------------------#
    print("Adaptive controller grid sweep …", flush=True)
    test_results["adaptive_ctrl"] = np.empty(grid_shape_adapt, dtype=object)

    params_base = {
            "W": train_results["model"]["W"],
            "b": train_results["model"]["b"]
        }

    for (i, j, l), (λ, k, p) in tqdm(enumerated_product(
        test_results['gains']['Λ'],
        test_results['gains']['K'],
        test_results['gains']['P']), total=np.prod(grid_shape_adapt)
    ):
        params_base['Λ'] = λ * jnp.eye(num_dof)
        params_base['K'] = k * jnp.eye(num_dof)
        params_base['P'] = p * jnp.eye(num_dof)
        params_base['Λ'] = params_base['Λ'].at[-1, -1].set(0.1 * λ)
        params_base['K'] = params_base['K'].at[-1, -1].set(0.1 * k)
        params_base['P'] = params_base['P'].at[-1, -1].set(0.1 * p)
        q, dq, u, τ, r, dr = simulate_adaptive(ts, wl_batched, t_knots, coefs, params_base)
        e = np.concatenate((q - r, dq - dr), axis=-1)
        rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
        rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
        test_results['adaptive_ctrl'][i, j, l] = {
            'params':    params_base,
            'rms_error': rms_e,
            'rms_ctrl':  rms_u,
        }


    # -----------------------------------------------------------------#
    #  3) Pure PID controller – grid sweep                             #
    # -----------------------------------------------------------------#
    print("PID controller grid sweep …", flush=True)
    test_results["pid"] = np.empty(grid_shape_pid, dtype=object)

    params_pid = {}
    for (i, j, l), (Kp_gain, Ki_gain, Kd_gain) in tqdm(enumerated_product(
        test_results["gains"]["Kp"],
        test_results["gains"]["Ki"],
        test_results["gains"]["Kd"],),total=np.prod(grid_shape_pid),
    ):
        params_pid['Kp'] = Kp_gain          # (shape (3,))
        params_pid['Ki'] = Ki_gain
        params_pid['Kd'] = Kd_gain
        q, dq, u, τ, r, dr = simulate_pid(ts, wl_batched, t_knots, coefs, params_pid)
        e = np.concatenate((q - r, dq - dr), axis=-1)
        rms_e = np.sqrt(np.mean(np.sum(e**2, axis=-1), axis=-1))
        rms_u = np.sqrt(np.mean(np.sum(u**2, axis=-1), axis=-1))
        test_results['pid'][i, j, l] = {
            'params':    params_pid,
            'rms_error': rms_e,
            'rms_ctrl':  rms_u,
        }

    # -----------------------------------------------------------------#
    #  Save results                                                    #
    # -----------------------------------------------------------------#
    out_dir = os.path.join(
        "data",
        "testing_results",
        "rvg",
        "model_uncertainty",
        "tanh",
        "all",
        "act_off",
        "ctrl_pen_6",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed={hparams['seed']}_M={hparams['num_subtraj']}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(test_results, f)

    print(f"Done! (elapsed {time.time() - t_start:.2f} s)")
