#!/usr/bin/env python3
"""
PID CEM Tuning – optimise nine log-gains with the cross-entropy method
and report the best candidate.

Usage
-----
    python pid_cem_tuning.py 42 --use_x64

Author
------
Kristian Magnus Roen – CEM version prepared May 2025
"""

# ---------------------------------------------------------------------#
#  Imports                                                             #
# ---------------------------------------------------------------------#
import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpy as np  # only for seed reproducibility in other libs

# Project-local helpers – paths identical to *test_all.py*
from jax_core.meta_adaptive_ctrl.rvg.dynamics import (
    disturbance, plant_6 as plant, prior_3dof_nom as prior,
)
from jax_core.simulator.waves.wave_load_jax_jit import wave_load
from jax_core.utils import random_ragged_spline, spline

# ---------------------------------------------------------------------#
#  CLI & precision flags                                               #
# ---------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help="seed for pseudo-random number generation")
parser.add_argument("--use_x64", action="store_true", help="use 64-bit precision (recommended)")
args = parser.parse_args()

# Reproducibility across JAX & NumPy RNGs
key = jax.random.PRNGKey(args.seed)
np.random.seed(args.seed)

if args.use_x64:
    jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------#
#  Hyper-parameters (copied from *test_all.py*)                        #
# ---------------------------------------------------------------------#
HPARAMS = {
    "T": 10.0,
    "dt": 1e-3,
    "num_knots": 6,
    "poly_orders": (9, 9, 6),
    "deriv_orders": (4, 4, 2),
    "min_step": (-3.0, -3.0, -jnp.pi / 3),
    "max_step": (3.0, 3.0,  jnp.pi / 3),
    "integral_abs_limit": 1e+5,  # Default integral clamp limit
}

NUM_DOF = 3   # 3 controlled DOF (X, Y, Yaw)

# ---------------------------------------------------------------------#
#  Reference spline (single trajectory)                                #
# ---------------------------------------------------------------------#
def make_reference_spline(key):
    key, subkey = jax.random.split(key)
    t_knots, knots, coefs = random_ragged_spline(
        subkey,
        HPARAMS["T"],
        HPARAMS["num_knots"],
        HPARAMS["poly_orders"],
        HPARAMS["deriv_orders"],
        jnp.asarray(HPARAMS["min_step"]),
        jnp.asarray(HPARAMS["max_step"]),
        jnp.array([-jnp.inf, -jnp.inf, -jnp.pi / 3]),
        jnp.array([ jnp.inf,  jnp.inf,  jnp.pi / 3]),
    )
    return key, t_knots, coefs

# ---------------------------------------------------------------------#
#  Wave realisation (single)                                           #
# ---------------------------------------------------------------------#
def make_wave(key):
    key, subkey = jax.random.split(key)
    hs, tp, wdir = 3.0, 15.0, 0.0        # moderate sea state
    wl = disturbance((hs, tp, wdir), subkey)
    return key, wl

# ---------------------------------------------------------------------#
#  PID closed-loop simulator (single trajectory)                       #
# ---------------------------------------------------------------------#
def simulate_pid(ts, wl, t_knots, coefs, Kp, Ki, Kd, integral_abs_limit, prior=prior):
    """Simulate the 3-DOF subsystem with diagonal PID gains."""
    Kp_mat = jnp.diag(Kp)
    Ki_mat = jnp.diag(Ki)
    Kd_mat = jnp.diag(Kd)

    # Define limits for the integral term I (for anti-windup via clamping)
    I_min_limits = jnp.full((NUM_DOF,), -integral_abs_limit)
    I_max_limits = jnp.full((NUM_DOF,), integral_abs_limit)

    # ----- reference spline -----------------------------------------
    def reference(t):
        r = jnp.array([spline(t, t_knots, c) for c in coefs])
        r = r.at[2].set(jnp.mod(r[2] + jnp.pi, 2 * jnp.pi) - jnp.pi)
        return r

    vel_fn = jax.jacfwd(reference)
    acc_fn = jax.jacfwd(vel_fn)

    def ref_derivatives(t):
        r  = reference(t)
        dr = vel_fn(t)
        ddr = jnp.nan_to_num(acc_fn(t), nan=0.0)
        return r, dr, ddr

    # ----- plant ODE ------------------------------------------------
    def ode(state, t, u):
        q, dq = state
        f_ext = wave_load(t, q, wl)
        dq, ddq = plant(q, dq, u, f_ext)
        return dq, ddq

    # ----- scan step -----------------------------------------------
    def step(carry, t):
        t_prev, q_prev, dq_prev, u_prev, I_prev = carry
        qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
        q, dq = qs[-1], dqs[-1]

        r_prev,_,_ = ref_derivatives(t_prev)
        r, dr, _ = ref_derivatives(t)
        e, de = q - r, dq - dr
        dt = t - t_prev

        # Calculate candidate integral update based on original formula
        delta_I = 0.5 * (e + (q_prev - r_prev)) * dt
        I_candidate = I_prev + delta_I

        # Apply integral clamping for anti-windup
        I_clamped = jnp.clip(I_candidate, I_min_limits, I_max_limits)

        M_mat, D, G, R = prior(q, dq)

        # Use the clamped integral I_clamped in the PID calculation
        τ = -(Kp_mat @ e + Ki_mat @ I_clamped + Kd_mat @ de)
        u = jnp.linalg.solve(R, τ)                                     # direct actuation

        new_carry = (t, q, dq, u, I_clamped)
        out_slice = (q, dq, u, τ, r, dr)
        return new_carry, out_slice

    # ----- initial conditions --------------------------------------
    t0 = ts[0]
    r0, dr0, _ = ref_derivatives(t0)
    q0, dq0 = r0, dr0
    I0 = jnp.zeros_like(q0)
    u0 = jnp.zeros_like(q0)

    carry, outputs = jax.lax.scan(step, (t0, q0, dq0, u0, I0), ts[1:])
    q, dq, u, τ, r, dr = outputs

    q  = jnp.vstack((q0,  q))
    dq = jnp.vstack((dq0, dq))
    r  = jnp.vstack((r0,  r))
    dr = jnp.vstack((dr0, dr))

    e = jnp.concatenate((q - r, dq - dr), axis=-1)
    rms_cost = jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1)))
    safe_rms_cost = jnp.nan_to_num(rms_cost, nan=jnp.inf)
    return safe_rms_cost

simulate_pid_jit = jax.jit(simulate_pid, static_argnames=['integral_abs_limit']) # XLA-compile once

# ---------------------------------------------------------------------#
#  Cross-Entropy Method optimiser                                      #
# ---------------------------------------------------------------------#
NUM_SAMPLES = 2000         # samples per generation
ELITE_FRAC  = 0.15        # top 10 % survive
NUM_ITER    = 15          # generations
DIM         = NUM_DOF * 3 # 9 log-gain parameters

init_mean = jnp.array([3.0, 3.0, 3.0,  # log10 Kp
                       2.0, 2.0, 2.0,  # log10 Ki
                       3.5, 3.5, 3.5]) # log10 Kd
init_std  = jnp.ones(DIM) * 1.0        # one order of magnitude spread

def cem_optimize(key, mean, std, ts, wl, t_knots, coefs, integral_abs_limit):
    """
    Cross-Entropy Method loop.

    Parameters
    ----------
    key        : jax.random.PRNGKey
    mean, std  : initial Gaussian parameters in log10-space
    ts, wl,
    t_knots,
    coefs      : simulation artefacts to forward to `simulate_pid_jit`
    integral_abs_limit : absolute limit for integral windup
    """
    best_cost  = jnp.inf
    best_gains = mean

    # Define log-space bounds for each of the DIM parameters
    # Kp: linear_min=1e-1, linear_max=1e6  => log10_min=-1.0, log10_max=6.0
    # Ki: linear_min=1e-1, linear_max=1e5  => log10_min=-1.0, log10_max=5.0
    # Kd: linear_min=1e-1, linear_max=1e5  => log10_min=-1.0, log10_max=5.0
    log_mins_kp = jnp.full((NUM_DOF,), -1.0)
    log_maxs_kp = jnp.full((NUM_DOF,), 7.0)
    log_mins_ki = jnp.full((NUM_DOF,), -1.0)
    log_maxs_ki = jnp.full((NUM_DOF,), 5.0)
    log_mins_kd = jnp.full((NUM_DOF,), -1.0)
    log_maxs_kd = jnp.full((NUM_DOF,), 6.0)
    
    log_mins = jnp.concatenate([log_mins_kp, log_mins_ki, log_mins_kd])
    log_maxs = jnp.concatenate([log_maxs_kp, log_maxs_ki, log_maxs_kd])
    log_ranges = log_maxs - log_mins
    
    epsilon = 1e-6 # Small constant for numerical stability

    @jax.jit
    def batch_cost(log10_batch):
        Kp, Ki, Kd = jnp.split(10 ** log10_batch, 3, axis=-1)

        return jax.vmap(
            simulate_pid_jit,
            in_axes=(None, None, None, None, 0, 0, 0, None) # Pass integral_abs_limit as a static argument
        )(ts, wl, t_knots, coefs, Kp, Ki, Kd, integral_abs_limit)

    for i in range(NUM_ITER): # Modified to get iteration number
        key, subkey = jax.random.split(key)

        # --- Sample using Beta distribution based on current mean and std ---
        # 1. Transform mean and std to the [0,1] space for Beta distribution
        mu_target_unclipped = (mean - log_mins) / log_ranges
        mu_target = jnp.clip(mu_target_unclipped, epsilon, 1.0 - epsilon)

        var_target_scaled_unclipped = (std / log_ranges)**2
        # Ensure variance is valid: V < M(1-M) and V > 0
        max_allowable_var = mu_target * (1.0 - mu_target) - epsilon 
        var_target_scaled = jnp.clip(var_target_scaled_unclipped, epsilon, max_allowable_var)

        # 2. Calculate Beta parameters a and b using method of moments
        # kappa = a + b = M(1-M)/V - 1
        # Add epsilon to var_target_scaled in denominator to prevent division by zero if std is extremely small
        kappa = mu_target * (1.0 - mu_target) / (var_target_scaled + epsilon) - 1.0
        kappa = jnp.maximum(kappa, epsilon) # Ensure kappa (a+b) is positive

        a_params = mu_target * kappa
        b_params = (1.0 - mu_target) * kappa

        # Ensure a and b are positive
        a_params = jnp.maximum(a_params, epsilon)
        b_params = jnp.maximum(b_params, epsilon)

        # 3. Generate Beta samples (shape will be (NUM_SAMPLES, DIM))
        # a_params and b_params are (DIM,), they will broadcast.
        beta_draws = jax.random.beta(subkey, a_params, b_params, shape=(NUM_SAMPLES, DIM))

        # 4. Scale Beta samples to the actual log-space range
        samples = log_mins + log_ranges * beta_draws
        # --- End of Beta sampling ---
        
        costs    = batch_cost(samples)

        elite_k  = int(NUM_SAMPLES * ELITE_FRAC)
        elite_ix = jnp.argsort(costs)[:elite_k]
        elite    = samples[elite_ix]

        mean = jnp.mean(elite, axis=0)
        std  = jnp.std(elite, axis=0) + 1e-6   # avoid collapse

        current_iter_best_cost = costs[elite_ix[0]]
        if current_iter_best_cost < best_cost:
            best_cost  = current_iter_best_cost
            best_gains = samples[elite_ix[0]]

        # Print progress (this will cause synchronization and slow down execution)
        print(f"Iteration {i + 1}/{NUM_ITER}: Best cost in iteration: {current_iter_best_cost.item():.4e}, Overall best: {best_cost.item():.4e}")

    Kp, Ki, Kd = jnp.split(10 ** best_gains, 3)
    return (Kp, Ki, Kd), best_cost


# ---------------------------------------------------------------------#
#  Main                                                                #
# ---------------------------------------------------------------------#
def main():
    global key

    key, t_knots, coefs = make_reference_spline(key)
    key, wl             = make_wave(key)

    T, dt = HPARAMS["T"], HPARAMS["dt"]
    ts    = jnp.arange(0.0, T + dt, dt)
    integral_abs_limit = HPARAMS["integral_abs_limit"]

    # ------------------ run CEM optimiser ---------------------------
    start = time.perf_counter()
    (Kp_best, Ki_best, Kd_best), best_cost = cem_optimize(
        key, init_mean, init_std, ts, wl, t_knots, coefs, integral_abs_limit
    )
    elapsed = time.perf_counter() - start

    # ------------------ print results -------------------------------
    print("\n=== best gain triplet (CEM) ===")
    kp_str = ", ".join([f"{k:.4e}" for k in Kp_best])
    ki_str = ", ".join([f"{k:.4e}" for k in Ki_best])
    kd_str = ", ".join([f"{k:.4e}" for k in Kd_best])
    print(f"Kp=[{kp_str}]")
    print(f"Ki=[{ki_str}]")
    print(f"Kd=[{kd_str}]")
    print(f"RMS cost = {best_cost:.4e}")
    print(f"Elapsed  = {elapsed:.2f} s")

    # ------------------ persist full record -------------------------
    result = dict(
        t_knots=t_knots,
        coefs=coefs,
        wave=wl,
        ts=ts,
        Kp=Kp_best,
        Ki=Ki_best,
        Kd=Kd_best,
        cost=best_cost,
        time_used=elapsed,
        cem=dict(samples=NUM_SAMPLES, elite_frac=ELITE_FRAC,
                 iterations=NUM_ITER, init_mean=init_mean, init_std=init_std),
    )

    script_root = Path(__file__).resolve().parent
    project_root = script_root.parent.parent.parent
    out_dir = project_root / "data" / "testing_results" / "rvg" / "pid_cem_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"seed={args.seed}.pkl"
    with out_file.open("wb") as f:
        pickle.dump(result, f)

    try:
        rel = out_file.relative_to(Path.cwd())
        print(f"\nResults written to {rel}")
    except ValueError:
        print(f"\nResults written to {out_file}")

if __name__ == "__main__":
    main()
