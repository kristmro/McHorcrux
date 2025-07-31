#!/usr/bin/env python3
"""
PID CEM Tuning - optimise nine log-gains with the cross-entropy method
and report the best candidate.

Usage
-----
    python pid_cem_tuning.py 42 --use_x64

Author
------
Kristian Magnus Roen - CEM version prepared May 2025
"""

# ---------------------------------------------------------------------#
#  Imports                                                             #
# ---------------------------------------------------------------------#
import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpy as np  # only for seed reproducibility in other libs

# Project-local helpers – paths identical to *test_all.py*
from jax_core.meta_adaptive_ctrl.csad.dynamics import (
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
    "min_step": (-0.9, -0.9, -jnp.pi / 5),
    "max_step": (0.9, 0.9,  jnp.pi / 5),
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
    hs, tp, wdir = 3.0*(1/90), 15.0*(1/90)**0.5, 0.0        # moderate sea state
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

    # adding a gain to the position error to make it more pronounced
    e = jnp.concatenate((q - r, dq - dr), axis=-1)
    e
    rms_cost = jnp.sqrt(jnp.mean(jnp.sum(e**2, axis=-1)))
    e = e.at[:, -1].set(e[:, -1] * 9.0) # heading error because it is in the e-1 region instead of 0
    rms_cost = 1000 * rms_cost  # scale for better visibility in plots
    safe_rms_cost = jnp.nan_to_num(rms_cost, nan=jnp.inf)
    return q, r, safe_rms_cost

simulate_pid_jit = jax.jit(simulate_pid, static_argnames=['integral_abs_limit']) # XLA-compile once

# ---------------------------------------------------------------------#
#  Cross-Entropy Method optimiser                                      #
# ---------------------------------------------------------------------#
NUM_SAMPLES = 2000         # samples per generation
ELITE_FRAC  = 0.15        # top 10 % survive
NUM_ITER    = 15          # generations
DIM         = NUM_DOF * 3 # 9 log-gain parameters

init_mean = jnp.array([0.2, 0.3, 0.1,  # log10 Kp
                       0.2, 0.1, 0.1,  # log10 Ki
                       0.1, 0.1, 0.1]) # log10 Kd
init_std  = jnp.ones(DIM) * 2.0        # one order of magnitude spread

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

    # Log-space bounds for PID gains
    log_mins_kp = jnp.full((NUM_DOF,), -3.0)
    log_maxs_kp = jnp.full((NUM_DOF,), 3.5)
    log_mins_ki = jnp.full((NUM_DOF,), -3.0)
    log_maxs_ki = jnp.full((NUM_DOF,), 3.0)
    log_mins_kd = jnp.full((NUM_DOF,), -3.0)
    log_maxs_kd = jnp.full((NUM_DOF,), 3.5)
    
    log_mins = jnp.concatenate([log_mins_kp, log_mins_ki, log_mins_kd])
    log_maxs = jnp.concatenate([log_maxs_kp, log_maxs_ki, log_maxs_kd])
    log_ranges = log_maxs - log_mins
    
    epsilon = 1e-6 # Small constant for numerical stability
    all_best_costs_per_iteration = [] # Store best cost for each iteration

    @jax.jit
    def batch_simulate_and_cost(log10_batch):
        Kp_batch, Ki_batch, Kd_batch = jnp.split(10 ** log10_batch, 3, axis=-1)

        # vmap over simulate_pid_jit which now returns (q, r, cost)
        # We only need the cost for CEM optimization logic here.
        # The full trajectories q and r are not needed for each sample within CEM.
        _qs, _rs, costs = jax.vmap(
            simulate_pid_jit,
            in_axes=(None, None, None, None, 0, 0, 0, None)
        )(ts, wl, t_knots, coefs, Kp_batch, Ki_batch, Kd_batch, integral_abs_limit)
        return costs

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
        
        costs = batch_simulate_and_cost(samples) # Use the new batch simulation function

        elite_k  = int(NUM_SAMPLES * ELITE_FRAC)
        elite_ix = jnp.argsort(costs)[:elite_k]
        elite    = samples[elite_ix]

        mean = jnp.mean(elite, axis=0)
        std  = jnp.std(elite, axis=0) + 1e-6   # avoid collapse

        current_iter_best_cost = costs[elite_ix[0]]
        if current_iter_best_cost < best_cost:
            best_cost  = current_iter_best_cost
            best_gains = samples[elite_ix[0]]
        
        all_best_costs_per_iteration.append(best_cost) # Store the overall best cost so far for this iteration

        # Print progress (this will cause synchronization and slow down execution)
        print(f"Iteration {i + 1}/{NUM_ITER}: Best cost in iteration: {current_iter_best_cost.item():.4e}, Overall best: {best_cost.item():.4e}")

    Kp, Ki, Kd = jnp.split(10 ** best_gains, 3)
    return (Kp, Ki, Kd), best_cost, all_best_costs_per_iteration


# ---------------------------------------------------------------------#
#  Plotting functions                                                  #
# ---------------------------------------------------------------------#
def plot_loss_over_iterations(costs_history, seed, figures_dir):
    """Plots the loss over iterations and saves the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(costs_history) + 1), costs_history, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Best RMS Cost (log scale)")
    plt.yscale('log')
    plt.title(f"CEM Optimization: Best Cost per Iteration (Seed {seed})")
    plt.grid(True, which="both", ls="-")
    fig_path = figures_dir / f"loss_over_iterations_seed_{seed}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Loss plot saved to {fig_path.relative_to(Path.cwd())}")

def plot_simulation_trip(ts, q_actual, r_reference, seed, figures_dir, gains_label="Best Gains"):
    """Plots the simulated trip (actual vs. reference) and saves the figure."""
    num_dof = q_actual.shape[1]
    fig, axes = plt.subplots(num_dof, 1, figsize=(12, 4 * num_dof), sharex=True)
    if num_dof == 1: # Make axes an array even if it's a single subplot
        axes = [axes]
        
    dof_labels = ['X (surge)', 'Y (sway)', 'Yaw (psi)']

    for i in range(num_dof):
        axes[i].plot(ts, q_actual[:, i], label=f'Actual {dof_labels[i]}')
        axes[i].plot(ts, r_reference[:, i], label=f'Reference {dof_labels[i]}', linestyle='--')
        axes[i].set_ylabel("Position/Angle")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Simulated Trip with {gains_label} (Seed {seed})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    fig_path = figures_dir / f"simulation_trip_{gains_label.lower().replace(' ', '_')}_seed_{seed}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Simulation plot saved to {fig_path.relative_to(Path.cwd())}")


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

    # ------------------ setup figure directory ----------------------
    script_root = Path(__file__).resolve().parent
    project_root = script_root.parent.parent.parent
    figures_dir = project_root / "figures" / "csad" / "pid_tuning"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ run CEM optimiser ---------------------------
    start = time.perf_counter()
    (Kp_best, Ki_best, Kd_best), best_cost, costs_history = cem_optimize(
        key, init_mean, init_std, ts, wl, t_knots, coefs, integral_abs_limit
    )
    elapsed = time.perf_counter() - start

    # ------------------ plot loss history ---------------------------
    plot_loss_over_iterations(costs_history, args.seed, figures_dir)

    # ------------------ simulate with best gains for plotting -------
    q_final, r_final, _ = simulate_pid_jit(
        ts, wl, t_knots, coefs, Kp_best, Ki_best, Kd_best, integral_abs_limit
    )
    plot_simulation_trip(ts, q_final, r_final, args.seed, figures_dir, gains_label="Best CEM Gains")


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

    out_dir = project_root / "data" / "testing_results" / "csad" / "pid_cem_tuning"
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
