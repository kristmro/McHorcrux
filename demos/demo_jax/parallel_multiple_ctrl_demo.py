#!/usr/bin/env python3
"""
PID Parallel Simulation Demo - evaluate a few random gain triplets in parallel
to demonstrate JAX's vmap capability.

We adapt the single-trajectory PID simulator from *test_all.py* and run it
in parallel with **jax.vmap**.  All runs share the same reference spline
and wave realisation; only the gains vary.  The cost function is the
root-mean-square of the stacked position & velocity tracking errors over
the full simulation horizon.

Run:
    python parallel_multiple_ctrl_demo.py 42 --use_x64

The script prints the results for each simulated candidate and writes a
pickled results dictionary to:
    data/testing_results/rvg/parallel_muliple_ctrl_demo/seed=<seed>.pkl

    Author: Kristian Magnus Roen
"""

# ---------------------------------------------------------------------#
#  Imports                                                             #
# ---------------------------------------------------------------------#
import argparse
import pickle
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

# Project‑local helpers – paths identical to *test_all.py*
from jax_core.meta_adaptive_ctrl.rvg.dynamics import (
    disturbance, plant_6 as plant, prior_3dof_nom as prior,
)
from jax_core.simulator.waves.wave_load_jax_jit import wave_load
from jax_core.utils import (
    random_ragged_spline,
    spline,
)

# ---------------------------------------------------------------------#
#  CLI & precision flags                                               #
# ---------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help="seed for pseudo‑random number generation")
parser.add_argument("--use_x64", action="store_true", help="use 64‑bit precision (recommended)")
args = parser.parse_args()

if args.use_x64:
    jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------#
#  Hyper‑parameters (copied from *test_all.py*)                        #
# ---------------------------------------------------------------------#
HPARAMS = {
    # Reference trajectories
    "T": 100.0,
    "dt": 1e-3,
    "num_knots": 6,
    "poly_orders": (9, 9, 6),
    "deriv_orders": (4, 4, 2),
    "min_step": (-3.0, -3.0, -jnp.pi / 3),
    "max_step": (3.0, 3.0, jnp.pi / 3),
}

NUM_DOF = 3       # 3 controlled DOF (X, Y, Yaw)
NUM_RUNS = 1000    # how many random gain triplets to evaluate (you can increase this to upto nearly 3000 for a gpu with 8GB memory)

key = jax.random.PRNGKey(args.seed)

# ---------------------------------------------------------------------#
#  Reference spline (single trajectory)                                #
# ---------------------------------------------------------------------#

def make_reference_spline(key):
    """Generate **one** random spline reference as in *test_all.py*."""
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
    """Draw one plausible wave realisation – identical for every PID run."""
    key, subkey = jax.random.split(key)
    # Moderate sea state in the middle of the *test_all.py* distribution
    hs, tp, wdir =3.0, 12.0, 0.0
    wl = disturbance((hs, tp, wdir), subkey)
    return key, wl

# ---------------------------------------------------------------------#
#  PID closed‑loop simulator (single trajectory)                       #
# ---------------------------------------------------------------------#

def simulate_pid(ts, wl, t_knots, coefs, Kp, Ki, Kd): # Kp, Ki, Kd are (NUM_DOF,) vectors
    """Simulate the 3‑DOF subsystem with diagonal PID gains for each DOF."""

    # Kp, Ki, Kd are now vectors of shape (NUM_DOF,)
    Kp_mat = jnp.diag(Kp)
    Ki_mat = jnp.diag(Ki)
    Kd_mat = jnp.diag(Kd)

    # ----- reference spline -----------------------------------------
    def reference(t):
        r = jnp.array([spline(t, t_knots, c) for c in coefs])
        # Bound yaw to [‑π, π] to avoid wrap‑around spikes in the cost.
        r = r.at[2].set(jnp.mod(r[2] + jnp.pi, 2 * jnp.pi) - jnp.pi)
        return r

    # Pre‑compute derivatives with automatic differentiation
    vel_fn = jax.jacfwd(reference)
    acc_fn = jax.jacfwd(vel_fn)

    def ref_derivatives(t):
        r = reference(t)
        dr = vel_fn(t)
        ddr = jnp.nan_to_num(acc_fn(t), nan=0.0)
        return r, dr, ddr

    # ----- plant ODE w/ zero‑order hold -----------------------------
    def ode(state, t, u):
        q, dq = state
        f_ext = wave_load(t, q, wl)
        dq, ddq = plant(q, dq, u, f_ext)
        return dq, ddq

    # ----- scan step -------------------------------------------------
    def step(carry, t):
        t_prev, q_prev, dq_prev, u_prev, I_prev = carry
        qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev) #adaptive step size
        q, dq = qs[-1], dqs[-1]

        r, dr, _ = ref_derivatives(t)
        e, de = q - r, dq - dr
        dt = t - t_prev
        # Trapezoidal rule for the integral term
        I = I_prev + 0.5 * (e + (q_prev - r)) * dt

        M_mat, D, G, R = prior(q, dq)
        τ = -(Kp_mat @ e + Ki_mat @ I + Kd_mat @ de)
        u = jnp.linalg.solve(R, τ)

        new_carry = (t, q, dq, u, I)
        out_slice = (q, dq, u, τ, r, dr)
        return new_carry, out_slice

    # ----- initial conditions ---------------------------------------
    t0 = ts[0]
    r0, dr0, _ = ref_derivatives(t0)
    q0, dq0 = r0, dr0
    I0 = jnp.zeros_like(q0)
    u0 = jnp.zeros_like(q0)

    carry, outputs = jax.lax.scan(step, (t0, q0, dq0, u0, I0), ts[1:])
    q, dq, u, τ, r, dr = outputs

    # Prepend t0 state to full‑length arrays
    q = jnp.vstack((q0, q))
    dq = jnp.vstack((dq0, dq))
    r = jnp.vstack((r0, r))
    dr = jnp.vstack((dr0, dr))

    # RMS cost over stacked position & velocity errors
    e = jnp.concatenate((q - r, dq - dr), axis=-1)
    rms_cost = jnp.sqrt(jnp.mean(jnp.sum(e ** 2, axis=-1)))
    return rms_cost

# Make the simulator JIT‑compatible – the body will be vmapped.
simulate_pid_jit = jax.jit(simulate_pid)

# ---------------------------------------------------------------------#
#  Random gain sampling                                                #
# ---------------------------------------------------------------------#

def sample_pid_gains(key, n_samples):
    """Draw log‑uniform random gains for each DOF inside plausibly‑stable ranges."""
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    # log10 ranges chosen from empirical grid in *test_all.py*
    # Sample NUM_DOF gains for each Kp, Ki, Kd for each of the n_samples runs
    Kp = 10 ** jax.random.uniform(subkey1, (n_samples, NUM_DOF), minval=1.0, maxval=5.0)
    Ki = 10 ** jax.random.uniform(subkey2, (n_samples, NUM_DOF), minval=1.0, maxval=3.7)
    Kd = 10 ** jax.random.uniform(subkey3, (n_samples, NUM_DOF), minval=1.0, maxval=4.7)
    return key, Kp, Ki, Kd

# ---------------------------------------------------------------------#
#  Main                                                                #
# ---------------------------------------------------------------------#

def main():
    global key
    # Reference & disturbance
    key, t_knots, coefs = make_reference_spline(key)
    key, wl = make_wave(key)

    # Time grid
    T, dt = HPARAMS["T"], HPARAMS["dt"]
    ts = jnp.arange(0.0, T + dt, dt)

    # Random PID gains
    key, Kp, Ki, Kd = sample_pid_gains(key, NUM_RUNS)

    # Vectorised evaluation (parallel on GPU/TPU/CPU via XLA)
    start_time = time.perf_counter()
    costs = jax.vmap(
        simulate_pid_jit,
        in_axes=(None, None, None, None, 0, 0, 0),
    )(ts, wl, t_knots, coefs, Kp, Ki, Kd)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Print results for all simulated candidates
    print("\n=== Parallel Simulation Results ===")
    if NUM_RUNS > 10:
        # Sort indices by cost in ascending order
        top_indices = jnp.argsort(costs)[:5]
        print("Top 5 Runs by Cost:")
        for rank, i in enumerate(top_indices, start=1):
            kp_str = ", ".join([f"{k:.3e}" for k in Kp[i]])
            ki_str = ", ".join([f"{k:.3e}" for k in Ki[i]])
            kd_str = ", ".join([f"{k:.3e}" for k in Kd[i]])
            print(
                f"Rank #{rank}: cost={costs[i]:.4e}  Kp=[{kp_str}]  Ki=[{ki_str}]  Kd=[{kd_str}]"
            )
    else:
        for i in range(NUM_RUNS):
            # Kp[i], Ki[i], Kd[i] are arrays of shape (NUM_DOF,)
            kp_str = ", ".join([f"{k:.3e}" for k in Kp[i]])
            ki_str = ", ".join([f"{k:.3e}" for k in Ki[i]])
            kd_str = ", ".join([f"{k:.3e}" for k in Kd[i]])
            print(
                f"Run #{i+1:2d}: cost={costs[i]:.4e}  Kp=[{kp_str}]  Ki=[{ki_str}]  Kd=[{kd_str}]"
            )

    print(f"\nElapsed time: {elapsed_time:.2f} seconds")
    # Save full result for offline analysis
    result = {
        "t_knots": t_knots,
        "coefs": coefs,
        "wave": wl,
        "ts": ts,
        "Kp": Kp,
        "Ki": Ki,
        "Kd": Kd,
        "costs": costs,
        "time_used": elapsed_time,
        "num_runs": NUM_RUNS,
    }

    # -----------------------------------------------------------------
    #  Robust output path handling
    # -----------------------------------------------------------------
    script_root = Path(__file__).resolve().parent
    # Determine project root by navigating up from the script's location
    project_root = script_root.parent.parent # Corrected based on actual file path relative to project root
    out_dir = project_root / "data" / "testing_results" / "rvg" / "parallel_multiple_ctrl_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"seed={args.seed}.pkl"
    with out_file.open("wb") as f:
        pickle.dump(result, f)

    # Always produce a usable path – fall back to absolute path if
    # relative_to() cannot be computed (e.g. cwd outside repo).
    try:
        rel = out_file.relative_to(Path.cwd())
        print(f"\nResults written to {rel}")
    except ValueError:
        print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
