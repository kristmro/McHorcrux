#!/usr/bin/env python3
"""
Positive-gain LQR / PID tuning for the 3-DOF RVG vessel model

"""

# ------------------------------------------------------------------#
# Imports
# ------------------------------------------------------------------#
import argparse, time, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from jax.flatten_util import ravel_pytree
import scipy.optimize

# === JAX-friendly SciPy minimise wrapper ===========================#
def minimize(fun, x0, method="Nelder-Mead", tol=None, options=None):
    x0_flat, unravel = ravel_pytree(x0)

    def fun_wrap(x_flat):
        return float(fun(unravel(x_flat)))

    res = scipy.optimize.minimize(
        fun_wrap, x0_flat, method=method, jac=None,
        tol=tol, options=options or {}
    )
    res.x = unravel(res.x)
    return res
# ===================================================================#

# Project-local helpers ---------------------------------------------#
from jax_core.meta_adaptive_ctrl.rvg.dynamics import (
    disturbance, plant_6 as plant, prior_3dof_nom as prior,
)
from jax_core.simulator.waves.wave_load_jax_jit import wave_load
from jax_core.utils import random_ragged_spline, spline
# ------------------------------------------------------------------#

# CLI & precision ---------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, nargs="?", default=0)
parser.add_argument("--use_x64", action="store_true")
args = parser.parse_args()

key = jax.random.PRNGKey(args.seed)
np.random.seed(args.seed)
if args.use_x64:
    jax.config.update("jax_enable_x64", True)

# Hyper-parameters --------------------------------------------------#
HP = dict(
    T=10.0, dt=1e-2,
    num_knots=6, poly_orders=(9,9,6), deriv_orders=(4,4,2),
    min_step=(-3.,-3.,-jnp.pi/3), max_step=(3.,3.,jnp.pi/3),
    integral_abs_limit=1e5,
)
NUM_DOF = 3

Q_POS = jnp.ones(NUM_DOF) * 1.0e4
Q_VEL = jnp.ones(NUM_DOF) * 5.0e1
Q_INT = jnp.zeros(NUM_DOF)
R_ACT = jnp.ones(NUM_DOF) * 1.0e-7

LOG_MIN, LOG_MAX = -6.0, 7.0         #  gains ∈ [1e-6, 1e7]

# ------------------------------------------------------------------#
# Generators
# ------------------------------------------------------------------#
def make_reference(k):
    k, sub = jax.random.split(k)
    t_knots, _, coefs = random_ragged_spline(
        sub, HP["T"], HP["num_knots"],
        HP["poly_orders"], HP["deriv_orders"],
        jnp.asarray(HP["min_step"]), jnp.asarray(HP["max_step"]),
        jnp.array([-jnp.inf,-jnp.inf,-jnp.pi/3]),
        jnp.array([ jnp.inf, jnp.inf, jnp.pi/3]),
    )
    return k, t_knots, coefs

def make_wave(k):
    k, sub = jax.random.split(k)
    wl = disturbance((0.0,15.0,0.0), sub)
    wl = jax.tree.map(jnp.real, wl)
    return k, wl

# ------------------------------------------------------------------#
# JIT-compiled simulator: returns `q`, `r`, cost
# ------------------------------------------------------------------#
def simulate(ts, wl, t_knots, coefs,
             Kp, Ki, Kd, limit):
    KpM, KiM, KdM = map(jnp.diag, (Kp, Ki, Kd))

    def ref(t):
        r = jnp.array([spline(t, t_knots, c) for c in coefs])
        return r.at[2].set(jnp.mod(r[2]+jnp.pi, 2*jnp.pi)-jnp.pi)

    vel = jax.jacfwd(ref)

    def ode(state, t, u):
        q, dq = state
        f = wave_load(t, q, wl)
        dq, ddq = plant(q, dq, u, f)
        return dq, ddq

    dt = ts[1]-ts[0]

    def step(carry, t):
        t0,q0,dq0,u0,I0,J0 = carry
        qs,dqs = odeint(ode, (q0,dq0), jnp.array([t0,t]), u0)
        q,dq = qs[-1], dqs[-1]

        r0  = ref(t0)
        r   = ref(t)
        e,de = q-r, dq-vel(t)
        I   = I0 + 0.5*(e + (q0-r0))*dt

        _,_,_,R = prior(q,dq)
        τ = -(KpM@e + KiM@I + KdM@de)
        u = jnp.linalg.solve(R, τ)

        J_step = dt*(jnp.sum(Q_INT*I**2)+jnp.sum(Q_POS*e**2)+
                     jnp.sum(Q_VEL*de**2)+jnp.sum(R_ACT*τ**2))
        out = (q, r)
        new_c = (t,q,dq,u,I,J0+J_step)
        return new_c, out

    r0 = ref(ts[0]); dr0 = vel(ts[0])
    init = (ts[0], r0, dr0,
            jnp.zeros(NUM_DOF), jnp.zeros(NUM_DOF), 0.0)
    final, outs = jax.lax.scan(step, init, ts[1:])
    q_hist, r_hist = outs
    q_hist = jnp.vstack((r0, q_hist))
    r_hist = jnp.vstack((r0, r_hist))
    cost   = jnp.nan_to_num(final[-1], nan=jnp.inf)
    return q_hist, r_hist, cost

simulate_jit = jax.jit(simulate, static_argnames=["limit"])

# Cost-only wrapper for the optimiser 
def make_objective(ts, wl, t_knots, coefs, limit):
    def obj(log_g):
        log_g = jnp.clip(log_g, LOG_MIN, LOG_MAX)
        gains = 10.0 ** log_g
        Kp,Ki,Kd = jnp.split(gains,3)
        _,_,J = simulate_jit(ts,wl,t_knots,coefs,Kp,Ki,Kd,limit)
        return J
    return obj


def main():
    global key
    key,t_knots,coefs = make_reference(key)
    key,wl           = make_wave(key)
    ts = jnp.arange(0.0, HP["T"]+HP["dt"], HP["dt"])
    limit = HP["integral_abs_limit"]

    x0 = jnp.log10(jnp.array([3000.,3000.,3000., 2.,2.,2., 3000.,3000.,3000.]))

    objective = make_objective(ts, wl, t_knots, coefs, limit)
    print(" compiling objective …")
    _ = objective(x0).block_until_ready()
    print(" compile done, starting Nelder–Mead")

    start = time.time()
    res = minimize(
        objective, x0, method="Nelder-Mead",
        options=dict(maxiter=3000, xatol=1e-3, fatol=1e-6, adaptive=False, disp=True)
    )
    print("Optimisation done in", time.time()-start, "s")
    gains = 10.0 ** res.x
    print("Gains:", gains)
    Kp,Ki,Kd = jnp.split(gains,3)

    # ---------- JIT trajectory with best gains --------------------
    q_hist, r_hist, J_val = simulate_jit(ts, wl, t_knots, coefs,
                                         Kp, Ki, Kd, limit)

    # ---------- Plot ----------------------------------------------
    t = np.asarray(ts)
    q_np = np.asarray(q_hist); r_np = np.asarray(r_hist)
    lbl = ["Surge X [m]", "Sway Y [m]", "Yaw ψ [rad]"]
    fig,axs = plt.subplots(3,1, figsize=(8,7), sharex=True)
    for i,ax in enumerate(axs):
        ax.plot(t, r_np[:,i], "--", label="reference")
        ax.plot(t, q_np[:,i], label="vessel")
        ax.set_ylabel(lbl[i]); ax.grid(True)
        if i==0: ax.legend()
    axs[-1].set_xlabel("time [s]")
    plt.suptitle("Tracking with optimal positive gains")
    plt.tight_layout()
    plt.show()

    # ---------- Print & save --------------------------------------
    f = lambda arr: ", ".join(f"{k:.4e}" for k in arr)
    print("\n=== Optimal positive gains (JIT trajectory) ===")
    print(f"Kp = [{f(Kp)}]")
    print(f"Ki = [{f(Ki)}]")
    print(f"Kd = [{f(Kd)}]")
    print(f"Best cost J* = {res.fun:.4e}  (validation J = {J_val:.4e})")
    print(f"Iterations    = {res.nit}")

    record = dict(Kp=np.asarray(Kp), Ki=np.asarray(Ki), Kd=np.asarray(Kd),
                  cost=float(res.fun), iterations=int(res.nit), seed=args.seed)
    out_dir = (Path(__file__).resolve().parent.parent.parent /
               "data/testing_results/rvg/dfo_lqr_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed={args.seed}.pkl"
    with out_file.open("wb") as f: pickle.dump(record, f)
    print("Results →", out_file)

if __name__ == "__main__":
    main()
