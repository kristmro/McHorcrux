#!/usr/bin/env python3
"""
voyager_jax.py

Differentiable 6-DOF DP vessel model for *voyager* implemented in JAX.

Author : Kristian Magnus Roen
Updated : 2025-05-05  (55-point table, frequency-dependent Bv)
"""

import json
import math
import jax.numpy as jnp
from jax_core.utils import Rz, J  


# ───────────────────────────────────────────────────────────────
# 1.  parameter loader
# ───────────────────────────────────────────────────────────────
def load_voyager_parameters(config_file="data/vessel_data/voyager/voyager.json"):
    """
    Read *voyager.json* and build the hydrodynamic matrices.

    Returns a dict with keys:
        Mrb, Ma, M, Minv, Dp, Dv, D, G, freqs
    """

    with open(config_file) as f:
        data = json.load(f)

    freqs = jnp.asarray(data["freqs"])          # length 55 (ω = 0.8 … 39.2 rad s⁻¹)
    target_w = 2.0 * math.pi                    # 1 Hz  → closest index = 40
    idx = int(jnp.argmin(jnp.abs(freqs - target_w)))

    Mrb = jnp.asarray(data["MRB"])
    Ma  = jnp.asarray(data["A"])[:, :, idx]

    M    = Mrb + Ma
    Minv = jnp.linalg.inv(M)

    Dp = jnp.asarray(data["B"])[:, :, idx]
    Dv = jnp.asarray(data["Bv"])[:, :, idx]     # voyager: full 6 × 6 × 55 tensor
    D  = Dp + Dv

    G  = jnp.asarray(data["C"])[:, :, 0]

    return {
        "Mrb": Mrb, "Ma": Ma, "M": M, "Minv": Minv,
        "Dp": Dp,   "Dv": Dv, "D": D, "G": G,
        "freqs": freqs
    }


# ───────────────────────────────────────────────────────────────
# 2.  continuous-time dynamics   ẋ = f(x, τ)
# ───────────────────────────────────────────────────────────────
def voyager_x_dot(x, Uc, betac, tau, p):
    """
    State derivative for the *voyager* 6-DOF model.

    Parameters
    ----------
    x      : (12,)  – state   [η, ν]
    Uc     : scalar – current speed (m s⁻¹)
    betac  : scalar – current direction (rad, inertial)
    tau    : (6,)   – external forces & moments
    p      : dict   – matrices from `load_voyager_parameters`
    """

    eta, nu = x[:6], x[6:]

    # 1) earth-fixed current velocity (NED)
    nu_cn = Uc * jnp.array([jnp.cos(betac), jnp.sin(betac), 0.0])

    # 2) rotate to body frame (yaw only)
    nu_c_lin = Rz(eta[-1]).T @ nu_cn
    nu_c = jnp.concatenate([nu_c_lin, jnp.zeros(3)])

    # 3) relative velocity
    nu_r = nu - nu_c

    # 4) kinematics
    eta_dot = J(eta) @ nu

    # 5) kinetics
    nu_dot = p["Minv"] @ (tau - p["D"] @ nu_r - p["G"] @ eta)

    return jnp.concatenate([eta_dot, nu_dot])


# ───────────────────────────────────────────────────────────────
# 3.  re-tune hydrodynamics at arbitrary frequency(-ies)
# ───────────────────────────────────────────────────────────────
def set_hydrod_parameters(freq,
                          params,
                          config_file="data/vessel_data/voyager/voyager.json"):
    """
    Return a **new** params dict where `Ma`, `Dp`, `Dv`, `M`, `Minv`, `D`
    are taken at `freq` (scalar) *or* at one frequency per DOF (shape 6,).

    `params` must originate from `load_voyager_parameters` so `Mrb` and
    `Dv`-template are present.
    """

    with open(config_file) as f:
        cfg = json.load(f)

    A  = jnp.asarray(cfg["A"])   # (6,6,55)
    B  = jnp.asarray(cfg["B"])
    Bv = jnp.asarray(cfg["Bv"])
    freqs = jnp.asarray(cfg["freqs"])

    freq = jnp.asarray(freq, dtype=freqs.dtype)

    if freq.ndim == 0:                               # scalar
        idx = int(jnp.argmin(jnp.abs(freqs - freq)))
        Ma = A[:, :, idx]
        Dp = B[:, :, idx]
        Dv = Bv[:, :, idx]

    elif freq.shape == (6,):                         # one per DOF
        # gather per-DOF columns
        idx = jnp.argmin(jnp.abs(freqs - freq[:, None]), axis=1)  # (6,)
        ax = 2
        Ma = jnp.take_along_axis(A, idx[None, None, :], axis=ax)
        Dp = jnp.take_along_axis(B, idx[None, None, :], axis=ax)
        Dv = jnp.take_along_axis(Bv, idx[None, None, :], axis=ax)
    else:
        raise ValueError("freq must be scalar or shape (6,)")

    M    = params["Mrb"] + Ma
    Minv = jnp.linalg.inv(M)
    D    = Dp + Dv
    D    = D.at[3, 3].set(D[3, 3] * 2.0)

    new_p = dict(params)
    new_p.update({"Ma": Ma, "Dp": Dp, "Dv": Dv,
                  "M": M, "Minv": Minv, "D": D})
    return new_p



# # Example of how to make this
# """6-DOF vessel dynamics exported as JAX callables (f, B)."""

# import casadi as cs
# import jax.numpy as jnp
# from jaxadi import convert

# _p = load_csad_parameters()
# _M  = jnp.array(_p["M_rb"]) + jnp.array(_p["M_a"])
# _Mi = jnp.linalg.inv(_M)
# _D  = jnp.array(_p["D_lin"])
# _g  = jnp.array(_p["g_eta"])

# _eta = cs.SX.sym("eta", 6)          # pos/att
# _nu  = cs.SX.sym("nu",  6)          # vel
# _tau = cs.SX.sym("tau", 6)
# _x   = cs.vertcat(_eta, _nu)

# phi, _, psi = _eta[3], _eta[4], _eta[5]
# c, s = cs.cos(psi), cs.sin(psi)
# J = cs.vertcat(cs.horzcat(c, -s, 0),
#                cs.horzcat(s,  c, 0),
#                cs.horzcat(0,  0, 1))
# Jb = cs.blockcat([[J, cs.SX.zeros(3,3)],
#                   [cs.SX.zeros(3,3), cs.SX.eye(3)]])

# x_dot = cs.vertcat(Jb @ _nu,
#                    cs.mtimes(cs.SX(_Mi), (_tau - cs.mtimes(_D, _nu) - _g)))

# _f = cs.Function("f", [_x, _tau], [x_dot])
# _B = cs.Function("B", [_x], [cs.jacobian(x_dot, _tau)])

# f = convert(_f, compile=True)   # (x, tau) → ẋ
# B = convert(_B, compile=True)   # (x) → 6×6