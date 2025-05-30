"""
rvg_jax.py

Implements a differentiable 6 Degrees-of-Freedom (DOF) dynamic positioning (DP) vessel model using JAX.
Loads vessel parameters and computes the state derivatives for use in simulations and gradient-based meta-learning.

Author: Kristian Magnus Roen (adapted from Jan-Erik Hygen)
Date:   2025-03-17
"""

import json
import jax.numpy as jnp
from jax_core.utils import Rz, J, Smat  # Ensure these are pure functions as well

def load_rvg_parameters(config_file="data/vessel_data/rvg/rvg.json"):
    """
    Load the vessel parameters from a JSON file and construct the system matrices.
    
    Returns a dictionary with keys:
      - Mrb: Rigid-body mass matrix.
      - Ma: Added mass matrix (initially selected at a specific index).
      - M: Total mass matrix (Mrb + Ma).
      - Minv: Inverse of total mass matrix.
      - Dp: Hydrodynamic damping matrix part from potential flow.
      - Dv: Viscous damping matrix.
      - D: Total damping matrix (Dp + Dv), with an adjustment at index (3,3).
      - G: Restoring matrix.
    """
    with open(config_file, "r") as f:
        data = json.load(f)

    Mrb = jnp.asarray(data["MRB"])
    Ma = jnp.asarray(data["A"])[:, :, 30, 0]
    M = Mrb + Ma
    Minv = jnp.linalg.inv(M)

    Dp = jnp.asarray(data["B"])[:, :, 30, 0]
    Dv = jnp.asarray(data["Bv"])
    D = Dp + Dv

    G = jnp.asarray(data["C"])[:, :, 0, 0]

    params = {
        "Mrb": Mrb,
        "Ma": Ma,
        "M": M,
        "Minv": Minv,
        "Dp": Dp,
        "Dv": Dv,
        "D": D,
        "G": G,
    }
    return params

def rvg_x_dot(x, Uc, betac, tau, params):
    """
    Compute the time derivative of the state for the 6 DOF vessel model.
    
    Parameters:
      x: A 12-element state vector [eta, nu], where eta (positions/orientations) 
         and nu (velocities) are 6-element vectors.
      Uc: Current speed.
      betac: Current direction (in radians).
      tau: External forces/torques (6-element vector).
      params: Dictionary containing system matrices (from load_csad_parameters or set_hydrod_parameters).
    
    Returns:
      dx/dt: A 12-element vector combining eta_dot and nu_dot.
    """
    eta = x[:6]
    nu  = x[6:]
    
    # Compute the current component in the inertial frame
    nu_cn = Uc * jnp.array([jnp.cos(betac), jnp.sin(betac), 0.0])
    
    # Rotate current into the body-fixed frame using the yaw angle (eta[-1])
    nu_c = jnp.transpose(Rz(eta[-1])) @ nu_cn
    # Insert zeros for the rotational DOFs (assuming indices 3,4,5 correspond to rotations)
    nu_c = jnp.insert(nu_c, jnp.array([3, 3, 3]), 0)
    
    # Relative velocity (subtracting current effects)
    nu_r = nu - nu_c

    # Calculate the time derivative of nu_c_b
    dnu_cb = -Smat([0.0, 0.0, nu[-1]]) @ jnp.transpose(Rz(eta[-1])) @ nu_cn
    dnu_cb = jnp.insert(dnu_cb, jnp.array([2, 2, 2]), 0)
    # Kinematics: transform body velocities to inertial rates (using a transformation matrix J)
    eta_dot = J(eta) @ nu
    
    # Kinetics: acceleration computed from external forces, damping, and restoring forces
    nu_dot = params["Minv"] @ (tau - params["D"] @ nu_r - params["G"] @ eta + params["Ma"] @ dnu_cb)
    
    return jnp.concatenate([eta_dot, nu_dot])

def set_hydrod_parameters(freq, params, config_file="data/vessel_data/rvg/rvg.json"):
    """
    Update hydrodynamic parameters for a given frequency (or per-DOF frequencies).
    
    Parameters:
      freq: A scalar frequency or a 1D array of length 6 (one per DOF).
      params: Dictionary containing initial parameters (including config_file).
    
    Returns:
      new_params: A new parameters dictionary with updated hydrodynamic matrices:
                  - Ma, Dp, M, Minv, and D.
    """
    # Ensure freq is a JAX array
    if not isinstance(freq, (list, tuple, jnp.ndarray)):
        freq = jnp.array([freq])
    else:
        freq = jnp.array(freq)

    # Check dimensions: if multiple frequencies, expect one per DOF (6)
    if freq.ndim == 1 and (freq.shape[0] > 1 and freq.shape[0] != 6):
        raise ValueError(f"freq must be a scalar or have shape (6,), got shape {freq.shape}.")

    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    freqs = jnp.asarray(config_data['freqs'])
    
    if freq.ndim == 1 and freq.shape[0] == 1:
        # Single frequency: choose index minimizing absolute difference
        freq_indx = jnp.argmin(jnp.abs(freqs - freq))
    else:
        # Multiple frequencies: per DOF index (assumes freq is (6,))
        freq_indx = jnp.argmin(jnp.abs(freqs - freq[:, None]), axis=1)
    
    all_dof = jnp.arange(6)
    # Gather new added mass and damping matrices using the computed indices
    Ma = jnp.asarray(config_data['A'])[:, all_dof, freq_indx, 0]
    Dp = jnp.asarray(config_data['B'])[:, all_dof, freq_indx, 0]
    
    M = params["Mrb"] + Ma
    Minv = jnp.linalg.inv(M)
    D = params["Dv"] + Dp

    new_params = dict(params)
    new_params.update({
        "Ma": Ma,
        "Dp": Dp,
        "M": M,
        "Minv": Minv,
        "D": D,
    })
    return new_params



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