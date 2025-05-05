#!/usr/bin/env python3
"""
voyager_torch.py

Differentiable 6-DOF DP vessel model for *voyager* implemented in PyTorch.

Author : Kristian Magnus Røen
Updated: 2025-05-05  (55-point frequency grid, robust index selection)
"""

import json
import math
from typing import Union

import torch
from torch_core.simulator.vessels.vessel_torch import Vessel
from torch_core.utils import Rz_torch, J_torch


class VOYAGER(Vessel):
    """
    6-DOF rigid-body vessel model:

        η = [x, y, z, ϕ, θ, ψ]   (position & Euler angles, NED)
        ν = [u, v, w, p, q, r]   (linear & angular body velocities)
    """

    DOF = 6  # sanity-check constant

    # ────────────────────────────────────────────────────────────────
    # constructor
    # ────────────────────────────────────────────────────────────────
    def __init__(self,
                 dt: float,
                 method: str = "RK4",
                 config_file: str = "data/vessel_data/voyager/voyager.json",
                 dof: int = 6):

        # base-class setup (integrator, buffers, …)
        super().__init__(dt=dt, method=method, config_file=config_file, dof=dof)

        # read hydrodynamic JSON
        with open(config_file, "r") as f:
            data = json.load(f)

        # frequency grid (already 55 points after trimming)
        w = torch.as_tensor(data["freqs"], dtype=torch.float32)
        self._freqs = w  # exposed for WaveLoad etc.

        # choose the 1 Hz slice (≈ 6.283 rad s⁻¹) robustly
        target_w = 2 * math.pi
        idx = int((w - target_w).abs().argmin())   # → 40 on a 55-point table

        # rigid-body and added mass
        self._Mrb  = torch.tensor(data["MRB"], dtype=torch.float32)
        self._Ma   = torch.tensor(data["A"], dtype=torch.float32)[:, :, idx]
        self._M    = self._Mrb + self._Ma
        self._Minv = torch.inverse(self._M)

        # potential + viscous damping (B and Bv are full 6×6×55 tensors)
        self._Dp = torch.tensor(data["B"],  dtype=torch.float32)[:, :, idx]
        self._Dv = torch.tensor(data["Bv"], dtype=torch.float32)[:, :, idx]
        self._D  = self._Dp + self._Dv

        # hydrostatic restoring
        self._G = torch.tensor(data["C"], dtype=torch.float32)[:, :, 0]

    # ────────────────────────────────────────────────────────────────
    # dynamics
    # ────────────────────────────────────────────────────────────────
    def x_dot(self,
              x:     torch.Tensor,
              Uc:    Union[float, torch.Tensor],
              betac: Union[float, torch.Tensor],
              tau:   torch.Tensor) -> torch.Tensor:
        """
        Compute **ẋ = f(x, τ)** for the voyager vessel.

        Parameters
        ----------
        x      : (12,) tensor – state vector [η, ν]
        Uc     : scalar – current speed (m s⁻¹)
        betac  : scalar – current direction, inertial frame (rad)
        tau    : (6,) tensor – external forces & moments

        Returns
        -------
        x_dot  : (12,) tensor – time derivative [η̇, ν̇]
        """

        # 0) split state
        eta, nu = x[:6], x[6:]

        # 1) earth-fixed current velocity, NED frame
        cos_b = torch.cos(torch.as_tensor(betac, dtype=x.dtype, device=x.device))
        sin_b = torch.sin(torch.as_tensor(betac, dtype=x.dtype, device=x.device))
        nu_cn = Uc * torch.stack([cos_b, sin_b, torch.tensor(0.0, dtype=x.dtype, device=x.device)])

        # 2) rotate current to body frame
        R_yaw_T = Rz_torch(eta[-1]).T
        nu_c_lin = R_yaw_T @ nu_cn
        nu_c = torch.cat([nu_c_lin,
                          torch.zeros(3, dtype=x.dtype, device=x.device)])

        # 3) relative velocity
        nu_r = nu - nu_c

        # 4) kinematics
        eta_dot = J_torch(eta) @ nu

        # 5) kinetics
        nu_dot = self._Minv @ (tau - self._D @ nu_r - self._G @ eta)

        # 6) concatenate
        return torch.cat([eta_dot, nu_dot])
