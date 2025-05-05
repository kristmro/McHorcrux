#!/usr/bin/env python3
"""
csad_torch.py

Differentiable 6 DOF DP vessel model for CSAD implemented in PyTorch.
This class replicates the functionality of the original CSAD_DP_6DOF class but
operates on torch.Tensors.

Author: [Kristian Magnus Roen/ adapted from Jan-Erik Hygen]
Date:   2025-02-17
"""

import torch
import os
import json
from typing import Union
from  torch_core.simulator.vessels.vessel_torch import Vessel
from torch_core.utils import Rz_torch, J_torch

class CSAD_6DOF(Vessel):
    """
    Differentiable 6-DOF DP simulator model for **CSAD**.

    ── State vector (size 12) ─────────────────────────────────────────
        η = [x, y, z, ϕ, θ, ψ]      position & Euler angles (NED frame)
        ν = [u, v, w, p, q, r]      linear / angular body velocities
        x = [η, ν]

    The model equations are identical to Fossen’s standard form
        η̇  =  J(η) ν
        ν̇  =  M⁻¹ ( τ  –  D ν_r  –  G η + … )
    where `ν_r` is the *current-relative* velocity.
    ------------------------------------------------------------------
    """

    DOF = 6  # (class constant – handy for sanity checks)

    # ─────────────────────────────────────────────────────────────────
    #                       constructor
    # ─────────────────────────────────────────────────────────────────
    def __init__(self,
                 dt: float,
                 method: str = "RK4",
                 config_file: str ="data/vessel_data/csad/csad.json",
                 dof: int = 6):
        """
        Parameters
        ----------
        dt          : integration time-step  [s]
        method      : "Euler" | "RK4" – passed straight to the base `Vessel`
        config_file : *absolute* path to the JSON with hydrodynamic data
        dof         : degrees-of-freedom (fixed at 6 for a full rigid-body)
        """

        # 1) Resolve the JSON file path exactly the same way as the original
        # ------------------------------------------------------------------
        cfg_path = config_file # Use the provided config_file path directly

        # 2) Call the generic differentiable-vessel constructor (handles the
        #    integration loop, buffers, etc.)
        # ------------------------------------------------------------------
        super().__init__(dt=dt, method=method, config_file=cfg_path, dof=dof)

        # 3) Load all hydrodynamic matrices from disk
        # ------------------------------------------------------------------
        with open(cfg_path, "r") as f:
            data = json.load(f)

        # 4) Convert all matrices to PyTorch tensors
        # ------------------------------------------------------------------
        # Rigid-body mass
        self._Mrb = torch.tensor(data["MRB"], dtype=torch.float32)

        # Added mass  (slice  [:, :, 41]  keeps DOF×DOF matrix at freq-index 41)
        self._Ma  = torch.tensor(data["A"], dtype=torch.float32)[:, :, 41]

        # Total mass & its inverse
        self._M   = self._Mrb + self._Ma
        self._Minv= torch.inverse(self._M)

        # Potential + viscous damping
        self._Dp  = torch.tensor(data["B"],  dtype=torch.float32)[:, :, 41]
        self._Dv  = torch.tensor(data["Bv"], dtype=torch.float32)
        self._D   = self._Dp + self._Dv
        self._D[3, 3] = self._D[3, 3] * 2.0     # empirical roll-damping tweak (Hygen 2023)

        # Hydrostatic restoring
        self._G   = torch.tensor(data["C"], dtype=torch.float32)[:, :, 0]


    # ─────────────────────────────────────────────────────────────────
    #                       dynamics
    # ─────────────────────────────────────────────────────────────────
    def x_dot(self,
              x:    torch.Tensor,
              Uc:   Union[float, torch.Tensor],
              betac:Union[float, torch.Tensor],
              tau:  torch.Tensor) -> torch.Tensor:
        """
        Compute **ẋ = f(x, τ)** for the CSAD 6-DOF vessel.

        Parameters
        ----------
        x      : (12,) tensor – current state [η, ν]
        Uc     : scalar – current speed  (m/s)
        betac  : scalar – current direction, inertial frame (rad)
        tau    : (6,) tensor – external forces & moments  (thrusters, wind, …)

        Returns
        -------
        x_dot  : (12,) tensor – time derivative [η̇, ν̇]
        """

        # ---- 0) split state --------------------------------------------------
        eta, nu = x[:6], x[6:]

        # ---- 1) earth-fixed current velocity  ν_cn  (NED frame) ------------
        cos_b, sin_b = torch.cos(torch.as_tensor(betac, dtype=x.dtype, device=x.device)), \
                       torch.sin(torch.as_tensor(betac, dtype=x.dtype, device=x.device))
        nu_cn = Uc * torch.stack([cos_b, sin_b, torch.tensor(0.0, dtype=x.dtype, device=x.device)])

        # ---- 2) rotate current into body frame via yaw ----------------------
        R_yaw_T = Rz_torch(eta[-1]).transpose(0, 1)   # (3×3) transpose = inverse
        nu_c_lin = R_yaw_T @ nu_cn                    # linear part (u_c, v_c, w_c)
        nu_c = torch.cat([nu_c_lin,
                          torch.zeros(3, dtype=x.dtype, device=x.device)])  # pad for full 6-DOF

        # ---- 3) relative velocity -------------------------------------------
        nu_r = nu - nu_c                              # everything in body frame

        # ---- 4) kinematics  η̇ = J(η) ν --------------------------------------
        eta_dot = J_torch(eta) @ nu

        # ---- 5) kinetics  ν̇ = M⁻¹( τ – D ν_r – G η ) ------------------------
        #       (no added-mass cross-term for CSAD; matches original model)
        nu_dot = self._Minv @ (tau - self._D @ nu_r - self._G @ eta)

        # ---- 6) concatenate & return ----------------------------------------
        return torch.cat([eta_dot, nu_dot], dim=0)