import torch
import os
import json
from  torch_core.simulator.vessels.vessel_torch import Vessel
from torch_core.utils import Rz_torch, J_torch, Smat_torch

class RVG_6DOF(Vessel):
    """
    Differentiable 6-DOF DP simulator model for R/V Gunnerus
    (PyTorch replica of the original `RVG_DP_6DOF` NumPy class).
    """

    DOF = 6

    def __init__(self,
                 dt: float,
                 method: str = "RK4",
                 config_file: str ="data/vessel_data/rvg/rvg.json",
                 dof: int = 6):

        cfg_path = os.path.expanduser(config_file)

        super().__init__(dt=dt, method=method, config_file=cfg_path, dof=dof)

        with open(cfg_path, "r") as f:
            data = json.load(f)

        # --- rigid-body & added mass ------------------------------------------------
        self._Mrb = torch.tensor(data["MRB"], dtype=torch.float32)
        self._Ma  = torch.tensor(data["A"],   dtype=torch.float32)[:, :, 30, 0]
        self._M   = self._Mrb + self._Ma
        self._Minv= torch.inverse(self._M)

        # --- potential + viscous damping -------------------------------------------
        self._Dp  = torch.tensor(data["B"],  dtype=torch.float32)[:, :, 30, 0]
        self._Dv  = torch.tensor(data["Bv"], dtype=torch.float32)
        self._D   = self._Dp + self._Dv
        # --- hydrostatic restoring --------------------------------------------------
        self._G   = torch.tensor(data["C"], dtype=torch.float32)[:, :, 0, 0]

    # --------------------------------------------------------------------------
    #                             dynamics
    # --------------------------------------------------------------------------
    def x_dot(self,
              x: torch.Tensor,
              Uc: float | torch.Tensor,
              betac: float | torch.Tensor,
              tau: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the 12-state vector  [η, ν].

        x     : (12,) tensor → first 6 pos/orient  η, last 6 body vel ν
        Uc    : scalar – current speed
        betac : scalar – current direction (rad, inertial frame)
        tau   : (6,) tensor – external forces & moments
        """
        # ----- split state -----------------------------------------------------
        eta, nu = x[:6], x[6:]

        # ----- 1) current in NED ----------------------------------------------
        cos_b, sin_b = torch.cos(torch.as_tensor(betac, dtype=x.dtype, device=x.device)), \
                       torch.sin(torch.as_tensor(betac, dtype=x.dtype, device=x.device))
        nu_cn = Uc * torch.stack([cos_b, sin_b, torch.tensor(0.0, dtype=x.dtype, device=x.device)])

        # ----- 2) rotate to body frame ----------------------------------------
        R_yaw_T = Rz_torch(eta[-1]).transpose(0, 1)         # (3,3)
        nu_c_lin = R_yaw_T @ nu_cn                          # (3,)
        nu_c = torch.cat([nu_c_lin, torch.zeros(3, dtype=x.dtype, device=x.device)])  # (6,)

        # ----- 3) relative velocity -------------------------------------------
        nu_r = nu - nu_c

        # ----- 4) d(ν_c,b)/dt  (uses cross-product) ---------------------------
        dnu_cb_lin = - Smat_torch(torch.tensor([0.0, 0.0, nu[-1]], dtype=x.dtype, device=x.device)) \
                     @ R_yaw_T @ nu_cn                      # (3,)
        dnu_cb = torch.cat([dnu_cb_lin[:2],
                            torch.tensor([0.0], dtype=x.dtype, device=x.device),
                            dnu_cb_lin[2:],
                            torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)])  # (6,)

        # ----- 5) kinematics ---------------------------------------------------
        eta_dot = J_torch(eta) @ nu                         # (6,)

        # ----- 6) kinetics -----------------------------------------------------
        nu_dot = self._Minv @ (tau - self._D @ nu_r - self._G @ eta + self._Ma @ dnu_cb)

        return torch.cat([eta_dot, nu_dot])

    # --------------------------------------------------------------------------
    #                    frequency-dependent hydrodynamics
    # --------------------------------------------------------------------------
    def set_hydrod_parameters(self,
                              freq: float | list | torch.Tensor,
                              config_file: str | None = None):
        """
        Update added-mass and potential damping matrices to match a
        particular encounter frequency  ω  (rad/s).

        freq : scalar or length-6 iterable/tensor (one per DOF).
        """

        cfg_path = os.path.expanduser(config_file) if config_file else self._config_file
        with open(cfg_path, "r") as f:
            param = json.load(f)

        # --- normalise `freq` to tensor --------------------------------------
        freq = torch.as_tensor(freq, dtype=torch.float32)
        if freq.ndim == 0:                       # → scalar
            freq = freq.unsqueeze(0)

        if freq.numel() not in (1, self.DOF):
            raise ValueError(f"`freq` must be a scalar or length-{self.DOF} array, got {freq.shape}")

        freqs_json = torch.tensor(param["freqs"], dtype=torch.float32)   # (Nfreqs,)

        # find nearest index/indices
        if freq.numel() == 1:
            idx = torch.argmin(torch.abs(freqs_json - freq))
        else:
            # per-DOF min |ω – ωᵢ|
            idx = torch.argmin(torch.abs(freqs_json.unsqueeze(0) - freq.unsqueeze(1)), dim=1)  # (6,)

        all_dof = torch.arange(6)

        self._Ma = torch.tensor(param["A"], dtype=torch.float32)[:, all_dof, idx, 0]
        self._Dp = torch.tensor(param["B"], dtype=torch.float32)[:, all_dof, idx, 0]

        self._M   = self._Mrb + self._Ma
        self._Minv= torch.inverse(self._M)
        self._D   = self._Dv + self._Dp