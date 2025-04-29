import torch
import torch.nn.functional as F
import numpy as np
import math
import json
import os
# Ensure pipi and to_positive_angle are imported if they are not already
from torch_core.utils import pipi, to_positive_angle, torch_lininterp_1d

################################################################
# The main differentiable WaveLoad class
################################################################

class WaveLoad(torch.nn.Module):
    """
    Differentiable wave load module in PyTorch, replicating the old wave_loads.py logic.

    This class calculates first- and second-order wave loads:
      - 1st order from vessel force RAOs
      - 2nd order from drift QTF matrices
    using data from a config file that includes forceRAO, driftfrc, freq, headings, etc.
    """
    QTF_METHODS = ["Newman", "geo-mean"]

    def __init__(self,
                 wave_amps,           # array-like (N,) wave amplitude
                 freqs,               # array-like (N,) wave frequencies in rad/s
                 eps,                 # array-like (N,) random phases
                 angles,              # array-like (N,) wave directions
                 config_file,         # vessel JSON path
                 rho=1025,
                 g=9.81,
                 dof=6,
                 depth=100.0,
                 deep_water=True,
                 qtf_method="Newman",
                 qtf_interp_angles=True,
                 interpolate=True):
        super().__init__()

        # 1) Store wave data as Buffers => not trainable, but moved with .to(device).
        self.register_buffer("_amp",    torch.tensor(wave_amps, dtype=torch.float32))
        self.register_buffer("_freqs",  torch.tensor(freqs,     dtype=torch.float32))
        self.register_buffer("_eps",    torch.tensor(eps,       dtype=torch.float32))
        self.register_buffer("_angles", torch.tensor(angles,    dtype=torch.float32))

        self._N = self._amp.shape[0]
        self._rho = rho
        self._g = g
        self._dof = dof
        self._depth = depth
        self._deep_water = deep_water
        self._qtf_method = qtf_method
        self._qtf_interp_angles = qtf_interp_angles
        self._interpolate = interpolate

        # 2) Load vessel config for drift & RAO data
        with open(config_file, 'r') as f:
            vessel_params = json.load(f)
        self._params = vessel_params

        # headings/freqs from config => torch
        # Keep original order for NumPy logic replication in _rao_interp
        # No sorting needed here if _rao_interp uses argmin logic
        qtf_angles_np = np.array(vessel_params['headings'])
        qtf_angles = torch.tensor(qtf_angles_np, dtype=torch.float32)
        qtf_freqs  = torch.tensor(vessel_params['freqs'],    dtype=torch.float32)
        self.register_buffer("_qtf_angles", qtf_angles)
        self.register_buffer("_qtf_freqs",  qtf_freqs)
        # Store degrees version for _rao_interp comparison
        self.register_buffer("_qtf_angles_deg", torch.tensor(np.rad2deg(qtf_angles_np), dtype=torch.float32))
        self._num_qtf_headings = len(qtf_angles_np) # Store number of headings (e.g., 36)

        # 3) Wave numbers (remains unchanged)
        if self._deep_water:
            self.register_buffer("_k", (self._freqs**2)/self._g)
        else:
            k_list = []
            for wval in self._freqs:
                w_ = wval.item()
                k_old = w_**2 / self._g
                k_new = w_**2/(self._g*math.tanh(k_old*self._depth))
                itcount=0
                while abs(k_new - k_old)>1e-5 and itcount<50:
                    k_old=k_new
                    k_new=w_**2/(self._g*math.tanh(k_old*self._depth))
                    itcount+=1
                k_list.append(k_new)
            self.register_buffer("_k", torch.tensor(k_list, dtype=torch.float32))

        # difference freq/phase (remains unchanged)
        W = self._freqs.view(-1,1) - self._freqs.view(1,-1)
        P = self._eps.view(-1,1) - self._eps.view(1,-1)
        self.register_buffer("_W", W)
        self.register_buffer("_P", P)

        # 4) Build Q from driftfrc (use original heading order)
        drift_amp_np = np.array(vessel_params['driftfrc']['amp'], dtype=np.float32)
        drift_amp_t  = torch.tensor(drift_amp_np, dtype=torch.float32)
        if drift_amp_t.dim() == 4:
            drift_amp_t = drift_amp_t[:, :, :, 0]
        # Need sorted angles if using interpolation in _build_qtf_6dof
        sort_idx_build = np.argsort(vessel_params['headings'])
        qtf_angles_sorted = torch.tensor(np.array(vessel_params['headings'])[sort_idx_build], dtype=torch.float32)
        drift_amp_t_sorted = torch.tensor(drift_amp_np[:, :, sort_idx_build, ...], dtype=torch.float32)
        if drift_amp_t_sorted.dim() == 4:
             drift_amp_t_sorted = drift_amp_t_sorted[:, :, :, 0]

        Q = self._build_qtf_6dof(self._freqs, qtf_angles_sorted, qtf_freqs, drift_amp_t_sorted,
                                 method=self._qtf_method,
                                 interpolate=self._interpolate,
                                 qtf_interp_angles=self._qtf_interp_angles)
        self.register_buffer("_Q", Q)

        # 5) Force RAOs from 'forceRAO' (use original heading order)
        force_amp_full_np   = np.array(vessel_params['forceRAO']['amp'],   dtype=np.float32)
        force_phase_full_np = np.array(vessel_params['forceRAO']['phase'], dtype=np.float32)
        force_amp_full   = torch.tensor(force_amp_full_np, dtype=torch.float32)
        force_phase_full = torch.tensor(force_phase_full_np, dtype=torch.float32)

        # Apply abs() to amplitude to match NumPy's _set_force_raos
        force_amp   = torch.abs(force_amp_full[:, :, :, 0])
        force_phase = force_phase_full[:, :, :, 0] * (math.pi/180.0)

        # Apply abs() here too for consistency
        force_amp_sorted = torch.abs(torch.tensor(force_amp_full_np[:, :, sort_idx_build, ...], dtype=torch.float32)[:, :, :, 0])
        force_phase_sorted = torch.tensor(force_phase_full_np[:, :, sort_idx_build, ...], dtype=torch.float32)[:, :, :, 0] * (math.pi/180.0)

        # Removed redundant calculation of fAmp_sorted, fPhase_sorted
        # Store RAOs corresponding to original heading order for _rao_interp
        fAmp_orig_order, fPhase_orig_order = self._build_force_raos(force_amp, force_phase, qtf_freqs)
        self.register_buffer("_forceRAOamp",   fAmp_orig_order)
        self.register_buffer("_forceRAOphase", fPhase_orig_order)


    ########################################################################
    # The "forward" and the load computations
    ########################################################################
    def forward(self, time, eta):
        """
        Equivalent to old __call__:
          returns total wave load = first_order + second_order
        """
        tau_wf = self.first_order_loads(time, eta)
        tau_sv = self.second_order_loads(time, eta[-1])
        return tau_wf + tau_sv

    def first_order_loads(self, t, eta):
        """
        Compute 1st-order wave loads by summing wave components:
          tau_wf = sum_j [ rao_amp[:,j] * cos(... - rao_phase[:,j] ) * wave_amp[j] ]
        """
        # 1) relative angles (result is 0 to 2pi)
        rel_angle = self._relative_incident_angle(eta[-1])  # shape(N,)
        # 2) rao amplitude & phase => shape(6,N) using NumPy logic
        rao_amp, rao_phase = self._rao_interp(rel_angle)

        # 3) wave phase => shape(N,)
        wave_phase = (self._freqs * t
                      - self._k * eta[0] * torch.cos(self._angles)
                      - self._k * eta[1] * torch.sin(self._angles)
                      - self._eps)
        # shape => (6,N)
        arg_matrix = wave_phase.unsqueeze(0) - rao_phase
        cos_val = torch.cos(arg_matrix)
        contrib = rao_amp * cos_val
        tau_wf = torch.matmul(contrib, self._amp.unsqueeze(-1)).squeeze(-1) # shape(6,)
        return tau_wf

    def second_order_loads(self, t, heading):
        """
        Compute 2nd-order slow-drift from QTF => real( amp * Q e^{i(Wt+P)} * amp ).
        We pick the Q row for the nearest heading (0..359 deg).
        Vectorized version.
        """
        # Use the mean relative angle (0 to 2pi)
        rel_angle_all = self._relative_incident_angle(heading) # shape(N,)
        mean_angle = torch.mean(rel_angle_all) # Scalar mean relative angle (0 to 2pi)

        # Find the index of the closest angle in the QTF headings
        # If _Q is interpolated to 360 degrees (M=360), use angles_1deg
        if self._qtf_interp_angles and self._Q.shape[1] == 360:
             angles_for_qtf = torch.linspace(0, 2*math.pi, 360, device=mean_angle.device)
        else:
             # If not interpolated, _Q corresponds to original _qtf_angles
             # Use the original _qtf_angles (radians) for comparison
             angles_for_qtf = self._qtf_angles

        # Use pipi for correct angle difference calculation, considering wrap-around
        # Ensure mean_angle is compared correctly (e.g., both 0 to 2pi or -pi to pi)
        # Since mean_angle is 0 to 2pi, and angles_for_qtf are likely 0 to 2pi, direct diff is ok
        # Or use pipi(angles_for_qtf - mean_angle) for safety
        diffs = torch.abs(pipi(angles_for_qtf - mean_angle))
        heading_index = torch.argmin(diffs)

        # pick Q => shape(6,N,N)
        Q_sel = self._Q[:, heading_index, :, :]

        # e^{i(W t - P)} => shape(N,N)  <-- Corrected sign for P
        amp_cplx = self._amp.to(torch.complex64) # shape(N,)
        # Corrected sign: exp(1j * W * t - 1j * P)
        exp_term = torch.exp(1j*(self._W*t - self._P))  # shape(N,N)

        # Combine Q and exp_term
        mat = Q_sel * exp_term # shape(6, N, N)

        # Perform the quadratic form: amp^T @ mat @ amp for each DOF
        # Using einsum: 'i, dij, j -> d' sums over i and j for each d
        tau_sv = torch.real(torch.einsum('i,dij,j->d', amp_cplx, mat, amp_cplx)) # shape(6,)

        return tau_sv

    ########################################################################
    # The angle-based interpolation for first-order RAOs
    ########################################################################
    def _rao_interp(self, rel_angle):
        """
        RAO interpolation replicating the NumPy logic using 10-degree floor bins.
        Finds the index in self._qtf_angles closest to the 10-degree floor
        of the relative angle, then interpolates between that index and the next.
        """
        device = rel_angle.device
        N = rel_angle.shape[0] # Number of wave components
        # Assuming _num_qtf_headings corresponds to 36 (0-350 deg) based on NumPy logic
        num_headings = self._num_qtf_headings # e.g., 36

        # 1. Convert relative angles (0 to 2pi) to degrees [0, 360)
        rel_angle_deg = rel_angle * (180.0 / math.pi)

        # 2. Calculate the 10-degree floor
        # Ensure correct handling near 360: floor(358/10)*10 = 350, floor(3/10)*10 = 0
        bin_floor_deg = torch.floor(rel_angle_deg / 10.0) * 10.0 # Shape (N,)

        # 3. Find index_lb: index in _qtf_angles_deg closest to bin_floor_deg
        # self._qtf_angles_deg shape (num_headings,) e.g., (36,)
        # bin_floor_deg shape (N,)
        # Need to compare each bin_floor_deg to all _qtf_angles_deg
        # diffs shape: (N, num_headings)
        diffs = torch.abs(self._qtf_angles_deg.unsqueeze(0) - bin_floor_deg.unsqueeze(1))
        index_lb = torch.argmin(diffs, dim=1) # Shape (N,)

        # 4. Find index_ub: index_lb + 1, wrapping around at num_headings
        # Replicates np.where(index_lb < 35, index_lb + 1, 0) assuming num_headings=36
        index_ub = torch.where(index_lb < (num_headings - 1),
                               index_lb + 1,
                               torch.tensor(0, device=device, dtype=torch.long)) # Shape (N,)

        # 5. Get the corresponding angles (radians) for interpolation factors
        # Use original _qtf_angles (radians)
        theta1 = self._qtf_angles[index_lb] # shape(N,)
        theta2 = self._qtf_angles[index_ub] # shape(N,)

        # 6. Calculate interpolation scale (factor)
        # Use original rel_angle (radians) and pipi for differences
        diff_t = pipi(theta2 - theta1)
        # Add small epsilon where diff_t is zero to avoid NaN
        diff_t = torch.where(torch.abs(diff_t) < 1e-9, torch.tensor(1e-9, device=device, dtype=diff_t.dtype), diff_t)
        numerator = pipi(rel_angle - theta1)
        scale = numerator / diff_t  # shape(N,)
        # Clamp scale to [0, 1] ? NumPy version doesn't explicitly clamp, but maybe should?
        # Let's omit clamp for now to match NumPy exactly. If issues arise, add clamp.
        # scale = torch.clamp(scale, 0.0, 1.0)

        # 7. Gather RAO data using advanced indexing
        # Assumes self._forceRAOamp/_phase have shape (dof, N_freq, N_heading)
        # Assumes N_freq == N (number of wave components matches number of frequencies)
        freq_ind = torch.arange(N, device=device) # shape(N,)

        # Gather lower bound values: shape (dof, N)
        # Use the RAO buffers stored in the original heading order
        rao_amp_lb   = self._forceRAOamp[:, freq_ind, index_lb]
        rao_phase_lb = self._forceRAOphase[:, freq_ind, index_lb]

        # Gather upper bound values: shape (dof, N)
        rao_amp_ub   = self._forceRAOamp[:, freq_ind, index_ub]
        rao_phase_ub = self._forceRAOphase[:, freq_ind, index_ub]

        # 8. Perform linear interpolation
        scale_2d = scale.unsqueeze(0) # shape (1, N)

        # Interpolate amplitude
        rao_amp = rao_amp_lb + (rao_amp_ub - rao_amp_lb) * scale_2d

        # Interpolate phase using pipi for difference to handle wrap-around correctly
        phase_diff = pipi(rao_phase_ub - rao_phase_lb) # Difference respecting wrap-around
        rao_phase = pipi(rao_phase_lb + phase_diff * scale_2d) # Interpolate and wrap result

        return rao_amp, rao_phase

    @staticmethod
    def _build_qtf_6dof(wave_freqs, qtf_headings, qtf_freqs, drift_amp,
                        method="Newman", interpolate=True, qtf_interp_angles=True):
        """
        Build the full 2nd-order QTF => shape(6, M, N, N).
        M is number of headings (e.g., 360 if interpolated, or len(qtf_headings) otherwise).
        Requires qtf_headings to be sorted if qtf_interp_angles is True.
        (Code remains largely the same as previous PyTorch version, uses sorted inputs)
        """
        wave_freqs = wave_freqs.to(torch.float32)
        qtf_freqs  = qtf_freqs.to(torch.float32)
        drift_amp  = drift_amp.to(torch.float32) # Shape (dof, Nfreq, Nhead_sorted)

        dof, Nfreq_cfg, Nhead_cfg = drift_amp.shape
        N = wave_freqs.shape[0]
        device = drift_amp.device

        # 1) freq interpolation => produce Qdiag(dof, N, Nhead_cfg)
        if interpolate:
            if wave_freqs[0].item()<qtf_freqs[0].item():
                zero_freq= torch.tensor([0.0], dtype=torch.float32, device=qtf_freqs.device)
                qtf_freqs_mod= torch.cat([zero_freq, qtf_freqs], dim=0)
                zero_drift   = torch.zeros((dof,1, Nhead_cfg), dtype=drift_amp.dtype, device=drift_amp.device)
                drift_amp_mod= torch.cat([zero_drift, drift_amp], dim=1)
            else:
                qtf_freqs_mod= qtf_freqs
                drift_amp_mod= drift_amp

            Qdiag = torch.zeros((dof, N, Nhead_cfg), dtype=drift_amp.dtype, device=device)
            for d in range(dof):
                for h in range(Nhead_cfg):
                    Qdiag[d,:,h] = torch_lininterp_1d(
                        qtf_freqs_mod, drift_amp_mod[d,:,h], wave_freqs,
                        left_fill=drift_amp_mod[d,0,h], right_fill=drift_amp_mod[d,-1,h]
                    )
        else: # Nearest frequency lookup
            diffs = torch.abs(wave_freqs.unsqueeze(1) - qtf_freqs.unsqueeze(0)) # Shape (N, Nfreq_cfg)
            freq_idx = torch.argmin(diffs, dim=1) # Shape (N,)
            Qdiag = torch.zeros((dof, N, Nhead_cfg), dtype=drift_amp.dtype, device=device)
            for d in range(dof):
                 Qdiag[d, :, :] = drift_amp[d, freq_idx, :]


        # 2) angle interpolation => produce shape(dof, M, N)
        if qtf_interp_angles:
            M = 360
            angles_1deg = torch.linspace(0, 2*math.pi, M, device=device)
            # Interpolate angles - requires qtf_headings (sorted)
            Qdiag2 = torch.zeros((dof, N, M), dtype=Qdiag.dtype, device=device)
            for d in range(dof):
                for iN in range(N):
                    # qtf_headings must be sorted here
                    Qdiag2[d,iN,:] = torch_lininterp_1d(
                        qtf_headings, Qdiag[d,iN,:], angles_1deg,
                        left_fill=Qdiag[d,iN,0], right_fill=Qdiag[d,iN,-1]
                    )
            Qdiag_final = Qdiag2.permute(0,2,1) # shape(dof, M, N)
        else:
            Qdiag_final = Qdiag.permute(0,2,1) # shape(dof, Nhead_cfg, N)
            M = Nhead_cfg

        # 3) Build full QTF matrix using Newman/geo-mean approx (Vectorized)
        Q_4d = torch.zeros((dof, M, N, N), dtype=Qdiag_final.dtype, device=device)
        qvals = Qdiag_final # shape (dof, M, N)
        a = qvals.unsqueeze(3).expand(-1, -1, N, N) # shape (dof, M, N, N)
        b = qvals.unsqueeze(2).expand(-1, -1, N, N) # shape (dof, M, N, N)

        if method.lower()=="newman":
            Q_4d = 0.5*(a+b)
        elif method.lower()=="geo-mean":
            sign_a = torch.sign(a)
            sign_b = torch.sign(b)
            same_sign = (sign_a == sign_b)
            ab = a * b
            val = sign_a * torch.sqrt(torch.abs(ab))
            Q_4d = torch.where(same_sign, val, torch.zeros_like(val))
        else:
            raise ValueError(f"Unknown QTF method: {method}")

        # Apply Q[5]=Q[2] fix if needed (matches NumPy version)
        Q_4d[5] = Q_4d[2].clone()
        Q_4d[2] = 0.0
        return Q_4d

    def _build_force_raos(self, force_amp, force_phase, freq_cfg):
        """
        Builds the 1st-order RAO amplitude & phase w.r.t. self._freqs.
        Requires force_amp/force_phase corresponding to sorted headings if interpolation is used.
        (Code remains largely the same as previous PyTorch version, uses sorted inputs)
        """
        dof, Nfreq_cfg, Nhead_cfg = force_amp.shape
        N = self._N # Number of wave frequencies
        device = force_amp.device

        out_amp   = torch.zeros((dof, N, Nhead_cfg), dtype=torch.float32, device=device)
        out_phase = torch.zeros_like(out_amp)

        if self._interpolate:
            for d in range(dof):
                for h in range(Nhead_cfg):
                    yA = force_amp[d,:,h]
                    yP = force_phase[d,:,h]
                    interpA = torch_lininterp_1d(freq_cfg, yA, self._freqs,
                                                left_fill=yA[0], right_fill=yA[-1])
                    interpP = torch_lininterp_1d(freq_cfg, yP, self._freqs,
                                                left_fill=yP[0], right_fill=yP[-1])
                    out_amp[d,:,h]   = interpA
                    out_phase[d,:,h] = interpP
        else: # Nearest frequency lookup
            diffs = torch.abs(self._freqs.unsqueeze(1) - freq_cfg.unsqueeze(0)) # Shape (N, Nfreq_cfg)
            freq_idx = torch.argmin(diffs, dim=1) # Shape (N,)
            for d in range(dof):
                out_amp[d, :, :] = force_amp[d, freq_idx, :]
                out_phase[d, :, :] = force_phase[d, freq_idx, :]

        return out_amp, out_phase

    def _relative_incident_angle(self, heading):
        """
        The relative wave incident angle gamma (0 to 2pi).
        Matches NumPy version's output range.
        """
        # angles is self._angles (wave directions, shape N)
        # heading is vessel heading (scalar)
        return to_positive_angle(pipi(self._angles - heading)) # shape (N,)

    ###############################################
    # Class method to fetch list of QTF methods
    ###############################################
    @classmethod
    def get_methods(cls):
        return cls.QTF_METHODS
