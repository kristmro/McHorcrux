from __future__ import annotations

"""
Extended plotting script for the adaptive-control study.

Changes w.r.t. the original version
-----------------------------------
* **Figure 2 titles** now show **both** the adaptive-controller gains
  (Λ, K, P) *and* the corresponding **PID tuning** (Kp, Ki, Kd).
  This makes it crystal-clear which gain triple each subplot refers to.
* The code is resilient to slightly different pickle layouts:
  it tries several key spellings (e.g. 'Λ' vs 'Lambda', 'Kd' vs 'KD'),
  and will fall back gracefully if a particular gain list is missing.
  When the PID grid is larger than the adaptive grid, the titles cycle
  through the PID list so you always see a valid set of PID gains.
* Matplotlib defaults tightened up (smaller title font so the longer
  titles fit nicely above each subplot).

Run exactly like before:
    python plot__adaptive.py [--no-figure3]
"""

import argparse
import itertools
import os
import pickle
from pathlib import Path
import warnings
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
from scipy.stats import beta

###############################################################################
# Matplotlib defaults – pick a serif font that actually exists ###############
###############################################################################


def _set_matplotlib_defaults():
    desired_serif = ["Times", "Times New Roman"]
    installed = {f.name for f in fm.fontManager.ttflist}
    serif_family = next((f for f in desired_serif if f in installed),
                        "DejaVu Serif")

    if serif_family not in desired_serif:
        warnings.warn(
            "Times fonts not found – falling back to 'DejaVu Serif'.",
            UserWarning,
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif_family],
        "mathtext.fontset": "cm",
        "font.size": 15,
        "legend.fontsize": "medium",
        "axes.titlesize": 10,   # ↓  so two-line titles fit
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "errorbar.capsize": 6,
        "savefig.dpi": 300,
        "savefig.format": "jpeg",
    })

###############################################################################
# Helper utilities ###########################################################
###############################################################################

ROOT_RESULTS = Path("data/testing_results/rvg/model_uncertainty/all/act_off/ctrl_pen_6") # Updated path
FIG_DIR = Path("figures/rvg")


def _first_key(d: dict, *candidates) -> Sequence:
    """Return the first entry in *candidates* that exists in *d*."""
    for k in candidates:
        if k in d:
            return d[k]
    return []  # graceful fallback


def load_test_result(seed: int, M: int):
    path = ROOT_RESULTS / f"seed={seed}_M={M}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def savefig(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    print(f"Saved → {FIG_DIR / name}.jpeg")

###############################################################################
# FIGURE 1 – wave-height distribution #######################################
###############################################################################


def figure_wave_height(seed: int = 0, M: int = 2):
    res = load_test_result(seed, M)
    a, b = res["beta_params"]
    hs_min, hs_max = res["hs_min"], res["hs_max"]
    hs_test = np.asarray(res["hs"]).ravel()

    x_unit = np.linspace(0.0, 1.0, 400)
    hs_pdf_x = hs_min + (hs_max - hs_min) * x_unit
    pdf = beta.pdf(x_unit, a, b) / (hs_max - hs_min)

    _, bins = np.histogram(hs_test, bins=15)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hs_pdf_x, pdf, label=r"analytic $p(h_s)$", color="tab:blue")
    ax.hist(hs_test, bins=bins, density=True, alpha=0.5, color="tab:orange",
            label="test samples")

    ax.set_xlabel(r"$h_s\;[\text{m}]$")
    ax.set_ylabel("sampling probability")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "hs_distribution")

###############################################################################
# FIGURE 2 – RMS tracking / control ##########################################
###############################################################################


def _build_gain_grids(sample) -> Tuple[Sequence[Tuple[float, float, float]],
                                       Sequence[Tuple[float, float, float]]]:
    """Return (adaptive_grid, pid_grid) as flat lists of tuples."""

    gains = sample["gains"]

    lambda_vals = _first_key(gains, "Λ", "Lambda")
    K_vals      = _first_key(gains, "K", "Kp_adapt")  # allow alt spelling
    P_vals      = _first_key(gains, "P", "Ki_adapt")

    kp_vals = _first_key(gains, "Kp", "KP")
    ki_vals = _first_key(gains, "Ki", "KI")
    kd_vals = _first_key(gains, "Kd", "KD")

    adapt_grid = list(itertools.product(lambda_vals, K_vals, P_vals))
    pid_grid   = list(itertools.product(kp_vals, ki_vals, kd_vals))
    return adapt_grid, pid_grid


def _title_for_subplot(adapt_gain: Tuple[float, float, float],
                       pid_gain:   Tuple[float, float, float]) -> str:
    λ, k, p = adapt_gain
    kp, ki, kd = pid_gain
    # Format PID gains in scientific notation with 1 decimal place
    return ("Adaptive: $\\Lambda={0:.1e}$, $K={1:.1e}$, $P={2:.1e}$\n"
            "PID: $K_p={3:.1e}$, $K_i={4:.1e}$, $K_d={5:.1e}$".format(
                λ, k, p, kp, ki, kd))


def figure_rms(seed_range=range(0, 3), Ms=(2, 5, 10, 20)):
    Ms = np.asarray(Ms)
    seeds = np.asarray(list(seed_range))

    sample = load_test_result(int(seeds[0]), int(Ms[0]))
    adapt_grid, pid_grid = _build_gain_grids(sample)
    G_adapt = len(adapt_grid)
    G_pid   = len(pid_grid)
    if G_adapt == 0 or G_pid == 0:
        raise RuntimeError("Could not find gains in the pickle – check the keys!")

    # storage:  [gain, seed, M]
    rms_err = {
        "pid":                 np.zeros((G_adapt, seeds.size, Ms.size)),
        "adaptive_ctrl":       np.zeros((G_adapt, seeds.size, Ms.size)),
        "meta_adaptive_ctrl":  np.zeros((seeds.size, Ms.size)),
    }
    rms_ctrl = {k: np.copy(v) for k, v in rms_err.items()}

    for s_idx, seed in enumerate(seeds):
        for m_idx, M in enumerate(Ms):
            res = load_test_result(int(seed), int(M))

            for g_idx, _ in enumerate(adapt_grid):
                for method in ("pid", "adaptive_ctrl"):
                    # Defensive: handle mismatch by cycling PID grid
                    linear_idx = g_idx % len(res[method].ravel())
                    rms_err[method][g_idx, s_idx, m_idx] = np.mean(
                        res[method].ravel()[linear_idx]["rms_error"])
                    rms_ctrl[method][g_idx, s_idx, m_idx] = np.mean(
                        res[method].ravel()[linear_idx]["rms_ctrl"])

            rms_err["meta_adaptive_ctrl"][s_idx, m_idx] = np.mean(
                res["meta_adaptive_ctrl"]["rms_error"])
            rms_ctrl["meta_adaptive_ctrl"][s_idx, m_idx] = np.mean(
                res["meta_adaptive_ctrl"]["rms_ctrl"])

    colors = ("tab:pink", "tab:green", "tab:blue")
    labels = ("PID", "adaptive_ctrl", "meta_adaptive_ctrl (meta-gains)")
    metrics = (r"$\frac{1}{N}\sum\text{RMS}(x - x^*)$",  # Corrected LaTeX
               r"$\frac{1}{N}\sum\text{RMS}(u)$")      # Corrected LaTeX

    fig, axes = plt.subplots(2, G_adapt, figsize=(4 * G_adapt + 2, 6), sharex=True)
    axes[0, 0].set_ylabel(metrics[0])
    axes[1, 0].set_ylabel(metrics[1])

    for g_idx, adapt_gain in enumerate(adapt_grid):
        pid_gain = pid_grid[g_idx % G_pid]  # cycle if fewer / more entries

        for m_idx, method in enumerate(("pid", "adaptive_ctrl")):
            mean_err_data = np.mean(rms_err[method][g_idx], axis=0)
            std_err_data = np.std(rms_err[method][g_idx], axis=0)
            axes[0, g_idx].errorbar(Ms, mean_err_data, std_err_data, fmt="-o", color=colors[m_idx])

            mean_ctrl_data = np.mean(rms_ctrl[method][g_idx], axis=0)
            std_ctrl_data = np.std(rms_ctrl[method][g_idx], axis=0)
            axes[1, g_idx].errorbar(Ms, mean_ctrl_data, std_ctrl_data, fmt="-o", color=colors[m_idx])

        mean_meta_err_data = np.mean(rms_err["meta_adaptive_ctrl"], axis=0)
        std_meta_err_data = np.std(rms_err["meta_adaptive_ctrl"], axis=0)
        axes[0, g_idx].errorbar(Ms, mean_meta_err_data, std_meta_err_data, fmt="-o", color=colors[-1])

        mean_meta_ctrl_data = np.mean(rms_ctrl["meta_adaptive_ctrl"], axis=0)
        std_meta_ctrl_data = np.std(rms_ctrl["meta_adaptive_ctrl"], axis=0)
        axes[1, g_idx].errorbar(Ms, mean_meta_ctrl_data, std_meta_ctrl_data, fmt="-o", color=colors[-1])

        axes[0, g_idx].set_title(_title_for_subplot(adapt_gain, pid_gain), pad=8)
        axes[1, g_idx].set_xlabel("$M$")
        axes[1, g_idx].set_yscale('log')

    handles = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.subplots_adjust(bottom=0.22, hspace=0.15)
    savefig(fig, "rms_lineplot:model_uncertainty")

###############################################################################
# FIGURE Meta vs Adaptive RMS ################################################
###############################################################################

def figure_rms_meta_vs_adaptive(seed_range=range(0, 3), Ms=(2, 5, 10, 20)):
    Ms = np.asarray(Ms)
    seeds = np.asarray(list(seed_range))

    sample = load_test_result(int(seeds[0]), int(Ms[0]))
    adapt_grid, pid_grid = _build_gain_grids(sample)
    G_adapt = len(adapt_grid)
    G_pid = len(pid_grid)
    if G_adapt == 0:
        raise RuntimeError("Could not find adaptive gains in the pickle – check the keys!")

    # Storage for relevant methods
    rms_err = {
        "adaptive_ctrl":       np.zeros((G_adapt, seeds.size, Ms.size)),
        "meta_adaptive_ctrl":  np.zeros((seeds.size, Ms.size)),
    }
    rms_ctrl = {
        "adaptive_ctrl":       np.zeros((G_adapt, seeds.size, Ms.size)),
        "meta_adaptive_ctrl":  np.zeros((seeds.size, Ms.size)),
    }

    for s_idx, seed in enumerate(seeds):
        for m_idx_ms, M_val in enumerate(Ms):
            res = load_test_result(int(seed), int(M_val))

            for g_idx, _ in enumerate(adapt_grid):
                if len(res["adaptive_ctrl"].ravel()) > 0:
                    linear_idx_adapt = g_idx % len(res["adaptive_ctrl"].ravel())
                    rms_err["adaptive_ctrl"][g_idx, s_idx, m_idx_ms] = np.mean(
                        res["adaptive_ctrl"].ravel()[linear_idx_adapt]["rms_error"])
                    rms_ctrl["adaptive_ctrl"][g_idx, s_idx, m_idx_ms] = np.mean(
                        res["adaptive_ctrl"].ravel()[linear_idx_adapt]["rms_ctrl"])
                else:
                    rms_err["adaptive_ctrl"][g_idx, s_idx, m_idx_ms] = np.nan
                    rms_ctrl["adaptive_ctrl"][g_idx, s_idx, m_idx_ms] = np.nan

            rms_err["meta_adaptive_ctrl"][s_idx, m_idx_ms] = np.mean(
                res["meta_adaptive_ctrl"]["rms_error"])
            rms_ctrl["meta_adaptive_ctrl"][s_idx, m_idx_ms] = np.mean(
                res["meta_adaptive_ctrl"]["rms_ctrl"])

    colors_new = ("tab:green", "tab:blue")
    labels_new = ("adaptive_ctrl", "meta_adaptive_ctrl (meta-gains)")
    metrics_new = (r"$\frac{1}{N}\sum\text{RMS}(x - x^*)$",  # Corrected LaTeX
                   r"$\frac{1}{N}\sum\text{RMS}(u)$")      # Corrected LaTeX

    fig, axes = plt.subplots(2, G_adapt, figsize=(4 * G_adapt + 2, 6), sharex=True)
    if G_adapt == 0:
        plt.close(fig)
        return
        
    axes[0, 0].set_ylabel(metrics_new[0])
    axes[1, 0].set_ylabel(metrics_new[1])

    for g_idx, adapt_gain_tuple in enumerate(adapt_grid):
        pid_gain_tuple = pid_grid[g_idx % G_pid] if G_pid > 0 else (np.nan, np.nan, np.nan)

        mean_err_adapt = np.mean(rms_err["adaptive_ctrl"][g_idx], axis=0)
        std_err_adapt = np.std(rms_err["adaptive_ctrl"][g_idx], axis=0)
        if not np.all(np.isnan(mean_err_adapt)):
            axes[0, g_idx].errorbar(Ms, mean_err_adapt, std_err_adapt, fmt="-o", color=colors_new[0])

        mean_ctrl_adapt = np.mean(rms_ctrl["adaptive_ctrl"][g_idx], axis=0)
        std_ctrl_adapt = np.std(rms_ctrl["adaptive_ctrl"][g_idx], axis=0)
        if not np.all(np.isnan(mean_ctrl_adapt)):
            axes[1, g_idx].errorbar(Ms, mean_ctrl_adapt, std_ctrl_adapt, fmt="-o", color=colors_new[0])

        # Plot meta_adaptive_ctrl
        mean_err_meta = np.mean(rms_err["meta_adaptive_ctrl"], axis=0)
        std_err_meta = np.std(rms_err["meta_adaptive_ctrl"], axis=0)
        if not np.all(np.isnan(mean_err_meta)):
            axes[0, g_idx].errorbar(Ms, mean_err_meta, std_err_meta, fmt="-o", color=colors_new[1])

        mean_ctrl_meta = np.mean(rms_ctrl["meta_adaptive_ctrl"], axis=0)
        std_ctrl_meta = np.std(rms_ctrl["meta_adaptive_ctrl"], axis=0)
        if not np.all(np.isnan(mean_ctrl_meta)):
            axes[1, g_idx].errorbar(Ms, mean_ctrl_meta, std_ctrl_meta, fmt="-o", color=colors_new[1])

        axes[0, g_idx].set_title(_title_for_subplot(adapt_gain_tuple, pid_gain_tuple), pad=8)
        axes[1, g_idx].set_xlabel("$M$")

    handles = [Patch(color=c, label=l) for c, l in zip(colors_new, labels_new)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.subplots_adjust(bottom=0.22, hspace=0.15)
    savefig(fig, "rms_lineplot_meta_vs_adaptive:model_uncertainty")

###############################################################################
# FIGURE 3 – single trajectory (unchanged) ###################################
###############################################################################

def figure_single_trajectory(seed: int = 0, M: int = 10):
    res = load_test_result(seed, M)
    if "trajectory" not in res:
        warnings.warn("Full time-series missing; skipping Figure 3.", UserWarning)
        return

    traj = res["trajectory"]
    q, r, u, t = traj["q"], traj["r"], traj["u"], traj["t"]
    e_norm = np.linalg.norm(q - r, axis=1)
    u_norm = np.linalg.norm(u, axis=1)

    fig = plt.figure(figsize=(15, 7.5))
    grid = plt.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1],
                        hspace=0.05, wspace=0.3)
    ax_path = plt.subplot(grid[:, 0])
    ax_err = plt.subplot(grid[0, 1])
    ax_ctrl = plt.subplot(grid[1, 1])

    ax_path.plot(r[:, 0], r[:, 1], "--", color="tab:red", lw=4, label="target")
    ax_path.plot(q[:, 0], q[:, 1], color="tab:blue", label="meta_adaptive_ctrl")
    ax_path.set_xlabel("$x\,[m]$")
    ax_path.set_ylabel("$y\,[m]$")
    ax_path.set_aspect("equal")

    ax_err.plot(t, e_norm, color="tab:blue")
    ax_err.set_ylabel(r"$\|e(t)\|")
    ax_err.tick_params(labelbottom=False)

    ax_ctrl.plot(t, u_norm, color="tab:blue")
    ax_ctrl.set_xlabel("$t\,[s]$")
    ax_ctrl.set_ylabel(r"$\|u(t)\|")

    fig.legend(loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.17)
    savefig(fig, "_single_traj")

###############################################################################
# CLI wrapper ################################################################
###############################################################################

def main():
    _set_matplotlib_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-range", type=str, default="0,1,2,3,4",
                        help="range of seeds, e.g. '0:3' or '0,1,2'")
    parser.add_argument("--Ms", type=str, default="2, 5, 10, 20",
                        help="comma-separated list of M values")
    parser.add_argument("--no-figure3", action="store_true",
                        help="skip the (optional) single-trajectory figure")
    args = parser.parse_args()

    # Parse seed range -----------------------------------------------------
    if ":" in args.seed_range:
        a, b = map(int, args.seed_range.split(":"))
        seeds = range(a, b)
    else:
        seeds = [int(s) for s in args.seed_range.split(",")]

    Ms = [int(m) for m in args.Ms.split(",")]

    figure_wave_height(seed=next(iter(seeds)), M=Ms[0])
    figure_rms(seed_range=seeds, Ms=Ms)
    figure_rms_meta_vs_adaptive(seed_range=seeds, Ms=Ms)

    if not args.no_figure3:
        try:
            figure_single_trajectory(seed=next(iter(seeds)), M=Ms[1])
        except Exception as exc:
            warnings.warn(str(exc), UserWarning)


if __name__ == "__main__":
    main()
