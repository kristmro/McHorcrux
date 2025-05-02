"""
Plot test-set results for the adaptive-control study that generates the
pickles in
    data/testing_results/all/act_off/ctrl_pen_3/seed={S}_M={M}.pkl

The script reproduces three figures that mirror those in Spencer Richards'
original paper, but it is **rewired** to match the variable names and folder
layout in `jax_core/meta_adaptive_ctrl/test_all.py`.

--------------------------------------------------------------------------
Figure 1 - distribution of the significant wave height *hₛ* used for testing.
Figure 2 - RMS tracking error / control effort versus the sub-sampling factor *M*
            for every gain triple (Λ,K,P).
Figure 3 - per-trajectory state / input norms for a single illustrative run
            (seed 0, M = 10).  Figure 3 is optional: it is generated only if the
            test pickle contains time-series arrays (q, r, u, …).
--------------------------------------------------------------------------

Run with
    python plot__adaptive.py [--no-figure3]

----------------------------------------------------------------------
Author:  Kristian Magnus Roen adapted from Spencer Richards original plotting code.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import beta
import matplotlib.font_manager as fm

###############################################################################
# Matplotlib defaults - fall back gracefully if Times isn’t installed
###############################################################################

def _set_matplotlib_defaults():
    """Set figure style and pick a serif font that actually exists."""

    # Try Times / Times New Roman first; otherwise fall back to DejaVu Serif
    desired_serif = ["Times", "Times New Roman"]
    installed = {f.name for f in fm.fontManager.ttflist}
    serif_family = next((f for f in desired_serif if f in installed),
                        "DejaVu Serif")

    if serif_family not in desired_serif:
        warnings.warn(
            "Times fonts not found - falling back to 'DejaVu Serif'. "
            "Install 'texlive-fonts-extra' (Linux) or 'Times New Roman' "
            "(Windows/macOS) to match the paper exactly.",
            UserWarning,
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif_family],
        "mathtext.fontset": "cm",
        "font.size": 16,
        "legend.fontsize": "medium",
        "axes.titlesize": "medium",
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "errorbar.capsize": 6,
        "savefig.dpi": 300,
        "savefig.format": "jpeg",
    })


###############################################################################
# Helper utilities
###############################################################################

ROOT_RESULTS = Path("data/testing_results/all/act_off/ctrl_pen_1")
FIG_DIR = Path("figures")


def load_test_result(seed: int, M: int):
    """Load one of the *test* pickles produced by `test_all.py`."""
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
# FIGURE 1 - wave-height distributions
###############################################################################

def figure_wave_height(seed: int = 0, M: int = 2):
    """Plot the PDF used to draw the significant wave-height *hₛ* in testing."""
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
# FIGURE 2 - aggregated RMS tracking / control across gains & M
###############################################################################

def figure_rms(seed_range=range(0, 3), Ms=(2, 5, 10, 20)):
    Ms = np.asarray(Ms)
    seeds = np.asarray(list(seed_range))

    sample = load_test_result(int(seeds[0]), int(Ms[0]))
    gain_grid = list(itertools.product(sample["gains"]["Λ"],
                                       sample["gains"]["K"],
                                       sample["gains"]["P"]))
    G = len(gain_grid)

    # storage:  [gain, seed, M]
    rms_err = {
        "pid":                 np.zeros((G, seeds.size, Ms.size)),
        "adaptive_ctrl":       np.zeros((G, seeds.size, Ms.size)),
        "meta_adaptive_ctrl":  np.zeros((seeds.size, Ms.size)),
    }
    rms_ctrl = {k: np.copy(v) for k, v in rms_err.items()}

    for s_idx, seed in enumerate(seeds):
        for m_idx, M in enumerate(Ms):
            res = load_test_result(int(seed), int(M))

            for g_idx, _ in enumerate(gain_grid):
                for method in ("pid", "adaptive_ctrl"):
                    rms_err[method][g_idx, s_idx, m_idx] = np.mean(
                        res[method].ravel()[g_idx]["rms_error"])
                    rms_ctrl[method][g_idx, s_idx, m_idx] = np.mean(
                        res[method].ravel()[g_idx]["rms_ctrl"])

            rms_err["meta_adaptive_ctrl"][s_idx, m_idx] = np.mean(
                res["meta_adaptive_ctrl"]["rms_error"])
            rms_ctrl["meta_adaptive_ctrl"][s_idx, m_idx] = np.mean(
                res["meta_adaptive_ctrl"]["rms_ctrl"])

    colors = ("tab:pink", "tab:green", "tab:blue")
    labels = ("PID", "adaptive_ctrl", "meta_adaptive_ctrl (meta-gains)")
    metrics = (r"$\frac{1}{N}\sum\text{RMS}(x - x^*)$",
               r"$\frac{1}{N}\sum\text{RMS}(u)$")

    fig, axes = plt.subplots(2, G, figsize=(4 * G + 2, 6), sharex=True)
    axes[0, 0].set_ylabel(metrics[0])
    axes[1, 0].set_ylabel(metrics[1])

    for g_idx, (λ, k, p) in enumerate(gain_grid):
        for m_idx, method in enumerate(("pid", "adaptive_ctrl")):
            axes[0, g_idx].errorbar(Ms,
                                    np.mean(rms_err[method][g_idx], axis=0),
                                    np.std(rms_err[method][g_idx], axis=0),
                                    fmt="-o", color=colors[m_idx])
            axes[1, g_idx].errorbar(Ms,
                                    np.mean(rms_ctrl[method][g_idx], axis=0),
                                    np.std(rms_ctrl[method][g_idx], axis=0),
                                    fmt="-o", color=colors[m_idx])

        axes[0, g_idx].errorbar(Ms,
                                np.mean(rms_err["meta_adaptive_ctrl"], axis=0),
                                np.std(rms_err["meta_adaptive_ctrl"], axis=0),
                                fmt="-o", color=colors[-1])
        axes[1, g_idx].errorbar(Ms,
                                np.mean(rms_ctrl["meta_adaptive_ctrl"], axis=0),
                                np.std(rms_ctrl["meta_adaptive_ctrl"], axis=0),
                                fmt="-o", color=colors[-1])

        axes[0, g_idx].set_title(
            rf"$(\Lambda,K,P)=({λ}I,{k}I,{p}I)$", pad=6)
        axes[1, g_idx].set_xlabel("$M$")

    handles = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.subplots_adjust(bottom=0.18, hspace=0.15)
    savefig(fig, "rms_lineplot")


###############################################################################
# FIGURE 3 - illustrative single trajectory (optional)
###############################################################################

def figure_single_trajectory(seed: int = 0, M: int = 10):
    """Plot q-space path, tracking error norm and control norm for one traj.

    This will only run if the pickle contains full time-series arrays with keys
    "q", "r", "u", "e", etc.  If they are missing we show a gentle warning and
    return.
    """
    res = load_test_result(seed, M)
    if "trajectory" not in res:  # Expect that you saved them under this key
        warnings.warn(
            "Full time-series were not saved by `test_all.py`; skipping Figure 3.",
            UserWarning,
        )
        return

    traj = res["trajectory"]  # Dict of arrays
    q, r, u, t = traj["q"], traj["r"], traj["u"], traj["t"]
    e_norm = np.linalg.norm(q - r, axis=1)
    u_norm = np.linalg.norm(u, axis=1)

    fig = plt.figure(figsize=(15, 7.5))
    grid = plt.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1],
                        hspace=0.05, wspace=0.3)
    ax_path  = plt.subplot(grid[:, 0])
    ax_err   = plt.subplot(grid[0, 1])
    ax_ctrl  = plt.subplot(grid[1, 1])

    ax_path.plot(r[:, 0], r[:, 1], "--", color="tab:red", lw=4, label="target")
    ax_path.plot(q[:, 0], q[:, 1], color="tab:blue", label="meta_adaptive_ctrl")
    ax_path.set_xlabel("$x\,[m]$")
    ax_path.set_ylabel("$y\,[m]$")
    ax_path.set_aspect("equal")

    ax_err.plot(t, e_norm, color="tab:blue")
    ax_err.set_ylabel(r"$\|e(t)\|$")
    ax_err.tick_params(labelbottom=False)

    ax_ctrl.plot(t, u_norm, color="tab:blue")
    ax_ctrl.set_xlabel("$t\,[s]$")
    ax_ctrl.set_ylabel(r"$\|u(t)\|$")

    fig.legend(loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.17)
    savefig(fig, "_single_traj")


###############################################################################
# Entrypoint - CLI wrapper
###############################################################################

def main():
    _set_matplotlib_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-range", type=str, default="10",
                        help="range of seeds, e.g. '0:3' or '0,1,2'")
    parser.add_argument("--Ms", type=str, default="10,20",
                        help="comma-separated list of M values")
    parser.add_argument("--no-figure3", action="store_true",
                        help="skip the (optional) single-trajectory figure")
    args = parser.parse_args()

    # Parse seed range
    if ":" in args.seed_range:
        a, b = map(int, args.seed_range.split(":"))
        seeds = range(a, b)
    else:
        seeds = [int(s) for s in args.seed_range.split(",")]

    Ms = [int(m) for m in args.Ms.split(",")]

    figure_wave_height(seed=next(iter(seeds)), M=Ms[0])
    figure_rms(seed_range=seeds, Ms=Ms)

    if not args.no_figure3:
        try:
            figure_single_trajectory(seed=next(iter(seeds)), M=Ms[1])
        except Exception as exc:
            warnings.warn(str(exc), UserWarning)


if __name__ == "__main__":
    main()
