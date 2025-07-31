from __future__ import annotations
"""
Extended plotting script for the adaptive-control study (tidy y-axis labels).

Only differences from the version you posted
─────────────────────────────────────────────
✓ Bottom-row tick labels shown only on first column → no more overlap.  
✓ Slightly larger wspace so the first-column labels have breathing room.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import argparse, itertools, pickle, warnings
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
from scipy.stats import beta
from matplotlib.ticker import ScalarFormatter

# ── Matplotlib defaults ──────────────────────────────────────────────────────
def _set_matplotlib_defaults():
    desired_serif = ["Times", "Times New Roman"]
    installed     = {f.name for f in fm.fontManager.ttflist}
    serif_family  = next((f for f in desired_serif if f in installed),
                         "DejaVu Serif")
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       [serif_family],
        "mathtext.fontset": "cm",
        "font.size":        15,
        "axes.titlesize":   8,
        "legend.fontsize":  "medium",
        "lines.linewidth":  2,
        "lines.markersize": 8,
        "errorbar.capsize": 6,
        "savefig.dpi":      300,
        "savefig.format":   "jpeg",
    })

# ── Helper paths / I/O ───────────────────────────────────────────────────────
ROOT_RESULTS = Path("data/testing_results/rvg/model_uncertainty/sat/all/act_off/ctrl_pen_6")
FIG_DIR      = Path("figures/rvg/meta")

def _first_key(d: dict, *candidates) -> Sequence:
    for k in candidates:
        if k in d: return d[k]
    return []

def load_test_result(seed: int, M: int):
    path = ROOT_RESULTS / f"seed={seed}_M={M}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)

def savefig(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.jpeg", bbox_inches="tight")
    print("Saved →", FIG_DIR / f"{name}.jpeg")

# ── Gain-grid helpers ────────────────────────────────────────────────────────
def _build_gain_grids(sample):
    g = sample["gains"]
    adapt = list(itertools.product(_first_key(g,"Λ","Lambda"),
                                   _first_key(g,"K","Kp_adapt"),
                                   _first_key(g,"P","Ki_adapt")))
    pid   = list(itertools.product(_first_key(g,"Kp","KP"),
                                   _first_key(g,"Ki","KI"),
                                   _first_key(g,"Kd","KD")))
    return adapt, pid

def _title_for_subplot(adapt_gain, pid_gain):
    λ,k,p    = adapt_gain
    kp,ki,kd = pid_gain
    f = lambda v,n: (f"||{n}||={np.linalg.norm(v):.1e}"
                     if isinstance(v,(np.ndarray,jnp.ndarray))
                     else f"{n}={v:.1e}")
    return (f"Adaptive: $\\Lambda={λ:.1e}$, $K={k:.1e}$, $P={p:.1e}$\n"
            f"PID: {f(kp,'Kp')}, {f(ki,'Ki')}, {f(kd,'Kd')}")

# ── Figure 1 – wave-height distribution (unchanged) ─────────────────────────
def figure_wave_height(seed=0, M=2):
    res = load_test_result(seed, M)
    a, b     = res["beta_params"]
    hs_min   = res["hs_min"]; hs_max = res["hs_max"]
    hs_test  = np.asarray(res["hs"]).ravel()
    x        = np.linspace(0,1,400)
    hs_pdf_x = hs_min + (hs_max-hs_min)*x
    pdf      = beta.pdf(x,a,b)/(hs_max-hs_min)

    _, bins = np.histogram(hs_test,bins=15)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(hs_pdf_x,pdf,label=r"analytic $p(h_s)$",color="tab:blue")
    ax.hist(hs_test,bins=bins,density=True,alpha=.5,
            color="tab:orange",label="test samples")
    ax.set_xlabel(r"$h_s\,[\mathrm{m}]$"); ax.set_ylabel("sampling probability")
    ax.legend(); fig.tight_layout()
    savefig(fig,"hs_distribution")

# ── Figure 2 – PID / Adaptive / Meta-Adaptive ───────────────────────────────
def figure_rms(seed_range=range(0,3), Ms=(2,5,10,20)):
    Ms, seeds  = np.asarray(Ms), np.asarray(list(seed_range))
    sample     = load_test_result(int(seeds[0]), int(Ms[0]))
    adapt_grid, pid_grid = _build_gain_grids(sample)
    G_a, G_p   = len(adapt_grid), len(pid_grid)

    rms_err  = {"pid": np.zeros((G_p,seeds.size,Ms.size)),
                "adaptive": np.zeros((G_a,seeds.size,Ms.size)),
                "meta": np.zeros((seeds.size,Ms.size))}
    rms_ctrl = {k: np.copy(v) for k,v in rms_err.items()}

    for s_i,seed in enumerate(seeds):
        for m_i,M in enumerate(Ms):
            res = load_test_result(int(seed),int(M))
            for g in range(G_a):
                ar = res["adaptive_ctrl"].ravel()[g]
                rms_err ["adaptive"][g,s_i,m_i]  = ar["rms_error"].mean()
                rms_ctrl["adaptive"][g,s_i,m_i]  = ar["rms_ctrl"].mean()
            for g in range(G_p):
                pr = res["pid"].ravel()[g]
                rms_err ["pid"][g,s_i,m_i]        = pr["rms_error"].mean()
                rms_ctrl["pid"][g,s_i,m_i]        = pr["rms_ctrl"].mean()
            rms_err ["meta"][s_i,m_i]            = res["meta_adaptive_ctrl"]["rms_error"].mean()
            rms_ctrl["meta"][s_i,m_i]            = res["meta_adaptive_ctrl"]["rms_ctrl"].mean()

    colors = ("tab:pink","tab:green","tab:blue")
    labels = ("PID","Adaptive","Meta-Adaptive")
    metrics = (r"$\frac{1}{N}\sum\text{RMS}(x - x^*)$",
               r"$\frac{1}{N}\sum\text{RMS}(u)$")

    fig, ax = plt.subplots(2,G_a,figsize=(4*G_a+2,6),
                           sharex=True,squeeze=False)
    ax[0,0].set_ylabel(metrics[0])   # top-row label
    # bottom-row label will be set for first column only

    for g,adapt in enumerate(adapt_grid):
        pid = pid_grid[g % G_p]

        # ─ top row: RMS tracking error (labels everywhere) ─
        ax[0,g].errorbar(Ms, rms_err["pid"][g].mean(0),  rms_err["pid"][g].std(0),
                         fmt="-o", color=colors[0], label=None if g else labels[0])
        ax[0,g].errorbar(Ms, rms_err["adaptive"][g].mean(0), rms_err["adaptive"][g].std(0),
                         fmt="-o", color=colors[1], label=None if g else labels[1])
        ax[0,g].errorbar(Ms, rms_err["meta"].mean(0), rms_err["meta"].std(0),
                         fmt="-o", color=colors[2], label=None if g else labels[2])
        ax[0,g].set_title(_title_for_subplot(adapt,pid), pad=8)

        # ─ bottom row: RMS control effort (labels only on first col) ─
        ax[1,g].errorbar(Ms, rms_ctrl["pid"][g].mean(0),  rms_ctrl["pid"][g].std(0),
                         fmt="-o", color=colors[0])
        ax[1,g].errorbar(Ms, rms_ctrl["adaptive"][g].mean(0), rms_ctrl["adaptive"][g].std(0),
                         fmt="-o", color=colors[1])
        ax[1,g].errorbar(Ms, rms_ctrl["meta"].mean(0), rms_ctrl["meta"].std(0),
                         fmt="-o", color=colors[2])
        ax[1,g].set_xlabel("$M$")
        ax[1,g].set_yscale("log")

        # tighten y-limits
        vals = np.concatenate([rms_ctrl["pid"][g].ravel(),
                               rms_ctrl["adaptive"][g].ravel(),
                               rms_ctrl["meta"].ravel()])
        vals = vals[np.isfinite(vals)&(vals>0)]
        if vals.size:
            ax[1,g].set_ylim(vals.min()*0.9, vals.max()*1.1)

        # bottom-row ticks & label only for first column
        if g == 0:
            ax[1,g].set_ylabel(metrics[1])
        else:
            ax[1,g].tick_params(labelleft=False)

    # spacing tweak ─────────────────────────────────────────────────────────
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],
               loc="lower center", ncol=3)
    fig.subplots_adjust(bottom=0.22, hspace=0.15, wspace=0.35)  # ← wider columns
    savefig(fig,"rms_lineplot_model_uncertainty_sat_ctrl_pen_6")

# ── Figure 2b – Adaptive vs Meta-Adaptive (same y-axis tweak) ───────────────
def figure_rms_meta_vs_adaptive(seed_range=range(0,3), Ms=(2,5,10,20)):
    Ms, seeds  = np.asarray(Ms), np.asarray(list(seed_range))
    sample     = load_test_result(int(seeds[0]), int(Ms[0]))
    adapt_grid,_ = _build_gain_grids(sample)
    G_a         = len(adapt_grid)

    rms_err  = {"adaptive": np.zeros((G_a,seeds.size,Ms.size)),
                "meta":     np.zeros((    seeds.size,Ms.size))}
    rms_ctrl = {k: np.copy(v) for k,v in rms_err.items()}

    for s_i,seed in enumerate(seeds):
        for m_i,M in enumerate(Ms):
            res = load_test_result(int(seed),int(M))
            for g in range(G_a):
                ar = res["adaptive_ctrl"].ravel()[g]
                rms_err ["adaptive"][g,s_i,m_i]  = ar["rms_error"].mean()
                rms_ctrl["adaptive"][g,s_i,m_i]  = ar["rms_ctrl"].mean()
            rms_err ["meta"][s_i,m_i]           = res["meta_adaptive_ctrl"]["rms_error"].mean()
            rms_ctrl["meta"][s_i,m_i]           = res["meta_adaptive_ctrl"]["rms_ctrl"].mean()

    colors = ("tab:green","tab:blue")
    labels = ("Adaptive","Meta-Adaptive")
    metrics= (r"$\frac{1}{N}\sum\text{RMS}(x - x^*)$",
              r"$\frac{1}{N}\sum\text{RMS}(u)$")

    fig, ax = plt.subplots(2,G_a,figsize=(4*G_a+2,6),
                           sharex=True,squeeze=False)
    ax[0,0].set_ylabel(metrics[0])

    for g,adapt in enumerate(adapt_grid):
        ax[0,g].errorbar(Ms, rms_err["adaptive"][g].mean(0), rms_err["adaptive"][g].std(0),
                         fmt="-o", color=colors[0], label=None if g else labels[0])
        ax[0,g].errorbar(Ms, rms_err["meta"].mean(0), rms_err["meta"].std(0),
                         fmt="-o", color=colors[1], label=None if g else labels[1])
        ax[0,g].set_title(_title_for_subplot(adapt,(np.nan,)*3), pad=8)

        ax[1,g].errorbar(Ms, rms_ctrl["adaptive"][g].mean(0), rms_ctrl["adaptive"][g].std(0),
                         fmt="-o", color=colors[0])
        ax[1,g].errorbar(Ms, rms_ctrl["meta"].mean(0), rms_ctrl["meta"].std(0),
                         fmt="-o", color=colors[1])
        ax[1,g].set_xlabel("$M$"); ax[1,g].set_yscale("log")

        vals = np.concatenate([rms_ctrl["adaptive"][g].ravel(),
                               rms_ctrl["meta"].ravel()])
        vals = vals[np.isfinite(vals)&(vals>0)]
        if vals.size:
            ax[1,g].set_ylim(vals.min()*0.9, vals.max()*1.1)

        if g == 0:
            ax[1,g].set_ylabel(metrics[1])
        else:
            ax[1,g].tick_params(labelleft=False)

    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],
               loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.22, hspace=0.15, wspace=0.35)
    savefig(fig,"rms_lineplot_meta_vs_adaptiv_model_uncertainty_sat_ctrl_pen_6")

# ── Figure 3 – single trajectory (unchanged) ────────────────────────────────
def figure_single_trajectory(seed=0,M=10):
    res = load_test_result(seed,M)
    if "trajectory" not in res:
        warnings.warn("Full time-series missing; skipping Figure 3.")
        return
    q,r,u,t = (res["trajectory"][k] for k in ("q","r","u","t"))
    e = np.linalg.norm(q-r,1); uu = np.linalg.norm(u,1)

    fig,(ax_p,ax_e,ax_u)=plt.subplots(1,3,figsize=(15,4.5))
    ax_p.plot(r[:,0],r[:,1],"--",lw=4,color="tab:red",label="target")
    ax_p.plot(q[:,0],q[:,1],     color="tab:blue",label="meta_adaptive_ctrl")
    ax_p.set_aspect("equal")
    ax_p.set_xlabel("$x$ [m]"); ax_p.set_ylabel("$y$ [m]")
    ax_e.plot(t,e);  ax_e.set_xlabel("$t$ [s]"); ax_e.set_ylabel(r"$\|e(t)\|$")
    ax_u.plot(t,uu); ax_u.set_xlabel("$t$ [s]"); ax_u.set_ylabel(r"$\|u(t)\|$")
    fig.legend(loc="lower center",ncol=2); fig.subplots_adjust(bottom=0.15)
    savefig(fig,"_single_traj")

# ── CLI wrapper ──────────────────────────────────────────────────────────────
def main():
    _set_matplotlib_defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--seed-range", default="0,1,2")
    p.add_argument("--Ms",        default="2,5,10,20")
    p.add_argument("--no-figure3", action="store_true")
    a = p.parse_args()

    seeds = (range(*map(int,a.seed_range.split(":"))) if ":" in a.seed_range
             else [int(s) for s in a.seed_range.split(",")])
    Ms    = [int(m) for m in a.Ms.split(",")]

    figure_wave_height(seed=seeds[0], M=Ms[0])
    figure_rms(seeds, Ms)
    figure_rms_meta_vs_adaptive(seeds, Ms)
    if not a.no_figure3:
        try: figure_single_trajectory(seed=seeds[0], M=Ms[1])
        except Exception as e: warnings.warn(str(e), UserWarning)

if __name__ == "__main__":
    main()
