# ------------------------------------------------------------
#  Four-corner test – compact, publication-grade plots
# ------------------------------------------------------------
import os, pickle, numpy as np, matplotlib.pyplot as plt

# ── Matplotlib setup: white canvas, serif, thin axes ──────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "figure.facecolor": "white",
    "axes.facecolor":  "white",
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
})

# ── Input specification ──────────────────────────────────────
cases = [(2.0,  8.0),
         (3.0, 10.0),
         (4.0, 14.0),
         (5.0, 16.0),
         (6.0, 17.0)]                # (Hs [m], Tp [s])

methods     = ['pid', 'adaptive_ctrl', 'meta_adap_ctrl']
nice_name   = {'pid':'PID', 'adaptive_ctrl':'Adaptive', 'meta_adap_ctrl':'Meta-adaptive'}
colour      = {'pid':'red', 'adaptive_ctrl':'green', 'meta_adap_ctrl':'blue'}
coord_lbl   = ['x', 'y', 'φ']        # surge, sway, yaw

# ── Containers ───────────────────────────────────────────────
rms_per_coord = {m: [] for m in methods}
rms_scalar    = {m: [] for m in methods}
effort_scalar = {m: [] for m in methods}

# ── Gather numbers ───────────────────────────────────────────
for hs, tp in cases:
    pkl_path = (f"data/testing_results/rvg/model_uncertainty/div/{hs}/{tp}/"
                f"train_act_off/four_corner/test_act_off/ctrl_pen_6/seed=2_M=5.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, "rb") as pkl:
        res = pickle.load(pkl)

    for m in methods:
        e = res[m]['e'][:, :3]
        rms = np.sqrt(np.mean(e**2, axis=0))
        rms_per_coord[m].append(rms)
        rms_scalar[m].append(np.linalg.norm(rms))

        u = res[m]['u']                       # change key if needed
        effort_scalar[m].append(np.sqrt(np.mean(u**2)))

Hs_vals = [hs for hs, _ in cases]
# ─────────────────────────────────────────────────────────────
#  FIGURE 1  – aggregate RMS + effort  (with Tp shown)
# ─────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(1, 2, figsize=(11.6, 4.2), sharex=True)

Hs_vals = [hs for hs, _ in cases]
Tp_vals = [tp for _, tp in cases]
xtick_lbl = [fr'$H_s={hs}\,$m' '\n' fr'$T_p={tp}\,$s'
             for hs, tp in cases]               # two-line tick label

# (a) Tracking error
for m in methods:
    ax[0].plot(Hs_vals, rms_scalar[m], 'o-', color=colour[m], label=nice_name[m])
ax[0].set_xlabel('Sea state')
ax[0].set_ylabel('Aggregate RMS position error  [m | rad]')
ax[0].set_title('(a) Tracking performance')
ax[0].grid(True);  ax[0].legend(frameon=False, loc='upper left')

# (b) Control effort
for m in methods:
    ax[1].plot(Hs_vals, effort_scalar[m], 's--', color=colour[m], label=nice_name[m])
ax[1].set_xlabel('Sea state')
ax[1].set_ylabel('RMS control effort  [N | Nm]')
ax[1].set_title('(b) Actuation demand')
ax[1].grid(True);  ax[1].legend(frameon=False, loc='upper left')

# **NEW** – identical ticks on both sub-axes
for a in ax:
    a.set_xticks(Hs_vals, xtick_lbl)

fig1.tight_layout()
# fig1.savefig('fig_tracking_vs_effort.pdf')    # still 300 dpi


# ─────────────────────────────────────────────────────────────
#  FIGURE 2 – per-axis RMS bar charts (x, y, φ aligned)
# ─────────────────────────────────────────────────────────────
fig2, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=False)

bar_w = 0.25
x = np.arange(len(cases))

for j, lbl in enumerate(coord_lbl):          # one column per coordinate
    for k, m in enumerate(methods):
        offs = -bar_w + k*bar_w
        axs[j].bar(x+offs,
                   [v[j] for v in rms_per_coord[m]],
                   bar_w, color=colour[m], label=nice_name[m] if j==0 else None)
    axs[j].set_xticks(x, [f'H{hs}\nT{tp}' for hs, tp in cases])
    axs[j].set_ylabel(f'RMS error in {lbl}')
    axs[j].set_title(f'({chr(99+j)}) {lbl.upper()}-axis')
    axs[j].grid(True, axis='y')

fig2.legend(methods, bbox_to_anchor=(0.5, -0.05), ncol=3,
            labels=[nice_name[m] for m in methods], frameon=False)
fig2.tight_layout()
# fig2.savefig('fig_per_axis_bars.pdf')

# ─────────────────────────────────────────────────────────────
#  FIGURE 3 – time-series overlay (optional, single panel)
# ─────────────────────────────────────────────────────────────
hs, tp = 4.0, 14.0                          # pick any case you want
with open(f"data/testing_results/rvg/model_uncertainty/div/{hs}/{tp}"
          f"/train_act_off/four_corner/test_act_off/ctrl_pen_6/seed=2_M=5.pkl", "rb") as f:
    res = pickle.load(f)

t = res['pid']['t']
plt.figure(figsize=(7.5, 3.8))
for m in methods:
    plt.plot(t, res[m]['e'][:, 0], color=colour[m], label=nice_name[m])
plt.xlabel('Time [s]');  plt.ylabel('$x$-error [m]')
plt.title(f'Time series example – $H_s$={hs} m, $T_p$={tp} s')
plt.grid(True);  plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('fig_time_series_example.pdf')

plt.show()
