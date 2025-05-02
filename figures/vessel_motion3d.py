import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib import cm, colormaps
from matplotlib.colors import LightSource

from mclsimpy.waves import JONSWAP, WaveLoad
from mclsimpy.simulator import CSAD_DP_6DOF, RVG_DP_6DOF
from mclsimpy.utils import J

import os

# ─────────────────────────────────────────────
# 1)  user-tweak: how much to lift everything?
# ─────────────────────────────────────────────
Z_OFFSET = 1.5          # metres ↑   ← just change this number

# ─────────────────────────────────────────────
# 2)  (rest of your code, unchanged until plot loop)
# ─────────────────────────────────────────────
fullscale = True
fps   = 20 if fullscale else 45
dt    = 1 / fps
time  = np.arange(0, 50, dt)

writer = PillowWriter(fps=fps,
                      metadata=dict(title="Wave", artist="Hygen"))

if fullscale:
    Lpp, B, T = 33, 9.6, 2.786
else:
    Lpp, B, T = 2.578, 0.3, 0.2

L, H, scale = Lpp/2, T, 3/7
points = np.array([
    [-L, -B, -H], [-L, -B,  H], [-L,  B,  H], [-L,  B, -H],
    [ L*scale,  B, -H], [ L*scale,  B,  H], [ L*scale, -B,  H], [ L*scale, -B, -H],
    [ L, 0, -H], [ L, 0,  H],
    [ L*scale,  B,  H], [ L*scale,  B, -H], [ L, 0, -H], [ L*scale, -B, -H],
    [-L, -B, -H], [-L, -B,  H], [ L*scale, -B,  H], [ L, 0,  H], [ L*scale,  B,  H], [-L,  B,  H]
])
points[:, 2] = -points[:, 2]                        # keep your sign convention

vessel = (RVG_DP_6DOF if fullscale else CSAD_DP_6DOF)(dt, method="RK4",
                                                      config_file="rvg_config.json")
hs, tp  = (2.5, 12.0) if fullscale else (0.06, 1.9)
wp      = 2*np.pi/tp
Nw      = 100
w       = np.linspace(wp/2, wp*3, Nw)
k       = w**2/9.81

Nx = Ny = 250
xlim = 3.5*L
ylim = xlim
x = np.linspace(-xlim, xlim, Nx)
y = np.linspace(-ylim, ylim, Ny)
X, Y = np.meshgrid(x, y)

jonswap         = JONSWAP(w)
_, spectrum     = jonswap(hs, tp, gamma=3.3)
wave_amps       = np.sqrt(2*spectrum*(w[1]-w[0]))
wave_angle      = -np.pi*np.ones(Nw)
eps             = np.random.uniform(0, 2*np.pi, Nw)

waveload = WaveLoad(wave_amps, freqs=w, eps=eps, angles=wave_angle,
                    config_file=vessel._config_file,
                    interpolate=True, qtf_method="geo-mean")

def wave_elevation(t):
    phase = w[:,None,None]*t - k[:,None,None]*(X*np.cos(wave_angle)[:,None,None]
                                              +Y*np.sin(wave_angle)[:,None,None]) - eps[:,None,None]
    return (wave_amps[:,None,None]*np.cos(phase)).sum(axis=0)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(12,12))
ls      = LightSource(azdeg=0, altdeg=60)

vessel.set_eta(np.array([17., 0., 0., 0., 0., 0.]))  # your original start

# ─────────────────────────────────────────────
# 3)  integration to remove transients
# ─────────────────────────────────────────────
for t0 in np.arange(-100, 0, dt):
    vessel.integrate(0., 0., waveload(t0, vessel.get_eta()))

# --------------------------------------------------------------
# helper – wave elevation at a single point (scalar)
# put this near wave_elevation()
def zeta_at_point(t, x0, y0):
    phase = w * t - k * (x0*np.cos(wave_angle) + y0*np.sin(wave_angle)) - eps
    return float(np.sum(wave_amps * np.cos(phase)))
# --------------------------------------------------------------


# ──────────────────────────────────────────────────────────────
# 4)  animation  (REPLACES your original with writer.saving …)
# ──────────────────────────────────────────────────────────────
gif_name = os.path.join(os.path.dirname(__file__),
                        "vessel_motion3d__rvg_waveangle_180.gif")

with writer.saving(fig, gif_name, dpi=100):
    for ti in time:

        # integrate dynamics ----------------------------------------------------
        vessel.integrate(0., 0., waveload(ti, vessel.get_eta()))
        eta = vessel.get_eta()                       # [x, y, z, φ, θ, ψ]

        # sea surface on the whole grid ----------------------------------------
        ζ_grid = wave_elevation(ti) + Z_OFFSET       # lift water too
        rgb    = ls.shade(ζ_grid, cmap=colormaps["Blues"])
        surf   = ax.plot_surface(X, Y, ζ_grid, alpha=0.7,
                                 facecolors=rgb, linewidth=0,
                                 vmin=-.7, vmax=7, antialiased=False)

        # local wave height at vessel position ---------------------------------
        ζ_centre = zeta_at_point(ti, eta[0], eta[1]) + Z_OFFSET

        # transform stick-model hull -------------------------------------------
        R     = J(eta)[:3, :3]                       # rotation matrix
        vpts  = (points @ R.T) + eta[:3]             # rotate & translate
        vpts[:, 2] += ζ_centre                      # ←▲ shift up to surface

        ax.plot(vpts[:, 0], vpts[:, 1], vpts[:, 2],
                'r-', lw=3.5)

        # little red CG dot -----------------------------------------------------
        ax.plot([eta[0]], [eta[1]], [eta[2] + ζ_centre],
                'ro', ms=6)

        # axes housekeeping -----------------------------------------------------
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_zlim(-4.1*H + Z_OFFSET, 2.1*H + Z_OFFSET)
        # ax.invert_zaxis()    # ❌ removed – “positive-up” is clearer
        ax.view_init(30, 30)
        plt.suptitle(f"t = {ti:4.2f} s")

        writer.grab_frame()

        # tidy for next frame ---------------------------------------------------
        surf.remove()
        for ln in list(ax.lines):          # remove hull + CG dot
            ln.remove()
