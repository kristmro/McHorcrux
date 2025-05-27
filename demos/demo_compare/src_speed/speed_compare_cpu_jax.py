#!/usr/bin/env python
"""
speed_compare_cpu_jax.py

Run the CSAD benchmark with **JAX on the CPU only** and append / update the
'jax_cpu' column in speed_compare.csv.

Usage examples
--------------
# run the full grid, add only jax_cpu results that are missing
python speed_compare_cpu_jax.py

# re-run for one point and overwrite any existing jax_cpu value
python speed_compare_cpu_jax.py --N 80 --dt 0.05 --st 160 --force

# write to a different CSV
python speed_compare_cpu_jax.py --out my_results.csv
"""

# ---- stdlib & third-party -------------------------------------------------
import argparse
import gc
import time
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# ---- project imports ------------------------------------------------------
from jax_core.simulator.vessels.csad_jax import load_csad_parameters, csad_x_dot
from jax_core.utils import rk4_step
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, wave_load
# ---- force JAX → CPU *before* importing jax -------------------------------
jax.config.update("jax_platform_name", "cpu")
# ---------------------------------------------------------------------------
CONFIG_FILE = Path("data/vessel_data/csad/csad.json").resolve()

# ---- argument parsing -----------------------------------------------------
parser = argparse.ArgumentParser(
    description="(CPU) benchmark for JAX only – adds the jax_cpu column"
)
parser.add_argument("--N",   type=int,   help="only run for this N (e.g. 40)")
parser.add_argument("--dt",  type=float, help="only run for this dt (e.g. 0.05)")
parser.add_argument("--st",  type=int,   help="only run for this simtime (e.g. 80)")
parser.add_argument("--out", type=str,   default="speed_compare.csv",
                    help="CSV file to read & write")
parser.add_argument("--force", action="store_true",
                    help="overwrite jax_cpu even if it is already present")
args = parser.parse_args()

# ---- parameter grids (same as original script) ----------------------------
dts      = [0.1, 0.05, 0.01, 0.005]
Ns       = [40, 80, 160, 320]
simtimes = [40, 80, 160, 320]

Ns_run       = [args.N]  if args.N  else Ns
dts_run      = [args.dt] if args.dt else dts
simtimes_run = [args.st] if args.st else simtimes

# ---- cached wave loader ---------------------------------------------------
@lru_cache(maxsize=None)
def wave_jax(N):
    hs, tp, gamma = 5.0/90.0, 19.0*np.sqrt(1/90), 3.3
    wp = 2*np.pi / tp
    wmin, wmax = 0.5*wp, 3.0*wp
    freqs = jnp.linspace(wmin, wmax, N)
    _, S  = jonswap_spectrum(freqs, hs, tp, gamma=gamma, freq_hz=False)
    dw    = (wmax - wmin) / N
    amps  = jnp.sqrt(2*S*dw)
    phases = jax.random.uniform(jax.random.PRNGKey(0), shape=(N,),
                                minval=0.0, maxval=2*jnp.pi)
    angles = jnp.ones(N) * (jnp.pi/4)
    return init_wave_load(amps, freqs, phases, angles,
                          config_file=str(CONFIG_FILE), rho=1025, g=9.81,
                          dof=6, depth=100, deep_water=True,
                          qtf_method="Newman", qtf_interp_angles=True,
                          interpolate=True)

# ---- JAX runner factory ---------------------------------------------------
def jax_runner_factory(N, dt):
    wl     = wave_jax(N)
    params = load_csad_parameters(str(CONFIG_FILE))
    x0     = jnp.zeros(12)
    tau_c  = jnp.zeros(6)

    def step(x, t):
        eta  = x[:6]
        tau  = tau_c + wave_load(t, eta, wl)
        return rk4_step(x, dt, csad_x_dot, 0.0, 0.0, tau, params), None

    @jax.jit
    def simulate(t_arr):
        final, _ = jax.lax.scan(step, x0, t_arr)
        return final

    def run(simtime):
        simulate(jnp.arange(0, simtime, dt)).block_until_ready()

    return run

# ---- benchmarking helper --------------------------------------------------
def benchmark(fn, simtime):
    fn(simtime)                   # warm-up / compile
    t0 = time.perf_counter()
    fn(simtime)
    gc.collect()
    return time.perf_counter() - t0

# ---- load existing CSV (if any) -------------------------------------------
csv_path = Path(args.out)
if csv_path.exists():
    df_all = pd.read_csv(csv_path)
else:
    df_all = pd.DataFrame()

# ---- choose rows to (re-)run ---------------------------------------------
def needs_run(N, dt, st):
    """True if there is no jax_cpu value yet or --force was given."""
    mask = (df_all["N"] == N) & (df_all["dt"] == dt) & (df_all["simtime"] == st)
    if not mask.any():
        return True
    if args.force:
        return True
    # row exists and not forcing – only run if value is NaN
    val = df_all.loc[mask, "jax_cpu"]
    return val.isna().any()

# ---- main loop ------------------------------------------------------------
new_records = []
for N in Ns_run:
    for dt in dts_run:
        run_jax = jax_runner_factory(N, dt)
        for st in simtimes_run:
            if not needs_run(N, dt, st):
                continue
            t_cpu = benchmark(run_jax, st)
            new_records.append({"N": N, "dt": dt, "simtime": st,
                                "jax_cpu": t_cpu})
            print(f"N={N} dt={dt:<6g} st={st:4d} | JAX-CPU {t_cpu:.3f}s")

# ---- merge & write --------------------------------------------------------
if new_records:
    df_new = pd.DataFrame(new_records)
    if not df_all.empty:
        df_all = df_all.merge(df_new, how="outer",
                              on=["N", "dt", "simtime"],
                              suffixes=("", "_new"))
        # keep the freshly computed jax_cpu where available
        df_all["jax_cpu"] = df_all["jax_cpu_new"].combine_first(df_all["jax_cpu"])
        df_all = df_all.drop(columns=["jax_cpu_new"])
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved updated results to {csv_path.resolve()}")
else:
    print("Nothing to do – all requested rows already contain jax_cpu.")
