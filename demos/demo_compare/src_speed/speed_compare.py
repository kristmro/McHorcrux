#!/usr/bin/env python
import argparse
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import jax, jax.numpy as jnp
import torch

from jax_core.simulator.vessels.csad_jax import load_csad_parameters, csad_x_dot
from jax_core.utils import rk4_step
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, wave_load

from torch_core.simulator.vessels.csad_torch import CSAD_6DOF as CSAD_DP_6DOF_Torch
from torch_core.simulator.waves.wave_load_torch import WaveLoad as WaveLoadTorch
from torch_core.simulator.waves.wave_spectra_torch import JONSWAP as JONSWAPTorch

from mclsimpy.simulator import CSAD_DP_6DOF
from mclsimpy.waves import JONSWAP, WaveLoad

from functools import lru_cache

# path to your JSON config
CONFIG_FILE = Path('data/vessel_data/csad/csad.json').resolve()

# --- argument parsing ---
parser = argparse.ArgumentParser(
    description="Chunked benchmark of JAX, NumPy & PyTorch CSAD simulators"
)
parser.add_argument("--N",   type=int,   help="only run for this N (e.g. 40)")
parser.add_argument("--dt",  type=float, help="only run for this dt (e.g. 0.05)")
parser.add_argument("--st",  type=int,   help="only run for this simtime (e.g. 80)")
parser.add_argument("--out", type=str,   default="speed_compare.csv",
                    help="CSV file to append to")
args = parser.parse_args()

# full param grids
dts      = [0.1, 0.05, 0.01, 0.005]
Ns       = [40, 80, 160, 320]
simtimes = [40, 80, 160, 320]

# filter by args
Ns_run       = [args.N] if args.N else Ns
dts_run      = [args.dt] if args.dt else dts
simtimes_run = [args.st] if args.st else simtimes

# --- cached wave loaders ---
@lru_cache(maxsize=None)
def wave_jax(N):
    hs,tp,gamma = 5.0/90.0, 19.0*np.sqrt(1/90), 3.3
    wp = 2*np.pi/tp
    wmin,wmax = 0.5*wp, 3.0*wp
    freqs = jnp.linspace(wmin, wmax, N)
    _,S = jonswap_spectrum(freqs, hs, tp, gamma=gamma, freq_hz=False)
    dw = (wmax - wmin) / N
    amps = jnp.sqrt(2*S*dw)
    phases = jax.random.uniform(jax.random.PRNGKey(0), shape=(N,), minval=0, maxval=2*jnp.pi)
    angles = jnp.ones(N) * (jnp.pi/4)
    return init_wave_load(amps, freqs, phases, angles,
                          config_file=str(CONFIG_FILE), rho=1025, g=9.81,
                          dof=6, depth=100, deep_water=True,
                          qtf_method="Newman", qtf_interp_angles=True,
                          interpolate=True)

@lru_cache(maxsize=None)
def wave_numpy(N):
    hs,tp,gamma = 5.0/90.0, 19.0*np.sqrt(1/90), 3.3
    wp = 2*np.pi/tp
    wmin,wmax = 0.5*wp, 3.0*wp
    freqs = np.linspace(wmin, wmax, N)
    jons = JONSWAP(freqs)
    _,S = jons(hs=hs, tp=tp, gamma=gamma)
    dw = (wmax - wmin) / N
    amps = np.sqrt(2*S*dw)
    phases = np.random.default_rng(0).uniform(0, 2*np.pi, size=N)
    angles = np.ones(N) * (np.pi/4)
    return WaveLoad(amps, freqs, phases, angles,
                    config_file=str(CONFIG_FILE), interpolate=True,
                    qtf_method="Newman", deep_water=True)

@lru_cache(maxsize=None)
def wave_torch(N):
    hs,tp,gamma = 5.0/90.0, 19.0*np.sqrt(1/90), 3.3
    wp = 2*np.pi/tp
    wmin,wmax = 0.5*wp, 3.0*wp
    freqs = torch.linspace(float(wmin), float(wmax), N)
    jons = JONSWAPTorch(freqs)
    _,S = jons(hs, tp, gamma)
    dw = (wmax - wmin) / N
    amps = torch.sqrt(2*S*dw)
    phases = torch.rand(N) * (2*np.pi)
    angles = torch.ones(N) * (np.pi/4)
    return WaveLoadTorch(amps, freqs, phases, angles,
                         config_file=str(CONFIG_FILE), interpolate=True,
                         qtf_method="Newman", deep_water=True)

# --- runner factories ---
def jax_runner_factory(N, dt):
    wl = wave_jax(N)
    params = load_csad_parameters(str(CONFIG_FILE))
    x0 = jnp.zeros(12)
    tau_c = jnp.zeros(6)
    def step(x, t):
        eta = x[:6]
        tau = tau_c + wave_load(t, eta, wl)
        return rk4_step(x, dt, csad_x_dot, 0.0, 0.0, tau, params), None

    @jax.jit
    def simulate(t_arr):
        final, _ = jax.lax.scan(step, x0, t_arr)
        return final

    def run(simtime):
        simulate(jnp.arange(0, simtime, dt)).block_until_ready()
    return run

def numpy_runner_factory(N, dt):
    wl = wave_numpy(N)
    vessel = CSAD_DP_6DOF(dt=dt, method='RK4')
    vessel.set_eta(np.zeros(6)); vessel.set_nu(np.zeros(6))
    tau_c = np.zeros(6)
    def run(simtime):
        for t in np.arange(0, simtime, dt):
            tau = tau_c + wl(t, vessel.get_eta())
            vessel.integrate(0.0, 0.0, tau)
    return run

def torch_runner_factory(N, dt):
    wl = wave_torch(N)
    vessel = CSAD_DP_6DOF_Torch(dt=dt, method='RK4', config_file=str(CONFIG_FILE))
    vessel.set_eta(torch.zeros(6)); vessel.set_nu(torch.zeros(6))
    tau_c = torch.zeros(6)
    def loop(simtime):
        for t in torch.arange(0, simtime, dt):
            tau = tau_c + wl(t, vessel.get_eta())
            vessel.integrate(0.0, 0.0, tau)
    return torch.compile(loop, mode='reduce-overhead')

# --- benchmarking helper ---
def benchmark(fn, simtime):
    fn(simtime)             # warm-up / compile
    t0 = time.perf_counter()
    fn(simtime)
    gc.collect()
    return time.perf_counter() - t0

# --- main loop & CSV output ---
records = []
for N in Ns_run:
    for dt in dts_run:
        jax_run   = jax_runner_factory(N, dt)
        numpy_run = numpy_runner_factory(N, dt)
        torch_run = torch_runner_factory(N, dt)
        for st in simtimes_run:
            tj = benchmark(jax_run,   simtime=st)
            tn = benchmark(numpy_run, simtime=st)
            tp = benchmark(torch_run, simtime=st)
            records.append({
                "N": N, "dt": dt, "simtime": st,
                "jax_time": tj, "numpy_time": tn, "torch_time": tp
            })
            print(f"N={N} dt={dt:<6g} st={st:4d} | JAX {tj:.3f}s NumPy {tn:.3f}s PT {tp:.3f}s")

# append to CSV
out_path = Path(args.out)
df = pd.DataFrame(records)
df.to_csv(
    out_path,
    mode=('a' if out_path.exists() else 'w'),
    header=(not out_path.exists()),
    index=False
)
