#!/usr/bin/env python3
"""
MC-GYM-JAX Simulation with Meta-Learned Adaptive Controller

This script demonstrates using the meta-learned adaptive controller (Spencer M. Richards'
version) as the policy within the MC-GYM-JAX environment. The meta-adaptive controller
computes a 3DOF control action by updating its adaptive parameters at each time step.
"""

import time
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np  # for converting to numpy arrays for plotting if needed

# MC-GYM-JAX environment
from jax_core.gym.mc_gym_csad_jax import McGymJAX
# Conversion utilities and parameter helper
from jax_core.utils import six2threeDOF, params_to_posdef
# For computing prior dynamics (used in controller)
from jax_core.meta_adaptive_ctrl.dynamics import prior_3dof

# --- Meta Adaptive Controller Functions ---

def adaptation_law(q, dq, r, dr, params):
    """
    Compute the adaptation law.
    
    Args:
        q   : Current 3DOF position.
        dq  : Current 3DOF velocity.
        r   : Desired 3DOF position.
        dr  : Desired 3DOF velocity.
        params: Dictionary of meta learned controller parameters.
        
    Returns:
        dA : Adaptation derivative (matrix).
        y  : Regressor features.
    """
    # Regressor features
    y = jnp.concatenate((q, dq))
    for W, b in zip(params['W'], params['b']):
        y = jnp.tanh(W @ y + b)
    Λ, P = params['Λ'], params['P']
    e = q - r
    de = dq - dr
    s = de + Λ @ e
    dA = P @ jnp.outer(s, y)
    return dA, y

def controller(q, dq, r, dr, ddr, f_hat, params):
    """
    Compute the control input using the meta-adaptive controller.
    
    Args:
        q     : Current 3DOF position.
        dq    : Current 3DOF velocity.
        r     : Desired 3DOF position.
        dr    : Desired 3DOF velocity.
        ddr   : Desired 3DOF acceleration.
        f_hat : Estimated disturbance/force using adaptation.
        params: Dictionary of meta learned controller parameters.
        
    Returns:
        u   : Control input (3DOF).
        τ   : Intermediate computed control force.
    """
    Λ, K = params['Λ'], params['K']
    e = q - r
    de = dq - dr
    s = de + Λ @ e
    v = dr - Λ @ e
    dv = ddr - Λ @ de
    # Obtain prior dynamics matrices (M, D, G, R) for the 3DOF system.
    M, D, G, R = prior_3dof(q, dq)
    tau = M @ dv + D @ v + G @ q - f_hat - K @ s
    u = jnp.linalg.solve(R, tau)
    return u, tau

# --- Main Simulation with MC-GYM-JAX and Meta Adaptive Controller ---

def main():
    # Simulation parameters
    dt = 0.01
    simtime = 300  # seconds
    max_steps = int(simtime / dt)
    
    # Create the MC-GYM-JAX environment.
    # Set render_on=True to visualize (requires pygame).
    env = McGymJAX(dt=dt, grid_width=15, grid_height=6, render_on=False, final_plot=True)
    
    # Task configuration: Four-corner test.
    start_position = (2.0, 2.0, 0.0)  # (north, east, heading in degrees)
    wave_conditions = (0.03, 1.5, 45)   # (significant wave height, peak period, wave direction in degrees)
    env.set_task(start_position=start_position,
                 wave_conditions=wave_conditions,
                 four_corner_test=True,
                 simtime=simtime,
                 ref_omega=[0.2, 0.2, 0.2])
    
    # Load meta learned controller parameters.
    # These are assumed to be stored in a pickle file.
    seed = 0
    M_val = 10  # Example value used during training
    filename = os.path.join('data', 'training_results', f'seed={seed}_M={M_val}.pkl')
    with open(filename, 'rb') as f:
        train_results = pickle.load(f)
    params = {
        'W': train_results['model']['W'],
        'b': train_results['model']['b'],
        'Λ': params_to_posdef(train_results['controller']['Λ']),
        'K': params_to_posdef(train_results['controller']['K']),
        'P': params_to_posdef(train_results['controller']['P']),
    }
    
    # Initialize the adaptive controller state.
    # Get the initial vessel state and convert to 3DOF.
    state = env.get_state()
    q = six2threeDOF(state["eta"])
    dq = six2threeDOF(state["nu"])
    # Get the initial reference from the onboard filter.
    eta_d, eta_d_dot, eta_d_ddot, _ = env.get_four_corner_nd(0)
    r = eta_d
    dr = eta_d_dot
    ddr = eta_d_ddot
    # Compute initial adaptation law.
    dA, y = adaptation_law(q, dq, r, dr, params)
    A = jnp.zeros((q.size, y.size))  # Initialize adaptive parameter matrix.
    dA_prev = dA  # Set initial previous derivative.
    
    print("Starting simulation with meta-learned adaptive controller...")
    start_time = time.time()
    
    for step in range(1, max_steps):
        # Update reference from the four-corner test filter.
        eta_d, eta_d_dot, eta_d_ddot, _ = env.get_four_corner_nd(step)
        r = eta_d
        dr = eta_d_dot
        ddr = eta_d_ddot
        
        # Retrieve current vessel state and convert to 3DOF.
        state = env.get_state()
        q = six2threeDOF(state["eta"])
        dq = six2threeDOF(state["nu"])
        
        # Compute the adaptation law.
        dA, y = adaptation_law(q, dq, r, dr, params)
        # Update adaptive parameters using the trapezoidal rule.
        A = A + dt * (dA_prev + dA) / 2.0
        # Estimate disturbance/uncertainty.
        f_hat = A @ y
        # Compute control input using the controller.
        u, tau = controller(q, dq, r, dr, ddr, f_hat, params)
        # Save the current adaptation derivative for next step.
        dA_prev = dA
        
        # Apply the 3DOF control input.
        state, done, info, reward = env.step(u)
        
        # Optionally print debug information every 50 steps.
        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  q (actual 3DOF): {q}")
            print(f"  r (desired 3DOF): {r}")
            print(f"  Control action (u): {u}")
            print(f"  Computed tau: {tau}")
        
        if done:
            print("Simulation terminated:", info)
            break
    
    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print("Simulation completed.")
    
    # Post-simulation: Plot trajectories and reference states.
    env.plot_trajectory()

if __name__ == "__main__":
    main()
