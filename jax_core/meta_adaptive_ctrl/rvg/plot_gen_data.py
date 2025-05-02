#!/usr/bin/env python3
"""
plot_data_gen.py

This script loads training data generated from the simulation study
and produces academic-style plots to document:
  - The statistical distributions of the wave parameters (wave height, period, and direction).
  - A 2D trajectory plot comparing the desired reference and the actual trajectory.
  - Time-series plots for the control input "u" for each degree of freedom.

All figures are saved in the folder: figures/data_gen.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the simulation training data from a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def plot_wave_params(data, save_dir):
    """
    Plot the wave parameters describing the training data.
    
    Creates three histograms for:
      - Wave Height (hs)
      - Wave Period (tp)
      - Wave Direction (wave_dir)
    
    All histograms are arranged side by side.
    """
    wave_parm = data["wave_parm"]
    hs, tp, wave_dir = wave_parm

    # Ensure the parameters are numpy arrays.
    hs = np.array(hs)
    tp = np.array(tp)
    wave_dir = np.array(wave_dir)

    # Create a figure with three subplots.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot histogram for Wave Height.
    axs[0].hist(hs, bins=20, color='blue', alpha=0.7)
    axs[0].set_title("Distribution of Wave Heights")
    axs[0].set_xlabel("Wave Height (scaled)")
    axs[0].set_ylabel("Frequency")
    axs[0].grid(True)

    # Plot histogram for Wave Period.
    axs[1].hist(tp, bins=20, color='green', alpha=0.7)
    axs[1].set_title("Distribution of Wave Periods")
    axs[1].set_xlabel("Wave Period (scaled)")
    axs[1].set_ylabel("Frequency")
    axs[1].grid(True)

    # Plot histogram for Wave Direction.
    axs[2].hist(wave_dir, bins=20, color='red', alpha=0.7)
    axs[2].set_title("Distribution of Wave Directions")
    axs[2].set_xlabel("Wave Direction (deg)")
    axs[2].set_ylabel("Frequency")
    axs[2].grid(True)

    plt.suptitle("Wave Parameters from Training Data", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure.
    save_path = os.path.join(save_dir, "wave_parameters.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_simulation_examples(data, save_dir):
    """
    Plot simulation examples from the training data, including:
      - A 2D trajectory comparison of the desired and actual paths.
      - A time series plot of the control input "u" for each degree of freedom.
    
    The plots are saved as separate figures.
    """
    # Extract the time, positions (q and r) and control inputs (u)
    t = np.array(data["t"])
    q = np.array(data["q"])
    r = np.array(data["r"])
    u = np.array(data["u"])

    # In case the data is batched (3D array), choose one simulation instance.
    if q.ndim == 3:
        idx = 1  # selected instance index; this can be made adjustable if desired
        q_plot = q[idx]
        r_plot = r[idx]
        u_plot = u[idx]
    else:
        q_plot = q
        r_plot = r
        u_plot = u

    # ---------------------- 2D Trajectory Plot ----------------------
    # Assumes the first two degrees of freedom are the spatial coordinates (x and y).
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(q_plot[:, 0], q_plot[:, 1], label='Actual Trajectory', linewidth=2)
    ax1.plot(r_plot[:, 0], r_plot[:, 1], '--', label='Desired Reference Trajectory', linewidth=2)
    ax1.set_title("2D Trajectory Comparison: Actual vs. Desired", fontsize=14)
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()

    traj_save_path = os.path.join(save_dir, "xy_trajectory.png")
    plt.savefig(traj_save_path, dpi=300)
    plt.close(fig1)

    # ---------------------- Control Input Time Series ----------------------
    dof_names = ['x', 'y', 'yaw']
    fig2, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[i].plot(t, u_plot[:, i], label=f'{dof_names[i]} Control Input')
        axs[i].set_ylabel(f'{dof_names[i]} Input')
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Time [s]")
    fig2.suptitle("Control Inputs over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    u_save_path = os.path.join(save_dir, "control_input.png")
    plt.savefig(u_save_path, dpi=300)
    plt.close(fig2)

if __name__ == "__main__":
    # Define file paths.
    data_path = 'data/training_data/rvg_training_data_wave_pm45_N15_hs5.pkl'
    save_dir = "figures/rvg/data_gen"
    
    # Create the output directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the training data.
    data = load_data(data_path)

    # Generate and save plots for the wave parameters.
    plot_wave_params(data, save_dir)
    
    # Generate and save plots for simulation examples.
    plot_simulation_examples(data, save_dir)
    
    print(f"All academic-style figures have been saved in '{save_dir}'.")
