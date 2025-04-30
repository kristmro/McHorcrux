import os
import re
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_best_index(
        results_root="data/training_results",
        act="off",
        ctrl_pen=2,
        figsize=(10, 5),
        cmap="viridis",
        save_to_figures=True):
    """
    Scan `results_root/act_<act>/ctrl_pen_<ctrl_pen>` for the pickles that
    `training_diag.py` wrote, extract the ``best_step_idx`` recorded in each,
    and plot them as a heat‑map (seeds on the y‑axis, M on the x‑axis).

    Parameters
    ----------
    results_root : str
        Directory under which the run‑specific folders live.
    act : str
        The value of the ``act`` flag you trained with ("off" by default).
    ctrl_pen : int
        The –log₁₀(ctrl_penalty) used when naming the sub‑folder.
    figsize : tuple
        Matplotlib figure size.
    cmap : str
        Colormap for the heat‑map.
    save_to_figures : bool
        Whether to save the figure to the figures directory.

    Notes
    -----
    • The function assumes result filenames follow the pattern
      ``seed=<int>_M=<int>.pkl`` exactly as produced in your script.  
    • If you trained with several ctrl_pen or act settings, just call the
      function again with the appropriate values.
    """

    run_dir = Path(results_root) / f"act_{act}" / f"ctrl_pen_{ctrl_pen}"
    if not run_dir.is_dir():
        # Corrected f-string syntax and used Path object for consistency
        raise FileNotFoundError(f"Directory '{run_dir}' does not exist")

    # ------------------------------------------------------------------ #
    # Collect best_step_idx for every (seed, M)                          #
    # ------------------------------------------------------------------ #
    pattern = re.compile(r"seed=(\d+)_M=(\d+)\.pkl")
    data = defaultdict(dict)       # data[seed][M] = best_idx

    # Use Path object for iteration
    for fpath in run_dir.glob("*.pkl"):
        match = pattern.fullmatch(fpath.name)
        if not match:
            continue                              # skip anything unexpected
        seed, M = map(int, match.groups())
        # Use Path object for opening file
        with open(fpath, "rb") as f:
            try:
                loaded_data = pickle.load(f)
                # Check if 'best_step_idx' exists in the loaded data
                if "best_step_idx" in loaded_data:
                    best_idx = loaded_data["best_step_idx"]
                    data[seed][M] = int(best_idx)
                else:
                    print(f"Warning: 'best_step_idx' not found in {fpath.name}")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not load pickle file {fpath.name}: {e}")


    if not data:
        # Corrected f-string syntax
        raise RuntimeError(f"No matching result files with 'best_step_idx' found in '{run_dir}'")

    # ------------------------------------------------------------------ #
    # Organise into a dense 2‑D array (rows = seeds, cols = M values)    #
    # ------------------------------------------------------------------ #
    seeds = sorted(data.keys())
    # Ensure Ms are integers and sorted
    Ms    = sorted({int(M) for d in data.values() for M in d})
    Z = np.full((len(seeds), len(Ms)), np.nan)     # default NaN for gaps
    for i, s in enumerate(seeds):
        for j, M in enumerate(Ms):
            # Ensure M is used as integer key
            if M in data[s]:
                Z[i, j] = data[s][M]

    # ------------------------------------------------------------------ #
    # Plot                                                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)
    # Handle case where Z might be empty or all NaN
    if Z.size == 0 or np.all(np.isnan(Z)):
        print(f"Warning: No valid data to plot for {run_dir}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        im = None # No image to show
    else:
        im = ax.imshow(Z, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Best step index")

        # Annotate each cell with its value
        for i in range(len(seeds)):
            for j in range(len(Ms)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, int(Z[i, j]), ha="center", va="center", color="white" if Z[i,j] < (Z[~np.isnan(Z)].max() / 2) else "black", fontsize=8)


    # Cosmetic touches
    ax.set_xticks(np.arange(len(Ms))) # Use np.arange for clarity
    ax.set_xticklabels(Ms)
    ax.set_xlabel("$M$ (sub‑sampled trajectories)")

    ax.set_yticks(np.arange(len(seeds))) # Use np.arange for clarity
    ax.set_yticklabels(seeds)
    ax.set_ylabel("Seed")

    ax.set_title(f"Best optimisation step index\n(act={act}, ctrl_pen={ctrl_pen})") # More informative title
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly for title

    # Save figure if requested
    if save_to_figures:
        # Create figures directory if it doesn't exist
        figures_dir = Path("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with parameters
        filename = f"best_idx_act_{act}_ctrl_pen_{ctrl_pen}.jpeg"
        save_path = figures_dir / filename

        # Save with high resolution
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig, ax

if __name__ == "__main__":
    # Example usage - consider adding argument parsing for flexibility
    try:
        plot_best_index(
            results_root="data/training_results",
            act="off",
            ctrl_pen=1, # Example value, adjust as needed
            figsize=(10, 5),
            cmap="viridis",
            save_to_figures=True
        )
        # Add more calls for different parameters if needed
        # plot_best_index(ctrl_pen=2)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
