import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Use Matplotlib's built-in mathtext (no external LaTeX required)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
# Try to use Computer Modern Roman; if not available, fallback to DejaVu Serif.
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11

# Number of players (adjust as needed)
N = 4

def get_trajectory(data, sim_steps="all"):
    r"""
    Extract trajectories, goals, and the ego player's mask from `data`.
    Skips any player that lacks an 'Initial State'.
    """
    trajectories = {}
    goals = {}
    for playerid in range(N):
        key_initial = f"Player {playerid+1} Initial State"
        if not data.get(key_initial):
            data.pop(key_initial, None)
            data.pop(f"Player {playerid+1} Goal", None)
            data.pop(f"Player {playerid+1} Trajectory", None)
            data.pop(f"Player {playerid+1} Control", None)
            continue
        traj_key = f"Player {playerid+1} Trajectory"
        if sim_steps == "all":
            trajectories[str(playerid+1)] = data[traj_key]
        else:
            if not isinstance(sim_steps, int):
                raise ValueError("sim_steps must be an integer or 'all'")
            trajectories[str(playerid+1)] = data[traj_key][:sim_steps]
        goals[str(playerid+1)] = data[f"Player {playerid+1} Goal"]
    
    # Retrieve the mask for the ego (Player 1)
    masks = data["Player 1 Mask"]
    return trajectories, goals, masks

###############################################################################
#  CONFIGURATION
###############################################################################
# List of method files (each file corresponds to one row)

methods = [
  #  r"data_closer_test_cooperative\receding_horizon_trajectories_[174]_[Hessian]_[2].json"
 #   r"data_closer_test_cooperative\receding_horizon_trajectories_[174]_[Nearest Neighbor]_[2].json",
    r"data_closer_test_cooperative\receding_horizon_trajectories_[174]_[Neural Network Rank]_[3].json"
]

n_rows = len(methods)  # each method is one row
n_cols = 5             # 5 equally spaced time steps

# Create figure with shared axes so that all subplots have the same x and y scales.
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(15, 4.5 * n_rows),
    sharex=True,
    sharey=True)
if n_rows == 1:
    axes = np.array([axes])
if n_cols == 1:
    axes = axes[:, np.newaxis]

# Set constant axis limits for every plot.
x_lim = (-3.5, 3.5)
y_lim = (-3.5, 3.5)

# Time labels for bottom row (formatted in math mode)
time_labels = [
    r"$t=1\,\mathrm{s}$",
    r"$t=2\,\mathrm{s}$",
    r"$t=3\,\mathrm{s}$",
    r"$t=4\,\mathrm{s}$",
    r"$t=5\,\mathrm{s}$",
]


###############################################################################
#  LEGEND (using lighter colors)
###############################################################################
# Lighter color definitions using hex codes:
color_ego      = "#66B3FF"  # light blue for Ego (Player 1)
color_other_on = "#FF9999"  # light red for Other (mask active)
color_other_off= "#99FF99"  # light green for Other (mask inactive)

blue_ego    = Line2D([], [], color=color_ego, marker='o', markersize=8,
                     linewidth=2, label=r"$\mathbf{Ego}$")
red_other   = Line2D([], [], color=color_other_on, marker='o', markersize=8,
                     linewidth=2, label=r"$\mathbf{Included \ in \ Game}$")
green_other = Line2D([], [], color=color_other_off, marker='o', markersize=8,
                     linewidth=2, label=r"$\mathbf{Excluded \ from \ Game}$")

fig.legend(
    handles=[blue_ego, red_other, green_other],
    loc='upper center',
    bbox_to_anchor=(0.5, .95),  # Move legend closer to the plots
    ncol=3,
    fontsize=12
)

###############################################################################
#  MAIN PLOTTING LOOP
###############################################################################
for row, method_file in enumerate(methods):
    # Load JSON data
    with open(method_file, 'r') as f:
        data = json.load(f)

    # Extract a clean method name from the filename.
    base_name = os.path.basename(method_file)
    name_no_ext = os.path.splitext(base_name)[0]
    parts = name_no_ext.split("_[")
    if len(parts) >= 3:
        method_label = parts[2].replace("]", "")
    else:
        method_label = name_no_ext
    # Wrap the method label in mathmode for bold formatting.
    method_label = r"$\mathbf{" + method_label + r"}$"

    trajectories, goals, masks = get_trajectory(data, sim_steps="all")
    total_steps = len(trajectories["1"])  # assume all players have same number of steps

    # Choose 5 equally spaced step indices.
    step_indices = np.array([t * 10 for t in range(1, 6)], dtype=int)

    for col, step in enumerate(step_indices):
        ax = axes[row, col]
        
        # Plot each player's trajectory (from time=0 to current 'step')
        for p_str, traj in trajectories.items():
            p_id = int(p_str)
            
            # Determine current marker color based on mask at the current time step.
            if step < len(masks) and masks[step][p_id - 1] == 1:
                current_color = color_ego if p_id == 1 else color_other_on
            else:
                current_color = color_ego if p_id == 1 else color_other_off
            
            # Plot full trajectory history from the beginning up to 'step'
            #start_idx = max(0, step - 9)
            start_idx = 0 # plot all previous steps
            for idx in range(start_idx, step):
                if idx + 1 >= len(traj):
                    break
                if idx < len(masks) and masks[idx][p_id - 1] == 1:
                    segment_color = color_ego if p_id == 1 else color_other_on
                else:
                    segment_color = color_ego if p_id == 1 else color_other_off
                x_vals = [traj[idx][0], traj[idx+1][0]]
                y_vals = [traj[idx][1], traj[idx+1][1]]
                ax.plot(x_vals, y_vals, color=segment_color, linewidth=1.5)
            
            # Plot the current position marker.
            if step < len(traj):
                ax.plot(traj[step][0], traj[step][1],
                        marker='o', color=current_color, markersize=8)
        
        # Set the axis limits and equal aspect
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        
        # Only bottom row subplots show the x-axis label and tick labels.
        if row == n_rows - 1:
            # ax.set_xlabel(r"$X\ \mathrm{Position}$")
            ax.set_xticks(np.arange(-3, 4, 1))
            # Place the time label beneath the subplot.
            if col < len(time_labels):
                ax.annotate(
                    time_labels[col],
                    xy=(0.5, -0.15),
                    xycoords='axes fraction',
                    ha='center',
                    va='center',
                    fontsize=11
                )
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        
        # Only leftmost column subplots show the y-axis label and tick labels.
        if col == 0:
            # ax.set_ylabel(r"$Y\ \mathrm{Position}$")
            # Place the method label to the left (rotated vertically).
            ax.annotate(
                method_label,
                xy=(-0.1, 0.5),
                xycoords='axes fraction',
                ha='center',
                va='center',
                rotation=90,
                fontsize=12,
                fontweight='bold'
            )
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.05)
# Save the plot as a PDF
plt.tight_layout()
plt.savefig("trajectories_grid_lighter_colors_latex.pdf", dpi=1000, bbox_inches='tight')
# plt.show()
