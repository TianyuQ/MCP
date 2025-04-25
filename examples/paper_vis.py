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

def clean_method_name_for_legend(method, option):
    # Replace specific parts of the method name with simplified names for the legend
    if option != "threshold":
        if "Neural Network Partial" in method:
            return "PSN-Partial"  # For Neural Network Partial methods, return 'PSN-Partial' for the legend
        elif "Neural Network" in method:
            return "PSN-Full"  # For Neural Network methods, return 'PSN' for the legend
        elif "Distance Threshold" in method:
            return "Distance"  # For Distance Threshold, return 'Distance' for the legend
        elif "Nearest Neighbor" in method:
            return "Distance"  # For Nearest Neighbor, return 'Distance' for the legend
        elif "Control Barrier Function" in method:
            return "CBF"  # For Control Barrier Function, return 'CBF' for the legend
        elif "Barrier Function" in method:
            return "BF"  # For Barrier Function, return 'BF' for the legend
        elif "Jacobian" in method:
            return "Jacobian"
        elif "Hessian" in method:
            return "Hessian"
        elif "Cost Evolution" in method:
            return "Cost Evolution"
    else:
        # For the threshold option, simplify the method name without keeping the parameter part
        if "Neural Network Partial" in method:
            return "PSN-Partial"
        elif "Neural Network Threshold" in method:
            return "PSN-Full-th"
        elif "Distance Threshold" in method:
            return "Distance"
        elif "Nearest Neighbor" in method:
            return "Distance"
        elif "Control Barrier Function" in method:
            return "CBF"
        elif "Barrier Function" in method:
            return "BF"
        elif "Jacobian" in method:
            return "Jacobian"
        elif "Hessian" in method:
            return "Hessian"
        elif "Cost Evolution" in method:
            return "Cost Evolution"

    return method  # Return the original method name if no change is needed

###############################################################################
#  CONFIGURATION
###############################################################################
# List of method files (each file corresponds to one row)

for scenario_id in range(160, 161):  # Loop over the range of scenarios
    methods = [
        f"data_test_4 _30\\receding_horizon_trajectories_[{scenario_id}]_[Neural Network Threshold]_[0.5].json",
        #f"data_test_4 _30\\receding_horizon_trajectories_[{scenario_id}]_[Neural Network Partial Threshold]_[0.5].json",
        f"data_test_4 _30\\receding_horizon_trajectories_[{scenario_id}]_[Neural Network Rank]_[2].json",
        f"data_test_4 _30\\receding_horizon_trajectories_[{scenario_id}]_[All]_[1].json"
        #f"data_test_4 _30\\receding_horizon_trajectories_[{scenario_id}]_[Neural Network Partial Rank]_[2].json", 
    ]

    n_rows = len(methods)  # each method is one row
    n_cols = 5             # NUMBER OF COLUMNS

    # Create figure with shared axes so that all subplots have the same x and y scales.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        sharex=True,
        sharey=True,
        gridspec_kw = {'wspace':0, 'hspace':0}
        )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Set constant axis limits for every plot.
    # Determine adaptive limits based on the trajectories
    all_x = []
    all_y = []
    for method_file in methods:
        with open(method_file, 'r') as f:
            data = json.load(f)
        trajectories, _, _ = get_trajectory(data, sim_steps="all")
        for traj in trajectories.values():
            for point in traj:
                all_x.append(point[0])
                all_y.append(point[1])

    # Calculate limits with some padding
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    padding = 0.5  # Add some padding to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)  # Ensure aspect ratio remains the same

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_lim = (x_center - max_range / 2 - padding, x_center + max_range / 2 + padding)
    y_lim = (y_center - max_range / 2 - padding, y_center + max_range / 2 + padding)

    # Time labels for bottom row (formatted in math mode)
    time_labels = [
        r"$t=1\,\mathrm{s}$",
        r"$t=2\,\mathrm{s}$",
        r"$t=3\,\mathrm{s}$",
        r"$t=4\,\mathrm{s}$",
        r"$t=5\,\mathrm{s}$",
    ]

    # LEGEND (using lighter colors)
    color_ego      = "#66B3FF"  # light blue for Ego (Player 1)
    color_other_on = "#FF9999"  # light red for Other (mask active)
    color_other_off= "#99FF99"  # light green for Other (mask inactive)

    blue_ego    = Line2D([], [], color=color_ego, marker='o', markersize=8,
                         linewidth=2, label=r"$\text{Ego \ Player}$")
    red_other   = Line2D([], [], color=color_other_on, marker='o', markersize=8,
                         linewidth=2, label=r"$\text{Included \ Player(s)}$")
    green_other = Line2D([], [], color=color_other_off, marker='o', markersize=8,
                         linewidth=2, label=r"$\text{Excluded \ Player(s)}$")

    fig.legend(
        handles=[blue_ego, red_other, green_other],
        loc='upper center',
        bbox_to_anchor=(0.5, .95),  # Move legend closer to the plots
        ncol=3,
        fontsize=18
    )

    # MAIN PLOTTING LOOP
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
        
        if "Threshold" in method_label:
            method_label = clean_method_name_for_legend(method_label, "threshold")
        else:
            method_label = clean_method_name_for_legend(method_label, "no_threshold")
        # Wrap the method label in mathmode for bold formatting.
        method_label = r"$\text{" + method_label + r"}$"

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
                #start_idx = 0 # plot all previous steps
                start_idx = max(0, step - 9)
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
                if col < len(time_labels):
                    ax.annotate(
                        time_labels[col],
                        xy=(0.5, -0.15),
                        xycoords='axes fraction',
                        ha='center',
                        va='center',
                        fontsize=18
                    )
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            
            # Only leftmost column subplots show the y-axis label and tick labels.
            if col == 0:
                ax.annotate(
                    method_label,
                    xy=(-0.1, 0.5),
                    xycoords='axes fraction',
                    ha='center',
                    va='center',
                    rotation=90,
                    fontsize=18,
                   # fontweight='bold'
                )
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', left=False, right=False)

    plt.savefig(f"nn_traj_vis\\trajectories_grid_scenario_{scenario_id}.pdf", dpi=1000, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
