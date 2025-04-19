import numpy as np
import matplotlib.pyplot as plt
import json
import os

plt.rcParams.update({
    'text.usetex': False,                     # turn off external TeX
    'mathtext.fontset': 'cm',                 # use Computer Modern math
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
    'figure.dpi': 300
})


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
        if "Neural Network Partial" in method and "[" in method:
            return "PSN-Partial"  # For Neural Network Partial methods, return 'PSN-Partial' for the legend
        elif "Neural Network" in method and "[" in method:
            return "PSN-Full"  # For Neural Network methods, return 'PSN' for the legend
        elif "Distance Threshold" in method and "[" in method:
            return "Distance"  # For Distance Threshold, return 'Distance' for the legend
        elif "Nearest Neighbor" in method and "[" in method:
            return "Distance"  # For Nearest Neighbor, return 'Distance' for the legend
        elif "Control Barrier Function" in method and "[" in method:
            return "CBF"  # For Control Barrier Function, return 'CBF' for the legend
        elif "Barrier Function" in method and "[" in method:
            return "BF"  # For Barrier Function, return 'BF' for the legend
        elif "Jacobian" in method and "[" in method:
            return "Jacobian"
        elif "Hessian" in method and "[" in method:
            return "Hessian"
        elif "Cost Evolution" in method and "[" in method:
            return "Cost Evolution"
    else:
        # For the threshold option, keep the parameter but simplify the method name
        if "Neural Network Threshold" in method and "[" in method:
            return "PSN" + method[method.index('['):]  # Keep the parameter part
        elif "Distance Threshold" in method and "[" in method:
            return "Distance" + method[method.index('['):]  # Keep the parameter part
        elif "Nearest Neighbor" in method and "[" in method:
            return "Distance" + method[method.index('['):]  # Keep the parameter part
        elif "Control Barrier Function" in method and "[" in method:
            return "CBF" + method[method.index('['):]  # Keep the parameter part
        elif "Barrier Function" in method and "[" in method:
            return "BF" + method[method.index('['):]  # Keep the parameter part
        elif "Jacobian" in method and "[" in method:
            return "Jacobian" + method[method.index('['):]  # Keep the parameter part
        elif "Hessian" in method and "[" in method:
            return "Hessian" + method[method.index('['):]  # Keep the parameter part
        elif "Cost Evolution" in method and "[" in method:
            return "Cost Evolution" + method[method.index('['):]  # Keep the parameter part

    return method  # Return the original method name if no change is needed

# ADJUSTABLE PARAMETERS. 
N=4
num_scenarios = 32
scenario_start_idx = 160

total = num_scenarios * 50

scenario_idx = np.arange(scenario_start_idx, scenario_start_idx + num_scenarios, 1)

methods = ['Neural Network Threshold', 'Neural Network Partial Threshold',  'Distance Threshold']
nnth = 0.5
dth = 2 

method_sums = {}
for method in methods:
    sum_list = []

    for i in scenario_idx:
        if "Neural Network" in method:
            with open(f"data_test_4 _30\\receding_horizon_trajectories_[{i}]_[{method}]_[{nnth}].json", "r") as f:
                data = json.load(f)
        else:
            with open(f"data_test_4 _30\\receding_horizon_trajectories_[{i}]_[{method}]_[{dth}].json", "r") as f:
                data = json.load(f)

        masks = np.array(get_trajectory(data, sim_steps="all")[2])
        summed_masks = masks.sum(axis=1)

        sum_list.extend(summed_masks.tolist())

    # ← after you've built the full list for this (N,method), do one np.unique
    unique_vals, counts = np.unique(sum_list, return_counts=True)
    scenario_sums = dict(zip(unique_vals.tolist(), counts.tolist()))

    method_sums[method] = scenario_sums


# Plotting

mask_values = np.arange(1, N+1)

# compute percentages
percent = {
    m: [method_sums[m].get(k,0)/total*100 for k in mask_values]
    for m in methods
}

color_map = {
    'Distance Threshold':            'tab:orange',  # matches Distance [2]
    'Neural Network Threshold':      'tab:brown',   # matches PSN‑Full [0.5]
    'Neural Network Partial Threshold': 'tab:olive' # matches PSN‑Partial [0.5]
}

# ——————————————
# 3) Plot
x = np.arange(len(mask_values))
width = 0.25

fig, ax = plt.subplots(figsize=(10,10))

for i, m in enumerate(methods):
    bars = ax.bar(
        x + (i-1)*width,
        percent[m],
        width,
        label=m,
        color=color_map[m],
    )
    # annotate each bar with its value
    ax.bar_label(
        bars,
        fmt='%.1f%%',
        padding=2,
        fontsize=8
    )

# ——————————————
# 4) Formatting
ax.set_xticks(x)
ax.set_xticklabels(mask_values)
ax.set_xlabel('Mask sum (number of players selected)')
ax.set_ylabel('Percentage of all masks (\%)')
ax.set_ylim(0, max(max(v) for v in percent.values())*1.1)

# clean spines & add light grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

ax.legend(title='Method', frameon=False)
ax.set_title('Distribution of Mask‑Sum by Method')

plt.tight_layout()

# ——————————————
# 5) Export
plt.savefig('mask_sum_distribution.pdf', format='pdf', bbox_inches='tight', dpi=1000)


# histogram:
# PSN-Full-th-0.5
# PSN-Partial-th-0.5
# Distance-th-2/2.5

