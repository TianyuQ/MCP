import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re

plt.rcParams.update({
    'text.usetex': False,                     # turn off external TeX
    'mathtext.fontset': 'cm',                 # use Computer Modern math
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'axes.labelsize': 18,                   
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 17,
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
    """
    Simplify the method name for the legend.
    If option == "threshold", also extract and show the 'th=…' value.
    """
    # try to pull out the threshold value (e.g. '0.5') if present
    thr_match = re.search(r'th=([0-9.]+)', method)
    thr = thr_match.group(1) if thr_match else None

    if option == "threshold" and thr is not None:
        # threshold‐style labels
        if "Neural Network Partial" in method:
            base = "PSN-Partial"
        elif "Neural Network Threshold" in method:
            base = "PSN-Full"
        elif "Distance Threshold" in method:
            base = "Distance"
        else:
            return method  # unknown pattern, fall back

        return f"{base} [{thr}]"

    # non-threshold or no match: keep original logic (or just return method)
    if option != "threshold":
        if "Neural Network Partial" in method:
            return "PSN-Partial"
        elif "Neural Network" in method:
            return "PSN-Full"
        elif "Distance Threshold" in method or "Nearest Neighbor" in method:
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

    # fallback
    return method

# ADJUSTABLE PARAMETERS. 
N=4
num_scenarios = 32
scenario_start_idx = 160

total = num_scenarios * 50

scenario_idx = np.arange(scenario_start_idx, scenario_start_idx + num_scenarios, 1)

methods = ['Neural Network Threshold th=0.1','Neural Network Threshold th=0.3','Neural Network Threshold th=0.5',\
           'Neural Network Partial Threshold th=0.1', 'Neural Network Partial Threshold th=0.3','Neural Network Partial Threshold th=0.5', \
            'Distance Threshold th=1.5', 'Distance Threshold th=2.0', 'Distance Threshold th=2.5'] # the syntax here is important!
method_names = [methods[i][:-7] for i in range(len(methods))]

method_sums = {}
for m, method in enumerate(method_names):
    sum_list = []

    for i in scenario_idx:
        if 'Distance Threshold' in methods[m]:
            threshold_value = int(float(methods[m].split('=')[-1]))
            with open(f"data_test_{N} _30\\receding_horizon_trajectories_[{i}]_[{method}]_[{threshold_value}].json", "r") as f:
                data = json.load(f)
        else:  
            with open(f"data_test_{N} _30\\receding_horizon_trajectories_[{i}]_[{method}]_[{methods[m][-3:]}].json", "r") as f:
                data = json.load(f)

        masks = np.array(get_trajectory(data, sim_steps="all")[2])
        summed_masks = masks.sum(axis=1)

        sum_list.extend(summed_masks.tolist())

    # ← after you've built the full list for this (N,method), do one np.unique
    unique_vals, counts = np.unique(sum_list, return_counts=True)
    scenario_sums = dict(zip(unique_vals.tolist(), counts.tolist()))

    method_sums[methods[m]] = scenario_sums

# Plotting

mask_values = np.arange(1, N+1)

# compute percentages
percent = {
    m: [method_sums[m].get(k,0)/total*100 for k in mask_values]
    for m in methods
}

color_map = {
    methods[0]: 'tab:blue',    # PSN‑Full-th-0.1
    methods[1]: 'tab:cyan',    # PSN‑Full-th-0.3
    methods[2]: 'tab:olive',   # PSN‑Full-th-0.5
    methods[3]: 'tab:orange',  # PSN‑Partial-th-0.1
    methods[4]: 'tab:purple',  # PSN‑Partial-th-0.3
    methods[5]: 'tab:brown',   # PSN‑Partial-th-0.5
    methods[6]: 'tab:green',   # Distance-th-1.5
    methods[7]: 'tab:pink',    # Distance-th-2.0
    methods[8]: 'tab:red'      # Distance-th-2.5
}

# ——————————————
# 3) Plot
x = np.arange(len(mask_values))

n_methods = len(methods)
bar_width = 0.5/n_methods

fig, ax = plt.subplots(figsize=(10,5))

for i, m in enumerate(methods):
    # offset each bar so the cluster is centered at x
    offset = (i - (n_methods-1)/2) * bar_width
    ax.bar(
        x + offset,
        percent[m],
        bar_width,
        label=clean_method_name_for_legend(m, "threshold"),
        color=color_map[m]
    )

    # annotate each bar with its value
    # ax.bar_label(
    #     bars,
    #     fmt='%.1f%%',
    #     padding=2,
    #     fontsize=8
    # )

# ——————————————
# 4) Formatting
ax.set_xticks(x)
ax.set_xticklabels(mask_values)
ax.set_xlabel('Number of Players')
ax.set_ylabel('Percentage of all interactions (%)')
ax.set_ylim(0, max(max(v) for v in percent.values())*1.1)

# clean spines & add light grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

ax.legend(frameon=False, ncol=3)
# ax.set_title('Distribution of Mask‑Sum by Method')

plt.tight_layout()

# ——————————————
# 5) Export

plt.savefig(f'mask_sum_distribution_{N}.pdf', format='pdf', bbox_inches='tight', dpi=1000)


# histogram:
# PSN-Full-th-0.5
# PSN-Partial-th-0.5
# Distance-th-2/2.5

