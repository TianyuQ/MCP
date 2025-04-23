import json
import numpy as np
import matplotlib.pyplot as plt

# === Metric Analysis Functions ===

def trajectory_smoothness_analysis(trajectory):
    positions = np.array(trajectory[1])
    smoothness = 0
    for i in range(1, len(positions) - 1):
        vec1 = positions[i] - positions[i - 1]
        vec2 = positions[i + 1] - positions[i]
        dir1 = vec1[:2] / np.linalg.norm(vec1[:2])
        dir2 = vec2[:2] / np.linalg.norm(vec2[:2])
        diff = dir2 - dir1
        smoothness += np.linalg.norm(diff)
    return round(smoothness / len(trajectory[1]), 3)

def trajectory_length_analysis(trajectory):
    length = 0
    for step in range(len(trajectory[1]) - 1):
        length += np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(trajectory[1][step + 1][:2]))
    return round(length, 3)

def safety_analysis(trajectory):
    min_distance = np.inf
    for player_id in range(2, 11):
        for step in range(len(trajectory[1])):
            distance = np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(trajectory[player_id][step][:2]))
            if distance < min_distance:
                min_distance = distance
    return round(min_distance, 3)

def mask_sum_analysis(masks):
    return np.sum(masks) / len(masks)

def rate_analysis(rates):
    return np.sum(rates) / len(rates)

# === Setup ===

modes_with_params = {
    "All": [1],
    "Distance Threshold": [1.5, 2.0, 2.5],
    "Nearest Neighbor": [3, 5, 7],
    "Cost Evolution": [3, 5, 7],
    "Barrier Function": [3, 5, 7],
    "Neural Network Threshold": [0.1, 0.3, 0.5],
    "Neural Network Partial Threshold": [0.1, 0.3, 0.5],
    "Neural Network Rank": [3, 5, 7],
    # "Neural Network Partial Rank": [3, 5, 7],
}

options = [
            "threshold", 
            # "ranking3",
            "ranking5",
            # "ranking7",
            ]
# option = "threshold"
# option = "ranking2"
# opttion = "ranking3"
for option in options:
    print(f"=== {option} ===")
    if option == "threshold":
        selected_methods = {
            # "Neural Network Threshold [0.1]",
            # "Neural Network Threshold [0.3]",
            "Neural Network Threshold [0.5]",
            # "Neural Network Partial Threshold [0.1]",
            # "Neural Network Partial Threshold [0.3]",
            "Neural Network Partial Threshold [0.5]",
            # "Distance Threshold [1.5]",
            # "Distance Threshold [2.0]",
            "Distance Threshold [2.5]",
            "All [1]"
        }
    elif option == "ranking3":
        selected_methods = {
            "Neural Network Rank [3]",
            "Neural Network Partial Rank [3]",
            "Nearest Neighbor [3]",
            "Jacobian [3]",
            "Hessian [3]",
            "Cost Evolution [3]",
            "Barrier Function [3]",
            "Control Barrier Function [3]",
            "All [1]"
        }
    elif option == "ranking5":
        selected_methods = {
            "Neural Network Rank [5]",
            "Neural Network Partial Rank [5]",
            "Nearest Neighbor [5]",
            "Cost Evolution [5]",
            "Barrier Function [5]",
            "All [1]"
        }
    elif option == "ranking7":
        selected_methods = {
            "Neural Network Rank [7]",
            "Neural Network Partial Rank [7]",
            "Nearest Neighbor [7]",
            "Jacobian [7]",
            "Hessian [7]",
            "Cost Evolution [7]",
            "Barrier Function [7]",
            "Control Barrier Function [7]",
            "All [1]"
        }
    else:
        print("Invalid option. Please choose 'threshold' or 'rank'.")
        exit()

    result_dir = "/home/tq877/Tianyu/player_selection/MCP/data_ped"
    metrics_dict = {}

    # === Collect Raw Values ===

    for mode in modes_with_params:
        for param in modes_with_params[mode]:
            traj_smoothness_list = []
            traj_length_list = []
            safety_list = []
            mask_sum_list = []
            rate_list = []

            for scenario_id in range(3,7):
                result_file = f"{result_dir}/trajectories_[{scenario_id}]_[{mode}]_[{param}].json"
                all_file = f"{result_dir}/trajectories_[{scenario_id}]_[All]_[1].json"

                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    with open(all_file, 'r') as f:
                        all_data = json.load(f)

                    trajectories = {pid: data[f'Player {pid} Trajectory'] for pid in range(1, 11)}
                    masks = data['Player 1 Mask']
                    rates = [1 / (np.sum(mask) ** 3) for mask in masks]

                    traj_smoothness_list.append(trajectory_smoothness_analysis(trajectories))
                    traj_length_list.append(trajectory_length_analysis(trajectories))
                    safety_list.append(safety_analysis(trajectories))
                    mask_sum_list.append(mask_sum_analysis(masks))
                    rate_list.append(rate_analysis(rates))

                except FileNotFoundError:
                    continue

            key = f"{mode} [{param}]"
            if len(traj_smoothness_list) > 0:
                metrics_dict[key] = {
                    "Smoothness": np.mean(traj_smoothness_list),
                    "Length": np.mean(traj_length_list),
                    "Safety": np.mean(safety_list),
                    "Mask Sum": np.mean(mask_sum_list),
                    "Rate": np.mean(rate_list)
                }
                print(f"{key}: {metrics_dict[key]}")

    # === Radar Chart with Custom Raw Value Mapping ===

    metric_names = ["Smoothness", "Length", "Safety", "Mask Sum", "Rate"]
    invert_metrics = {"Smoothness", "Length", "Mask Sum"}

    # Use min, mean, max as tick references
    metric_ticks = {}
    for metric in metric_names:
        values = [metrics[metric] for metrics in metrics_dict.values()]
        metric_ticks[metric] = {
            "min": min(values),
            "mean": np.mean(values),
            "max": max(values)
        }
        print(f"{metric}: {metric_ticks[metric]}")

    metric_ticks["Smoothness"]["min"] = 0.001
    metric_ticks["Smoothness"]["max"] = 0.02
    metric_ticks["Length"]["min"] = 11
    metric_ticks["Length"]["max"] = 17
    metric_ticks["Safety"]["min"] = 0.3
    metric_ticks["Safety"]["max"] = 1.5
    metric_ticks["Mask Sum"]["min"] = 1
    metric_ticks["Mask Sum"]["max"] = 10
    metric_ticks["Rate"]["min"] = 0
    metric_ticks["Rate"]["max"] = 1

    # Map raw value to radius: r = 0 for min, r = 0.5 for mean, r = 1 for max
    def map_raw_to_radius(value, metric):
        ticks = metric_ticks[metric]
        if metric in invert_metrics:
            if value <= ticks["mean"]:
                ratio = (value - ticks["mean"]) / (ticks["min"] - ticks["mean"] + 1e-6)
                r = 0.7 + 0.3 * ratio
            else:
                ratio = (value - ticks["max"]) / (ticks["mean"] - ticks["max"] + 1e-6)
                r = 0.7 * ratio
        else:
            if value <= ticks["mean"]:
                ratio = (value - ticks["min"]) / (ticks["mean"] - ticks["min"] + 1e-6)
                r = 0.7 * ratio
            else:
                ratio = (value - ticks["mean"]) / (ticks["max"] - ticks["mean"] + 1e-6)
                r = 0.7 + 0.3 * ratio    
        return r

    # === Plotting ===

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
            if "Neural Network Partial" in method and "[" in method:
                return "PSN-Partial " + method[method.index('['):]  # Keep the parameter part
            elif "Neural Network" in method and "[" in method:
                return "PSN-Full " + method[method.index('['):]  # Keep the parameter part
            elif "Distance Threshold" in method and "[" in method:
                return "Distance " + method[method.index('['):]  # Keep the parameter part

        return method  # Return the original method name if no change is needed

    labels = metric_names
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], labels=labels, fontsize=18)
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    ax.spines['polar'].set_visible(False)

    ax.plot(angles, [0.7]*len(labels) + [0.7], linestyle='--', color='gray', linewidth=2, label='mean')

    # Plot selected methods
    for method, metric_vals in metrics_dict.items():
        if method not in selected_methods:
            continue
        data = [map_raw_to_radius(metric_vals[metric], metric) for metric in labels]
        data += data[:1]
        
        # Clean the method name for the legend
        legend_label = clean_method_name_for_legend(method, option)
        # legend_label = method

        # Plot the data with the cleaned label for the legend
        if method == "All [1]":
            ax.plot(angles, data, linewidth=2.5, color='black', label='All')
            # ax.fill(angles, data, color='black', alpha=0.08)
        else:
            ax.plot(angles, data, linewidth=3, label=legend_label)
            ax.fill(angles, data, alpha=0.07)

    # Final formatting
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1), fontsize=18)
    # plt.title(f"Radar chart for {option}-based methods", size=14, y=1.05)
    plt.tight_layout()
    plt.savefig(f"radar_plot_ped_{option}.pdf", dpi=1000)
    print(f"Radar plot saved as 'radar_plot_ped_{option}.pdf'.")
    # plt.show()
