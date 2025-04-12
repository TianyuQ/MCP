import json
import numpy as np


def trajectory_similarity_analysis(trajectory, ref_trajectory):
    similarity = 0
    for step in range(len(trajectory[1])):
        similarity += np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(ref_trajectory[1][step][:2]))
    return round(similarity / len(trajectory[1]), 3)

def trajectory_smoothness_analysis(trajectory):
    positions = np.array(trajectory[1])  # assuming trajectory[1] contains list of [x, y, ...] points
    smoothness = 0
    for i in range(1, len(positions) - 1):
        vec1 = positions[i] - positions[i - 1]
        vec2 = positions[i + 1] - positions[i]
        dir1 = vec1[:2] / np.linalg.norm(vec1[:2])  # unit direction vector
        dir2 = vec2[:2] / np.linalg.norm(vec2[:2])
        diff = dir2 - dir1
        smoothness += np.linalg.norm(diff)  # L2 norm of direction change
    return round(smoothness / len(trajectory[1]), 3)

def trajectory_length_analysis(trajectory):
    length = 0
    for step in range(len(trajectory[1]) - 1):
        length += np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(trajectory[1][step + 1][:2]))
    return round(length, 3)

def safety_analysis(trajectory):
    min_distance = np.inf
    for player_id in range(2, 5):
        for step in range(len(trajectory[1])):
            distance = np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(trajectory[player_id][step][:2]))
            if distance < min_distance:
                min_distance = distance
    return round(min_distance, 3)

def mask_sum_analysis(masks):
    # return sum(mask)
    return np.sum(masks) / len(masks)

def quantile_analysis(array):
    q1 = np.quantile(array, 0.25)
    q2 = np.quantile(array, 0.5)
    q3 = np.quantile(array, 0.75)
    return q1, q2, q3

def mean_analysis(array):
    mean = np.mean(array)
    return mean

modes = [
    "Nearest Neighbor",
    "Distance Threshold", 
    "Jacobian", 
    "Hessian", 
    "Cost Evolution",
    "Barrier Function",
    "Control Barrier Function",
    "Neural Network Threshold",
    "Neural Network Rank",
    "All"
]

result_dir = "/home/tq877/Tianyu/player_selection/MCP/data_closer_test_cooperative"

traj_similarity_list = []
traj_smoothness_list = []
traj_length_list = []
safety_list = []
mask_sum_list = []

# mode = "Distance Threshold"
# mode = "Nearest Neighbor"
mode = "Neural Network Rank"
# mode = "Neural Network Threshold"
# mode = "Jacobian"
# mode = "Hessian"
# mode = "Cost Evolution"
# mode = "All"

mode_parameter = 3

for scenario_id in range(160, 192):
    result_file = f"{result_dir}/receding_horizon_trajectories_[{scenario_id}]_[{mode}]_[{mode_parameter}].json"
    all_file = f"{result_dir}/receding_horizon_trajectories_[{scenario_id}]_[All]_[1].json"
    with open(result_file, 'r') as f:
        data = json.load(f)
    with open(all_file, 'r') as f:
        all_data = json.load(f)
    
    # Extract trajectories and goals
    trajectories = {}
    masks = {}
    ref_trajectories = {}
    goals = {}

    for player_id in range(1, 5):
        trajectories[player_id] = data[f'Player {player_id} Trajectory']
        goals[player_id] = data[f'Player {player_id} Goal']
        ref_trajectories[player_id] = all_data[f'Player {player_id} Trajectory']

    masks = data[f'Player 1 Mask']
    traj_similarity_list.append(trajectory_similarity_analysis(trajectories, ref_trajectories))
    traj_smoothness_list.append(trajectory_smoothness_analysis(trajectories))
    traj_length_list.append(trajectory_length_analysis(trajectories))
    safety_list.append(safety_analysis(trajectories))
    mask_sum_list.append(mask_sum_analysis(masks))

print(f"Mode: {mode}, Parameter: {mode_parameter}")
print("Trajectory similarity analysis results:", quantile_analysis(traj_similarity_list), "Mean:", mean_analysis(traj_similarity_list))
print("Trajectory smoothness analysis results:", quantile_analysis(traj_smoothness_list), "Mean:", mean_analysis(traj_smoothness_list))
print("Trajectory length analysis results:", quantile_analysis(traj_length_list), "Mean:", mean_analysis(traj_length_list))
print("Safety analysis results:", quantile_analysis(safety_list), "Mean:", mean_analysis(safety_list))
print("Mask sum analysis results:", quantile_analysis(mask_sum_list), "Mean:", mean_analysis(mask_sum_list))