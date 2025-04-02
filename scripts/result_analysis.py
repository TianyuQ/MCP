import json
from cv2 import norm
import numpy as np


def trajectory_similarity_analysis(trajectory, ref_trajectory):
    similarity = 0
    for step in range(len(trajectory[1])):
        similarity += np.linalg.norm(np.array(trajectory[1][step][:2]) - np.array(ref_trajectory[1][step][:2]))
    return round(similarity / len(trajectory[1]), 3)
    

# def trajectory_smoothness_analysis(trajectory):
#     angle = 0
#     for step in range(len(trajectory[1]) - 1):
#         angle += np.arccos(np.dot(trajectory[1][step][:2], trajectory[1][step + 1][:2]) / (np.linalg.norm(trajectory[1][step][:2]) * np.linalg.norm(trajectory[1][step + 1][:2])))
#     return round(angle / (len(trajectory[1]) - 1), 3)

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


methods = [
    "Nearest Neighbor",
    "Distance Threshold", 
    "Jacobian", 
    "Hessian", 
    "Cost Evolution",
    "Barrier Function",
    "Control Barrier Function",
    "Neural Network",
    "All"
]

result_dir = "/home/tq877/Tianyu/player_selection/MCP/data_closer_test"

traj_similarity_list = []
traj_smoothness_list = []
safety_list = []
mask_sum_list = []

# method = "Distance Threshold"
# method = "Neural Network"
method = "All"

for scenario_id in range(160, 192):
    result_file = f"{result_dir}/receding_horizon_trajectories_[{scenario_id}]_{method}.json"
    all_file = f"{result_dir}/receding_horizon_trajectories_[{scenario_id}]_All.json"
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
    safety_list.append(safety_analysis(trajectories))
    mask_sum_list.append(mask_sum_analysis(masks))

print("Method:", method)
print("Trajectory similarity analysis results:", quantile_analysis(traj_similarity_list))
print("Trajectory smoothness analysis results:", quantile_analysis(traj_smoothness_list))
print("Safety analysis results:", quantile_analysis(safety_list))
print("Mask sum analysis results:", quantile_analysis(mask_sum_list))



        # traj_similarity_list.append(trajectory_similarity_analysis(trajectories[1], ref_trajectories[1]))