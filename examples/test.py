import torch
import torch.nn as nn
import numpy as np
import json
import re
import glob
from julia.api import Julia

# Initialize PyJulia
jl = Julia(compiled_modules=False)
from julia import Main  # Import Julia after initializing
Main.eval('import Pkg; Pkg.activate("."); Pkg.instantiate()')  # Activate Julia environment

# Load Julia functions
Main.include("/home/tq877/Tianyu/player_selection/MCP/examples/PlayerSelectionTraining.jl")
run_solver = Main.PlayerSelectionTraining.run_solver  # Bind Julia function
load_all_json_data = Main.PlayerSelectionTraining.load_all_json_data

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
N = 4  # Number of players
T = 10  # Time steps
d = 4  # State dimension per player
input_size = N * T * d + N  # Model input size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
def load_dataset(directory):
    """
    Load the dataset from JSON files in the given directory.

    Each file contains:
    - Trajectories of players
    - Initial states
    - Goals
    - Ego agent indices (all possible ones)

    Returns:
        dataset: List of tuples (trajectories, ego_index, initial_states, goals, ground_truth_mask)
    """
    json_files = glob.glob(f"{directory}/simulation_results_0[*.json")  # Adjusted format
    dataset = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        # Extract player trajectories
        trajectories = {i: np.array(data[f"Player {i} Trajectory"]) for i in range(1, N+1)}

        # Extract initial states and goals
        initial_states = np.concatenate([np.array(data[f"Player {i} Initial State"]) for i in range(1, N+1)])
        goals = np.concatenate([np.array(data[f"Player {i} Goal"]) for i in range(1, N+1)])

        # Extract binary mask from filename using regex
        match = re.search(r"simulation_results_0\[(.*?)\]", file)
        if match:
            mask_str = match.group(1)  # Extract the contents inside the brackets
            mask_values = list(map(int, mask_str.split(",")))  # Convert to a list of integers

            # Traverse all possible ego indices where value == 1
            active_players = [i for i in range(N) if mask_values[i] == 1]
            if not active_players:
                raise ValueError(f"No active players found in mask: {mask_values} in file {file}")

            # Append a separate dataset entry for each possible ego index
            for ego_index in active_players:
                dataset.append((trajectories, ego_index, initial_states, goals, np.array(mask_values)))
        else:
            raise ValueError(f"Could not extract mask from filename: {file}")

    return dataset

print("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_dataset(directory)  # Use the same data loading method as in `train.py`
print(f"Dataset loaded successfully. Total samples: {len(dataset)}")

# -----------------------------------------------------------------------------
# Define Model (Matches Training)
# -----------------------------------------------------------------------------
class MaskPredictionModel(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=4):
        super(MaskPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid activation for binary mask
        return x

# -----------------------------------------------------------------------------
# Load Trained Model
# -----------------------------------------------------------------------------
print("Loading trained model...")
checkpoint = torch.load("trained_model_batch=2_epochs=2_lr=0.001.pth", map_location=device)

# Ensure model architecture matches
input_size = checkpoint["input_size"]
model = MaskPredictionModel(input_size=input_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])  # ✅ Load only model parameters
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")

# -----------------------------------------------------------------------------
# Prepare Input Function (Matches Training)
# -----------------------------------------------------------------------------
def prepare_input(trajectories, ego_index):
    """
    Prepare input for the model by flattening all player trajectories and adding a one-hot encoding for ego index.
    """
    flat_traj = np.concatenate([trajectories[i].flatten() for i in range(1, N+1)])
    ego_onehot = np.zeros(N)
    ego_onehot[ego_index] = 1.0  # Convert 1-based index to 0-based
    return np.concatenate((flat_traj, ego_onehot))

# -----------------------------------------------------------------------------
# Testing Function (Traverse All Trials)
# -----------------------------------------------------------------------------
def test_model(dataset, model):
    """
    Traverse all trials in the dataset and evaluate predictions for each trial.
    """
    total_loss = 0.0
    all_results = []

    for sample in dataset:
        trajectories, ego_index, initial_states, goals, ground_truth_mask = sample

        # Prepare input vector
        input_vec = prepare_input(trajectories, ego_index)
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).to(device)

        # Predict mask
        with torch.no_grad():
            predicted_mask = model(input_tensor).cpu().numpy()

        # Convert predicted mask to binary
        bin_mask = (predicted_mask >= 0.5).astype(int).flatten()

        # Run solver with predicted mask
        computed_traj = run_solver(bin_mask.tolist(), initial_states.tolist(), goals.tolist(), N, T, 1)

        # Convert computed trajectories to NumPy array
        computed_traj = np.array(computed_traj)

        # Compute MSE loss
        ground_truth_traj = np.concatenate([trajectories[i].flatten() for i in range(1, N+1)])
        mse_loss = np.mean((computed_traj - ground_truth_traj) ** 2)
        total_loss += mse_loss

        # Store results for analysis
        all_results.append({
            "ego_index": ego_index + 1,
            "ground_truth_mask": ground_truth_mask.tolist(),
            "predicted_mask": predicted_mask.tolist(),
            "binary_mask": bin_mask.tolist(),
            "ground_truth_traj": ground_truth_traj.tolist(),
            "computed_traj": computed_traj.tolist(),
            "mse_loss": mse_loss
        })

        # Print sample results
        print("\n===== Trial Results =====")
        print(f"Ego-Agent Index: {ego_index + 1}")
        print(f"Ground Truth Mask: {ground_truth_mask}")
        print(f"Predicted Mask: {predicted_mask:.4f}")
        print(f"Binary Mask (Thresholded at 0.5): {bin_mask}")
        print(f"MSE Loss: {mse_loss:.4f}")

    # Compute final average loss
    final_loss = total_loss / len(dataset)
    return final_loss, all_results

# -----------------------------------------------------------------------------
# Run Testing
# -----------------------------------------------------------------------------
test_loss, results = test_model(dataset, model)

print(f"\nTesting complete. Final MSE loss: {test_loss:.6f}")

# Optionally save results for further analysis
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Test results saved to 'test_results.json'")
