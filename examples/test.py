import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import re
import glob
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main  # Import Julia after initializing
Main.eval('import Pkg; Pkg.activate("."); Pkg.instantiate()')  # Activate Julia environment

# Load Julia functions
Main.include("/home/tq877/Tianyu/player_selection/MCP/examples/PlayerSelectionTraining.jl")
run_solver = Main.PlayerSelectionTraining.run_solver  # Bind Julia function
load_all_json_data = Main.PlayerSelectionTraining.load_all_json_data

# Hyperparameters
N = 4  # Number of players
T = 10  # Time steps
d = 4  # State dimension per player
input_size = N * T * d + N
batch_size = 1
epochs = 1
learning_rate = 0.001

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
        dataset: List of tuples (trajectories, ego_index, initial_states, goals)
    """
    json_files = glob.glob(f"{directory}/simulation_results_2[*.json")  # Adjusted to match format
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
        match = re.search(r"simulation_results_2\[(.*?)\]", file)
        if match:
            mask_str = match.group(1)  # Extract the contents inside the brackets
            mask_values = list(map(int, mask_str.split(",")))  # Convert to a list of integers

            # Traverse all possible ego indices where value == 1
            active_players = [i for i in range(N) if mask_values[i] == 1]
            if not active_players:
                raise ValueError(f"No active players found in mask: {mask_values} in file {file}")

            # Append a separate dataset entry for each possible ego index
            for ego_index in active_players:
                dataset.append((trajectories, ego_index, initial_states, goals))
        else:
            raise ValueError(f"Could not extract mask from filename: {file}")

    return dataset

print("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_dataset(directory)  # Use the same data loading method as in `train.py`
print("Dataset loaded successfully. Total samples:", len(dataset))

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("trained_model_batch=2_epochs=2_lr=0.001.pth", map_location=device)

# Ensure model architecture matches
input_size = checkpoint["input_size"]
model = MaskPredictionModel(input_size=input_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])  # ✅ Load only model parameters
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")

# -----------------------------------------------------------------------------
# Define DataLoader (Same as Training)
# -----------------------------------------------------------------------------
class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))  # Keep the dataset order for testing

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            batch_inputs = []
            batch_targets = []

            for sample in batch:
                trajectories, ego_index, initial_states, goals = sample

                # Prepare input (Same as in training)
                input_vec = prepare_input(trajectories, ego_index)
                batch_inputs.append(input_vec)

                # Prepare ground truth (Flatten ground truth trajectories)
                ground_truth_traj = np.concatenate([trajectories[i].flatten() for i in range(1, N+1)])
                
                batch_targets.append(ground_truth_traj)

            yield torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(device), \
                  torch.tensor(np.array(batch_targets), dtype=torch.float32).to(device)

# -----------------------------------------------------------------------------
# Prepare Input Function (Matches Training)
# -----------------------------------------------------------------------------
def prepare_input(trajectories, ego_index):
    flat_traj = np.concatenate([trajectories[i].flatten() for i in range(1, N+1)])
    ego_onehot = np.zeros(N)
    ego_onehot[ego_index - 1] = 1.0  # Convert 1-based index to 0-based
    return np.concatenate((flat_traj, ego_onehot))

# -----------------------------------------------------------------------------
# Testing Function
# -----------------------------------------------------------------------------
def test_model(dataloader, batch_size, model):
    total_loss = 0.0
    num_samples = 0
    for batch_inputs, batch_targets in dataloader:
        print(len(batch_inputs))
        # print(batch_inputs)

        # Predict mask
        with torch.no_grad():
            predicted_masks = model(batch_inputs).cpu().numpy()

        # print(predicted_masks)
        # Convert predicted masks to binary

        batch_computed_trajs = []
        for i in range(batch_size):
            # Extract the correct dataset sample
            sample_idx = num_samples + i
            _, _, initial_states, goals = dataset[sample_idx]

            # Run solver with predicted mask
            bin_masks = (predicted_masks[i] >= 0.5).astype(int).flatten()
            # print(bin_masks.tolist())
            computed_traj = run_solver(bin_masks.tolist(), initial_states.tolist(), goals.tolist(), N, T, 1)
            
            batch_computed_trajs.append(computed_traj)

        # Convert computed trajectories to a NumPy array
        computed_trajs = np.array(batch_computed_trajs)

        # Compute MSE loss
        ground_truth_trajs = batch_targets.cpu().numpy()
        mse_loss = np.mean((computed_trajs - ground_truth_trajs) ** 2)
        total_loss += mse_loss * batch_size
        num_samples += batch_size

        # Print results for first batch
        print("\n===== Sample Results =====")
        print(f"Predicted Masks:\n{predicted_masks}")
        print(f"Binary Masks:\n{bin_masks}")
        # print(f"Ground Truth Masks:\n{batch_targets}")
        # print(f"Ground Truth Trajectories:\n{ground_truth_trajs}")
        # print(f"Computed Trajectories:\n{computed_trajs}")
        print(f"Batch MSE Loss: {mse_loss:.4f}")

    # Compute final average loss
    final_loss = total_loss / num_samples
    return final_loss

# -----------------------------------------------------------------------------
# Run Testing
# -----------------------------------------------------------------------------
test_dataloader = DataLoader(dataset, batch_size)  # Use batch size = 1 for evaluation
test_loss = test_model(test_dataloader, batch_size, model)

print(f"\nTesting complete. Final MSE loss: {test_loss:.6f}")
