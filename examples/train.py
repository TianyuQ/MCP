import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import bson
from tqdm import tqdm
import glob
import json
import re
from julia.api import Julia
from julia import Main  # Now import Julia after initializing
jl = Julia(compiled_modules=False)

Main.eval('import Pkg; Pkg.activate("."); Pkg.instantiate()')  # Activate Julia environment

# Initialize PyJulia and import Julia functions
Main.include("/home/tq877/Tianyu/player_selection/MCP/examples/PlayerSelectionTraining.jl")  # Load the Julia script
run_solver = Main.PlayerSelectionTraining.run_solver  # Bind the Julia function to Python

# Hyperparameters
N = 4  # Number of players
T = 10  # Time steps
d = 4  # State dimension per player
input_size = N * T * d + N
batch_size = 1
epochs = 1
learning_rate = 0.001

# Define Neural Network Model
class MaskPredictor(nn.Module):
    def __init__(self):
        super(MaskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, N)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)  # Outputs values between 0 and 1

# Load dataset function
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

# Prepare input vector
def prepare_input(trajectories, ego_index):
    flat_traj = np.concatenate([trajectories[i].flatten() for i in range(1, N+1)])
    ego_onehot = np.zeros(N)
    ego_onehot[ego_index - 1] = 1.0  # Convert 1-based index to 0-based
    return np.concatenate((flat_traj, ego_onehot))

# Load dataset
print("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_dataset(directory)
print(f"Dataset loaded successfully. Total samples: {len(dataset)}")

# Initialize Model & Optimizer
print("Initializing model...")
model = MaskPredictor()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Training Loop
print("Starting training...")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    total_loss = 0.0
    progress = tqdm(dataset, desc=f"Epoch {epoch + 1} Training Progress")

    for trajectories, ego_index, initial_states, goals in progress:
        # Prepare input for the model
        input_vec = torch.tensor(prepare_input(trajectories, ego_index), dtype=torch.float32).unsqueeze(0)

        # Forward pass
        pred_masks = model(input_vec)
        bin_mask = (pred_masks >= 0.5).int().numpy().flatten()  # Convert to binary mask

        # Call Julia `run_solver`
        computed_traj = run_solver(bin_mask.tolist(), initial_states.tolist(), goals.tolist(), N, T, 1)

        # Convert computed trajectory to tensor
        computed_traj = torch.tensor(computed_traj, dtype=torch.float32).view(-1, 1)

        # Compute ground truth trajectory
        ground_truth_traj = torch.tensor(np.concatenate([trajectories[j].flatten() for j in range(1, N+1)]),
                                         dtype=torch.float32).view(-1, 1)

        # Compute loss
        traj_error = criterion(computed_traj, ground_truth_traj)
        mask_reg = torch.sum(pred_masks)
        loss = traj_error + 0.1 * mask_reg  # Regularization

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataset)}")

# Save trained model
print("\nSaving trained model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'input_size': input_size,
}, f"trained_model_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pth")
print("Model saved successfully!")
