import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import glob
import numpy as np
import re
from tqdm import tqdm

# from julia.api import Julia
# from julia import Main  # Now import Julia after initializing
# jl = Julia(compiled_modules=False)

# Main.eval('import Pkg; Pkg.activate("."); Pkg.instantiate()')  # Activate Julia environment

# # Initialize PyJulia and import Julia functions
# Main.include("PlayerSelectionTraining.jl")  # Load the Julia script
# run_solver = Main.PlayerSelectionTraining.run_solver  # Bind the Julia function to Python
# game = Main.PlayerSelectionTraining.game
# parametric_game = Main.PlayerSelectionTraining.parametric_game

# Initialize Generative Model & Optimizer
print("Initializing generative model...")

# Hyperparameters
player_num = 4
horizon = 30
num_steps = 1
state_dim = 4
input_size = player_num * state_dim * horizon + player_num  # Should be 484
epochs = 3
batch_size = 4
learning_rate = 0.0001

dir_path = "/home/tq877/Tianyu/player_selection/MCP/data_test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskPredictor(nn.Module):
    def __init__(self, input_size, player_num):
        super(MaskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, player_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)  # Outputs values between 0 and 1

input_size = player_num * state_dim * horizon + player_num  # Example input size

model = MaskPredictor(input_size, player_num)
optimizer = optim.SGD(model.parameters(), lr=0.001)

print("Model initialized successfully!")

# # Define Loss Function
# def loss_fun(trajectories, pred_probs, computed_trajs):
#     traj_error = torch.mean((computed_trajs - trajectories) ** 2)  # Mean Squared Error
#     mask_loss = torch.sum(torch.tensor(pred_probs))  # Sum of probabilities
#     return 100 * traj_error + mask_loss  # Loss function

# Bernoulli Sampling Function (Non-Differentiable)
def sample_bernoulli(probabilities):
    return (torch.rand_like(probabilities) < probabilities).float()

class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset  # List of (trajectories, ego_index, initial_states, goals)
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)  # Shuffle indices at initialization

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        batch_indices = []  # Track original indices
        batch_initial_states = []
        batch_goals = []

        for _ in range(self.batch_size):
            if self.current_idx >= len(self.dataset):
                break
            
            idx = self.indices[self.current_idx]
            batch_indices.append(idx)
            
            trajectories, ego_index, initial_states, goals = self.dataset[idx]
            batch_initial_states.append(initial_states)
            batch_goals.append(goals)
            
            input_vec = self.prepare_input(trajectories, ego_index)
            batch_inputs.append(input_vec)

            ground_truth_traj = []
            # print("trajectories", trajectories[1])
            # input("Press Enter to continue...")
            for j in trajectories.keys():
                ground_truth_traj.extend(trajectories[j].flatten())
            batch_targets.append(ground_truth_traj)
            
            self.current_idx += 1
        batch_inputs = torch.stack(batch_inputs, dim=0)
        batch_targets = torch.stack([torch.tensor(target, dtype=torch.float32) for target in batch_targets], dim=0)
        
        return batch_inputs, batch_targets, batch_indices, batch_initial_states, batch_goals
    
    def prepare_input(self, trajectories, ego_index):
        player_num = len(trajectories)
        flat_traj = torch.cat([torch.tensor(trajectories[i], dtype=torch.float32).flatten() for i in range(1, player_num+1)])
        ego_onehot = torch.zeros(player_num, dtype=torch.float32)
        ego_onehot[ego_index - 1] = 1.0  # Convert 1-based index to 0-based
        return torch.cat((flat_traj, ego_onehot))

# Load Dataset
def load_dataset(directory):
    json_files = glob.glob(f"{directory}/simulation_results_0[*.json")
    # json_files = glob.glob(f"{directory}/simulation_results_0[0,1,1,1].json")
    dataset = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        N = len([key for key in data.keys() if key.startswith("Player ") and "Trajectory" in key])
        trajectories = {i: np.array(data[f"Player {i} Trajectory"]) for i in range(1, N+1)}
        initial_states = np.concatenate([np.array(data[f"Player {i} Initial State"]) for i in range(1, N+1)])
        goals = np.concatenate([np.array(data[f"Player {i} Goal"]) for i in range(1, N+1)])

        match = re.search(r"simulation_results_0\[(.*?)\]", file)
        if match:
            mask_values = list(map(int, match.group(1).split(",")))
            active_players = [i for i in range(N) if mask_values[i] == 1]
            if not active_players:
                raise ValueError(f"No active players found in mask: {mask_values} in file {file}")
            for ego_index in active_players:
                dataset.append((trajectories, ego_index, initial_states, goals))
        else:
            raise ValueError(f"Could not extract mask from filename: {file}")

    return dataset

# Initialize Model
class MaskPredictor(nn.Module):
    def __init__(self, input_size, player_num):
        super(MaskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, player_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)  # Outputs values between 0 and 1

# Load dataset
dataset = load_dataset(dir_path)
dataloader = DataLoader(dataset, batch_size)

# Initialize model and optimizer
model = MaskPredictor(input_size, player_num).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Loss Function
def loss_fun(batch_targets, pred_probs, computed_trajs):
    traj_error = torch.mean((computed_trajs - batch_targets) ** 2)  # Mean Squared Error
    mask_loss = torch.sum(pred_probs)  # Sum of probabilities
    # return 100 * traj_error + mask_loss  # Loss function
    return 100 * traj_error  # Loss function

# Training Loop
print("Starting training...")
training_losses = {}

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_inputs, batch_targets, batch_indices, batch_initial_states, batch_goals in progress:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        pred_probs = model(batch_inputs)
        pred_probs_cpu = pred_probs.cpu().detach().numpy()
        print("pred_probs", pred_probs_cpu.round(3))

#         batch_computed_trajs = [
#             run_solver(
#                 game, parametric_game, pred_probs_cpu[i], batch_initial_states[i], batch_goals[i],
#                 player_num, horizon, num_steps
#             ) for i in range(batch_size)
#         ]
#         # print("batch_computed_trajs", batch_computed_trajs)
#         # print("batch_computed_trajs", len(batch_computed_trajs))
#         batch_computed_trajs = torch.tensor(batch_computed_trajs, dtype=torch.float32).view(-1).to(device)

#         # computed_trajs = torch.stack(batch_computed_trajs, dim=1)
#         loss = loss_fun(batch_targets, pred_probs, batch_computed_trajs)
#         print("loss", loss.cpu().detach().numpy())
        
#         # Compute Gradient Manually
#         grads = torch.zeros(batch_size, player_num).to(device)
        
#         for k in range(batch_size):
#             for j in range(player_num):
#                 pred_probs_cpu[k][j] += 1e-2  
#                 batch_computed_trajs_perturb = [
#                     run_solver(game, parametric_game, pred_probs_cpu[i], batch_initial_states[i], batch_goals[i],
#                                player_num, horizon, num_steps)
#                     for i in range(batch_size)
#                 ]
#                 batch_computed_trajs_perturb = torch.tensor(batch_computed_trajs_perturb, dtype=torch.float32).view(-1).to(device)
#                 loss_plus = loss_fun(batch_targets, torch.tensor(pred_probs_cpu, dtype=torch.float32).to(device), batch_computed_trajs_perturb)
#                 pred_probs_cpu[k][j] -= 2e-2
#                 batch_computed_trajs_perturb = [
#                     run_solver(game, parametric_game, pred_probs_cpu[i], batch_initial_states[i], batch_goals[i],
#                                player_num, horizon, num_steps)
#                     for i in range(batch_size)
#                 ]
#                 batch_computed_trajs_perturb = torch.tensor(batch_computed_trajs_perturb, dtype=torch.float32).view(-1).to(device)
#                 loss_minus = loss_fun(batch_targets, torch.tensor(pred_probs_cpu, dtype=torch.float32).to(device), batch_computed_trajs_perturb)
#                 grads[k, j] = (loss_plus - loss_minus) / 2e-2
#                 pred_probs_cpu[k][j] += 1e-2
#                 # # print("loss_perturb", loss_perturb)
#                 # grads[k, j] = (loss_perturb - loss) / 1e-3
#                 # pred_probs_cpu[k][j] += 1e-3
#         print("grads", grads)
#         pred_probs.backward(grads)
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataset)
#     training_losses[str(epoch)] = round(avg_loss, 6)
#     print(f"Epoch {epoch}, Average Loss: {training_losses[str(epoch)]}")

# # Save Trained Model
# torch.save(model.state_dict(), f"trained_model_bs_{batch_size}_ep_{epochs}_hor_{horizon}.pth")
# with open(f"training_losses_bs_{batch_size}_ep_{epochs}_hor_{horizon}.json", "w") as f:
#     json.dump(training_losses, f, indent=4)
# print("Training complete. Model and losses saved.")