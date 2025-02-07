# In this script, we will train the very basic model with goal and game structure all known.
# The network only needs to infer the mask.

###############################################################################
# Import Required Packages
###############################################################################
using Flux
using Flux.Losses: mse
using Statistics
using Random
using LinearAlgebra

###############################################################################
# Problem and Data Dimensions
###############################################################################
const N = 4      # total number of players
const T = 10     # number of time steps in the past trajectory
const d = 4      # state dimension for each player

# The network input is a concatenation of:
#   1. The flattened trajectories for all players (N x T x d values), and
#   2. The one-hot encoding of the ego-agent index (N values)
const input_size = N * T * d + N

###############################################################################
# Neural Network Model
###############################################################################
# A simple multilayer perceptron (MLP) that outputs an N-dimensional mask vector.
# A sigmoid activation is used to force the outputs to lie in [0, 1].
model = Chain(
    Dense(input_size, 64, relu),
    Dense(64, 32, relu),
    Dense(32, N),
    σ  # sigmoid activation for each element in the mask
)

###############################################################################
# Input Preparation Function
###############################################################################
"""
    prepare_input(trajectories, ego_index)

- `trajectories`: a Float32 array of size (N, T, d) representing the past trajectories of all players.
- `ego_index`: an integer (1:N) indicating which player is the ego-agent.

Returns a vector that concatenates the flattened trajectories and the one-hot encoded ego-agent.
"""
function prepare_input(trajectories::Array{Float32,3}, ego_index::Int)
    # Flatten the trajectory data (all players, timesteps, features)
    flat_traj = vec(trajectories)
    # Create one-hot encoding for the ego-agent
    ego_onehot = zeros(Float32, N)
    ego_onehot[ego_index] = 1.0
    return vcat(flat_traj, ego_onehot)
end

###############################################################################
# Dummy Differential Game Solver Function
###############################################################################
"""
    differential_game_solver(trajectories, mask)

This is a placeholder for your differential game solver. In this dummy example,
the computed trajectory is given as the weighted sum (using the mask values)
of the trajectories of each player.

- `trajectories`: a Float32 array of size (N, T, d)
- `mask`: an N-dimensional vector (with entries in [0, 1])

Returns a computed trajectory of size (T, d).
"""
function differential_game_solver(trajectories::Array{Float32,3}, mask::AbstractVector)
    T = size(trajectories, 2)
    d = size(trajectories, 3)
    computed_traj = zeros(Float32, T, d)
    for i in 1:size(trajectories, 1)
       computed_traj .+= mask[i] .* trajectories[i, :, :]
    end
    return computed_traj
end

###############################################################################
# Loss Function
###############################################################################
# Hyperparameter for regularization strength on the mask
λ = 0.1

"""
    loss_example(trajectories, ego_index, ground_truth_traj)

Computes the loss as the sum of:
  - The trajectory error (mean squared error between the computed trajectory and
    the ground truth trajectory), and
  - The regularization term on the mask (here the L2 norm of the mask).

- `trajectories`: Float32 array of size (N, T, d)
- `ego_index`: integer in 1:N
- `ground_truth_traj`: Float32 array of size (T, d)
"""
function loss_example(trajectories::Array{Float32,3}, ego_index::Int, ground_truth_traj::Array{Float32,2})
    # Prepare the network input and get the mask vector.
    input_vec = prepare_input(trajectories, ego_index)
    mask = model(input_vec)   # output: N-dimensional mask vector in [0, 1]
    
    # Use the mask with the differential game solver to get a computed trajectory.
    computed_traj = differential_game_solver(trajectories, mask)
    
    # Compute trajectory error (mean squared error).
    traj_error = mse(computed_traj, ground_truth_traj)
    
    # Regularization: norm of the mask.
    mask_reg = sum(mask)
    
    return traj_error + λ * mask_reg
end

###############################################################################
# Generate Dummy Training Data
###############################################################################
# For demonstration, we create dummy examples. In practice, use your actual data.
# Each training example consists of:
#   - trajectories: (N, T, d) array,
#   - ego_index: an integer (1:N),
#   - ground_truth_traj: (T, d) array computed from a "ground truth" mask.
batch_size = 100
training_data = Vector{Tuple{Array{Float32,3}, Int, Array{Float32,2}}}(undef, batch_size)

for i in 1:batch_size
    # Generate random trajectories for N players.
    trajectories = rand(Float32, N, T, d)
    
    # Randomly choose the ego-agent index.
    ego_index = rand(1:N)
    
    # For the ground truth, we assume there is a known binary mask indicating which
    # players are important. Here we randomly generate a ground truth mask.
    gt_mask = Float32.(rand(Bool, N))
    
    # Compute the ground truth trajectory as the weighted sum of the players’ trajectories.
    ground_truth_traj = zeros(Float32, T, d)
    for j in 1:N
        ground_truth_traj .+= gt_mask[j] .* trajectories[j, :, :]
    end
    
    training_data[i] = (trajectories, ego_index, ground_truth_traj)
end

###############################################################################
# Training Loop
###############################################################################
opt = ADAM(0.001)
epochs = 20

for epoch in 1:epochs
    total_loss = 0.0
    for (trajectories, ego_index, ground_truth_traj) in training_data
        # Compute gradients for the current example.
        gs = gradient(() -> loss_example(trajectories, ego_index, ground_truth_traj), params(model))
        # Update the network parameters.
        Flux.Optimise.update!(opt, params(model), gs)
        total_loss += loss_example(trajectories, ego_index, ground_truth_traj)
    end
    println("Epoch $epoch, Average Loss: $(total_loss / batch_size)")
end

###############################################################################
# Testing the Model on a New Example
###############################################################################
# Generate a new test example.
test_trajectories = rand(Float32, N, T, d)
test_ego = rand(1:N)
# Create a ground truth mask and the corresponding ground truth trajectory.
test_gt_mask = Float32.(rand(Bool, N))
test_ground_truth_traj = zeros(Float32, T, d)
for j in 1:N
    test_ground_truth_traj .+= test_gt_mask[j] .* test_trajectories[j, :, :]
end

# Prepare the test input and obtain the predicted mask.
input_test = prepare_input(test_trajectories, test_ego)
predicted_mask = model(input_test)

# Compute the trajectory using the (predicted) mask.
computed_traj_test = differential_game_solver(test_trajectories, predicted_mask)

println("\n===== Test Example =====")
println("Test ego-agent index: $test_ego")
println("Predicted mask vector (values near 1 indicate importance):")
println(predicted_mask)
println("Computed trajectory (from solver with predicted mask):")
println(computed_traj_test)
println("Ground truth trajectory:")
println(test_ground_truth_traj)
