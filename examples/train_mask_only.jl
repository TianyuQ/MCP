###############################################################################
# Import Required Packages
###############################################################################
using Flux
using Flux.Losses: mse
using JSON
using Statistics
using Random
using LinearAlgebra
using Glob  # To dynamically find all JSON files

###############################################################################
# Problem and Data Dimensions
###############################################################################
const N = 4      # Total number of players
const T = 10     # Number of time steps in the past trajectory
const d = 4      # State dimension for each player
const input_size = N * T * d + N  # Input size for the neural network

# Predefined masks (one per scenario)
const masks = [bitstring(i)[end-N+1:end] |> x -> parse.(Int, collect(x)) for i in 1:(2^N - 1)]

###############################################################################
# Neural Network Model
###############################################################################
model = Chain(
    Dense(input_size, 64, relu),
    Dense(64, 32, relu),
    Dense(32, N),
    σ  # Sigmoid activation for mask output
)

###############################################################################
# Input Preparation Function
###############################################################################
"""
    prepare_input(trajectories, ego_index)

- `trajectories`: Dictionary with player trajectories (N, T, d)
- `ego_index`: Integer (1:N) representing the ego-agent index

Returns:
- A concatenated vector combining flattened trajectories and one-hot ego-agent encoding
"""
function prepare_input(trajectories::Dict{Int,Array{Float32,3}}, ego_index::Int)
    flat_traj = vcat([vec(trajectories[i]) for i in 1:N]...)
    ego_onehot = zeros(Float32, N)
    ego_onehot[ego_index] = 1.0
    return vcat(flat_traj, ego_onehot)
end

###############################################################################
# Differential Game Solver Function (Dummy Placeholder)
###############################################################################
"""
    differential_game_solver(trajectories, mask)

Computes the predicted trajectory using the mask.

- `trajectories`: Dictionary with (N, T, d) player trajectories
- `mask`: N-dimensional vector (values between [0,1])

Returns:
- Computed trajectory (T, d)
"""
function differential_game_solver(trajectories::Dict{Int,Array{Float32,3}}, mask::AbstractVector)
    T, d = size(trajectories[1], 1), size(trajectories[1], 2)
    computed_traj = zeros(Float32, T, d)
    for i in 1:N
       computed_traj .+= mask[i] .* trajectories[i]
    end
    return computed_traj
end

###############################################################################
# Loss Function
###############################################################################
λ = 0.1  # Regularization strength

"""
    loss_example(trajectories, ego_index, ground_truth_traj)

Computes:
- Mean squared error between computed trajectory and ground truth
- Regularization on the mask (sum norm)

Returns:
- Total loss
"""
function loss_example(trajectories::Dict{Int,Array{Float32,3}}, ego_index::Int, ground_truth_traj::Array{Float32,2})
    input_vec = prepare_input(trajectories, ego_index)
    mask = model(input_vec)  # N-dimensional mask output
    computed_traj = differential_game_solver(trajectories, mask)
    
    traj_error = mse(computed_traj, ground_truth_traj)
    mask_reg = sum(mask)  # Regularization term
    return traj_error + λ * mask_reg
end

###############################################################################
# Load JSON Data (ALL FILES)
###############################################################################
"""
    load_all_json_data(directory::String)

Loads all JSON scenario files in the given directory and assigns each one a mask.

Returns:
- `dataset`: Vector of tuples containing:
    - Trajectories dictionary (player → (T, d))
    - Assigned mask
    - Ego agent index
    - Ground truth trajectory
"""
function load_all_json_data(directory::String)
    json_files = glob("*.json", directory)  # Find all JSON files
    dataset = Vector{Tuple{Dict{Int,Array{Float32,3}}, Vector{Int}, Int, Array{Float32,2}}}()
    
    for file in json_files
        json_data = JSON.parsefile(file)

        # Extract trajectories
        trajectories = Dict()
        for (key, value) in json_data
            if occursin("Trajectory", key)
                player_num = parse(Int, split(key, " ")[2])
                trajectories[player_num] = permutedims(Float32.(value[1]), (1, 2))  # Convert list to matrix
            end
        end
        
        # Extract mask from file name
        mask_str = match(r"\[(.*?)\]", file).match
        assigned_mask = parse.(Int, split(mask_str, ", "))

        # Traverse all possible ego-agents
        for ego_index in 1:N
            if assigned_mask[ego_index] == 1
                # Generate ground truth trajectory using the assigned mask
                ground_truth_traj = zeros(Float32, size(trajectories[1]))
                for i in 1:N
                    ground_truth_traj .+= assigned_mask[i] .* trajectories[i]
                end

                push!(dataset, (trajectories, assigned_mask, ego_index, ground_truth_traj))
            end
        end
    end
    
    return dataset
end

###############################################################################
# DataLoader Implementation
###############################################################################
"""
    DataLoader(dataset, batch_size)

Creates an iterable dataloader for training.

Yields:
- `batch_inputs`: (batch_size, input_size)
- `batch_targets`: (batch_size, T, d)
"""
struct DataLoader
    dataset::Vector{Tuple{Dict{Int,Array{Float32,3}}, Vector{Int}, Int, Array{Float32,2}}}
    batch_size::Int
end

function Base.iterate(dl::DataLoader, state=1)
    if state > length(dl.dataset)
        return nothing
    end

    batch_inputs = []
    batch_targets = []

    for i in state:min(state + dl.batch_size - 1, length(dl.dataset))
        trajectories, assigned_mask, ego_index, ground_truth_traj = dl.dataset[i]
        input_vec = prepare_input(trajectories, ego_index)
        
        push!(batch_inputs, input_vec)
        push!(batch_targets, ground_truth_traj)
    end

    return (hcat(batch_inputs...), hcat(batch_targets...)), state + dl.batch_size
end

###############################################################################
# Training the Model
###############################################################################
directory = "/mnt/data/"
dataset = load_all_json_data(directory)

# Initialize DataLoader
batch_size = 10
dataloader = DataLoader(dataset, batch_size)

# Optimizer
opt = ADAM(0.001)
epochs = 20

# Training loop
for epoch in 1:epochs
    total_loss = 0.0
    for (batch_inputs, batch_targets) in dataloader
        loss_val, grads = Flux.withgradient(params(model)) do
            pred_masks = model(batch_inputs)
            computed_traj = differential_game_solver(dataset[1][1], pred_masks)
            traj_error = mse(computed_traj, batch_targets)
            mask_reg = sum(pred_masks)
            traj_error + λ * mask_reg
        end
        
        Flux.Optimise.update!(opt, params(model), grads)
        total_loss += loss_val
    end
    println("Epoch $epoch, Average Loss: $(total_loss / length(dataloader.dataset))")
end
