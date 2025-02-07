###############################################################################
# Problem and Data Dimensions
###############################################################################
const N = 4      # Total number of players
const T = 10     # Number of time steps in the past trajectory
const d = 4      # State dimension for each player
const input_size = N * T * d + N  # Input size for the neural network

# Generate all possible masks dynamically (binary representation from 1 to 2^N - 1)
const masks = [bitstring(i)[end-N+1:end] |> x -> parse.(Int, collect(x)) for i in 1:(2^N - 1)]

###############################################################################
# Load JSON Data (ALL FILES)
###############################################################################
"""
    load_all_json_data(directory::String)

Loads all JSON scenario files in the given directory and assigns each one a valid mask.

Returns:
- `dataset`: Vector of tuples containing:
    - Trajectories dictionary (player → (T, d))
    - Ego agent index (chosen from active players in mask)
    - Initial states
    - Goals
    - Ground truth trajectory
"""
function load_all_json_data(directory::String)
    json_files = glob("*.json", directory)  # Find all JSON files
    dataset = Vector{Tuple{Dict{Int,Array{Float32,3}}, Int, Array{Float32,1}, Array{Float32,1}, Array{Float32,2}}}()
    
    for (idx, file) in enumerate(json_files)
        json_data = JSON.parsefile(file)

        # Extract trajectories
        trajectories = Dict()
        for (key, value) in json_data
            if occursin("Trajectory", key)
                player_num = parse(Int, split(key, " ")[2])
                trajectories[player_num] = permutedims(Float32.(value[1]), (1, 2))  # Convert list to matrix
            end
        end
        
        # Extract initial states & goals
        initial_states = vcat([Float32.(json_data["Player $i Initial State"]) for i in 1:N]...)
        goals = vcat([Float32.(json_data["Player $i Goal"]) for i in 1:N]...)

        # Assign mask based on scenario index (loop over masks dynamically)
        assigned_mask = masks[mod1(idx, length(masks))]

        # **Ensure ego-agent is selected from active players**
        active_players = findall(x -> x == 1, assigned_mask)
        ego_index = rand(active_players)  # Pick a random player who is active

        # Generate ground truth trajectory using the mask
        ground_truth_traj = zeros(Float32, size(trajectories[1]))
        for i in 1:N
            ground_truth_traj .+= assigned_mask[i] .* trajectories[i]
        end

        push!(dataset, (trajectories, ego_index, initial_states, goals, ground_truth_traj))
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
    dataset::Vector{Tuple{Dict{Int,Array{Float32,3}}, Int, Array{Float32,1}, Array{Float32,1}, Array{Float32,2}}}
    batch_size::Int
end

function Base.iterate(dl::DataLoader, state=1)
    if state > length(dl.dataset)
        return nothing
    end

    batch_inputs = []
    batch_targets = []

    for i in state:min(state + dl.batch_size - 1, length(dl.dataset))
        trajectories, ego_index, initial_states, goals, ground_truth_traj = dl.dataset[i]
        input_vec = prepare_input(trajectories, ego_index)
        
        push!(batch_inputs, input_vec)
        push!(batch_targets, ground_truth_traj)
    end

    return (hcat(batch_inputs...), hcat(batch_targets...)), state + dl.batch_size
end

###############################################################################
# Training the Model
###############################################################################
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_all_json_data(directory)

# Initialize DataLoader
batch_size = 2 #10
dataloader = DataLoader(dataset, batch_size)

# Optimizer
opt = ADAM(0.001)
epochs = 2 #20

# Training loop
for epoch in 1:epochs
    total_loss = 0.0
    for (batch_inputs, batch_targets) in dataloader
        loss_val, grads = Flux.withgradient(params(model)) do
            pred_masks = model(batch_inputs)
            bin_mask = Float32.(pred_masks .>= 0.5)  # Convert to binary
            computed_traj = run_solver(bin_mask, dataset[1][3], dataset[1][4], N)
            traj_error = mse(computed_traj, batch_targets)
            mask_reg = sum(pred_masks)
            traj_error + λ * mask_reg
        end
        
        Flux.Optimise.update!(opt, params(model), grads)
        total_loss += loss_val
    end
    println("Epoch $epoch, Average Loss: $(total_loss / length(dataloader.dataset))")
end

# Save the trained model
# Save model after training
BSON.bson("trained_model.bson", Dict(:model => model))
println("Model saved successfully!")
