###############################################################################
# Initialize GPU (if available)
###############################################################################
# device = cpu
# println("Using CPU.")


###############################################################################
# Problem and Data Dimensions
###############################################################################
const N = 4      # Number of players
const T = 10     # Time steps in past trajectory
const d = 4      # State dimension per player
const input_size = N * T * d + N  # Input size for the neural network

# Generate all possible masks dynamically (binary representation from 1 to 2^N - 1)
const masks = [bitstring(i)[end-N+1:end] |> x -> parse.(Int, collect(x)) for i in 1:(2^N - 1)]

###############################################################################
# Neural Network Model
###############################################################################
function build_model()
    model = Chain(
        Dense(input_size, 64, relu),
        Dense(64, 32, relu),
        Dense(32, N),
        x -> σ.(x)  # Apply element-wise sigmoid activation
    )  # ✅ No more GPU move
    return model
end


###############################################################################
# Data Processing Functions
###############################################################################
function prepare_input(trajectories::Dict{Int, Matrix{Float64}}, ego_index::Int)
    flat_traj = vcat([vec(trajectories[i]) for i in 1:N]...)
    ego_onehot = zeros(Float64, N)
    ego_onehot[ego_index] = 1.0
    return vcat(flat_traj, ego_onehot)  # ✅ No `.device`
end


###############################################################################
# Integrate the Solver (Correct Input Format)
###############################################################################
function run_solver(mask, initial_states, goals, N, horizon, num_sim_steps)

    results = run_example(
        initial_states = initial_states,
        goals = goals,
        N = N,
        horizon = horizon,
        num_sim_steps = num_sim_steps,
        mask = mask
    )
    computed_traj = Float64[]
    for i in 1:N
        if haskey(results, "Player $i Trajectory")
            traj_result = results["Player $i Trajectory"]
            append!(computed_traj, vcat(traj_result[1]...))
        end
    end
    return computed_traj
end

###############################################################################
# Loss Function
###############################################################################
λ = 0.1  # Regularization strength

function loss_example(model, trajectories, ego_index, initial_states, goals, ground_truth_traj)
    input_vec = prepare_input(trajectories, ego_index)
    mask = model(input_vec)

    bin_mask = Float64.(mask .>= 0.5)
    computed_traj = run_solver(bin_mask, initial_states, goals, N)
    
    traj_error = mse(computed_traj, ground_truth_traj)
    mask_reg = sum(mask)
    return traj_error + λ * mask_reg
end

###############################################################################
# Load JSON Data
###############################################################################
function load_all_json_data(directory::String)
    json_files = glob("simulation_results_0*.json", directory)
    dataset = Vector{Tuple{Dict{Int,Array{Float64,2}}, Int, Array{Float64,1}, Array{Float64,1}}}()  

    for (idx, file) in enumerate(json_files)
        json_data = JSON.parsefile(file)

        # Extract trajectories for all players
        trajectories = Dict{Int, Array{Float64,2}}()  # Store (T, d) for all players
        for i in 1:N
            key = "Player $i Trajectory"
            if haskey(json_data, key)
                value = json_data[key]  # This should be a nested list of lists

                # Ensure `value` is a valid list of lists
                if isa(value, Vector) && all(x -> isa(x, Vector), value)
                    # Convert each inner list to a Float64 array
                    converted_values = [Float64.(vcat(x...)) for x in value]  # Flatten each sublist
                    data_matrix = hcat(converted_values...)'  # Stack into matrix and transpose to (T, d)
                    trajectories[i] = reshape(data_matrix, (T, d))  # Ensure correct shape
                else
                    error("Unexpected format for trajectory data in $file for Player $i")
                end
            else
                error("Missing trajectory data for Player $i in $file")
            end
        end

        # Extract initial states and goals
        initial_states = vcat([json_data["Player $i Initial State"] for i in 1:N]...)  # Shape (4N,)
        goals = vcat([json_data["Player $i Goal"] for i in 1:N]...)  # Shape (2N,)

        # Assign a valid mask and select an ego agent
        assigned_mask = masks[mod1(idx, length(masks))]
        active_players = findall(x -> x == 1, assigned_mask)
        ego_index = rand(active_players)  # Ensure ego-agent is an active player

        push!(dataset, (trajectories, ego_index, initial_states, goals))  # ✅ No redundant ground truth trajectory
    end

    return dataset
end

###############################################################################
# DataLoader
###############################################################################
struct DataLoader
    dataset::Vector{Tuple{Dict{Int, Matrix{Float64}}, Int, Vector{Float64}, Vector{Float64}}}  # Expecting 4 elements now
    batch_size::Int
    indices::Vector{Int}  # Shuffle indices for each epoch
end

function DataLoader(dataset::Vector{Tuple{Dict{Int,Matrix{Float64}}, Int, Vector{Float64}, Vector{Float64}}}, batch_size::Int)
    return DataLoader(dataset, batch_size, shuffle(1:length(dataset)))
end

function Base.iterate(dl::DataLoader, state=1)
    if state > length(dl.dataset)
        return nothing
    end

    batch_inputs = []
    batch_targets = []

    for i in state:min(state + dl.batch_size - 1, length(dl.dataset))
        idx = dl.indices[i]
        trajectories, ego_index, initial_states, goals = dl.dataset[idx]

        input_vec = prepare_input(trajectories, ego_index)  # ✅ No `.device`
        push!(batch_inputs, input_vec)

        # Flatten ground truth trajectories similar to `computed_traj`
        ground_truth_traj = Float64[]
        for j in 1:N
            append!(ground_truth_traj, vec(trajectories[j]))
        end

        push!(batch_targets, ground_truth_traj)  # ✅ No `.device`
    end

    return (hcat(batch_inputs...), hcat(batch_targets...)), state + dl.batch_size
end
