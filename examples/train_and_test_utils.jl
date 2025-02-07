###############################################################################
# Initialize GPU (if available)
###############################################################################
device = CUDA.has_cuda() ? gpu : cpu
if CUDA.has_cuda()
    println("Using GPU.")
else
    println("Using CPU.")
end

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
        σ  # Sigmoid activation for mask output
    ) |> device
    return model
end

###############################################################################
# Data Processing Functions
###############################################################################
function prepare_input(trajectories::Dict{Int,Array{Float32,3}}, ego_index::Int)
    flat_traj = vcat([vec(trajectories[i]) for i in 1:N]...)
    ego_onehot = zeros(Float32, N)
    ego_onehot[ego_index] = 1.0
    return vcat(flat_traj, ego_onehot)
end

###############################################################################
# Integrate the Solver (Correct Input Format)
###############################################################################
function run_solver(mask::Vector{Float32}, initial_states::Array{Float32,1}, 
                    goals::Array{Float32,1}, N::Int, horizon=3, num_sim_steps=T)
    results = run_example(
        initial_states = initial_states,
        goals = goals,
        N = N,
        horizon = horizon,
        num_sim_steps = num_sim_steps,
        mask = mask
    )

    computed_traj = zeros(Float32, T, d)
    for i in 1:N
        if mask[i] >= 0.5 && haskey(results, "Player $i Trajectory")
            computed_traj .+= results["Player $i Trajectory"][end]
        end
    end
    return computed_traj
end

###############################################################################
# Loss Function
###############################################################################
λ = 0.1  # Regularization strength

function loss_example(model, trajectories, ego_index, initial_states, goals, ground_truth_traj)
    input_vec = prepare_input(trajectories, ego_index) |> device
    mask = model(input_vec)

    bin_mask = Float32.(mask .>= 0.5) |> cpu
    computed_traj = run_solver(bin_mask, initial_states, goals, N)
    
    traj_error = mse(computed_traj |> device, ground_truth_traj |> device)
    mask_reg = sum(mask)
    return traj_error + λ * mask_reg
end

###############################################################################
# Load JSON Data
###############################################################################
function load_all_json_data(directory::String)
    json_files = glob("*.json", directory)
    dataset = Vector{Tuple{Dict{Int,Array{Float32,3}}, Int, Array{Float32,1}, Array{Float32,1}, Array{Float32,2}}}()
    
    for (idx, file) in enumerate(json_files)
        json_data = JSON.parsefile(file)

        trajectories = Dict()
        for (key, value) in json_data
            if occursin("Trajectory", key)
                player_num = parse(Int, split(key, " ")[2])
                if isa(value, Tuple)
                    data_matrix = hcat([Float32.(v) for v in value]...)'  
                    trajectories[player_num] = reshape(data_matrix, (T, d))
                else
                    error("Unexpected format for trajectory data in $file")
                end
            end
        end

        initial_states = vcat([Float32.(json_data["Player $i Initial State"]) for i in 1:N]...)
        goals = vcat([Float32.(json_data["Player $i Goal"]) for i in 1:N]...)

        assigned_mask = masks[mod1(idx, length(masks))]
        active_players = findall(x -> x == 1, assigned_mask)
        ego_index = rand(active_players)

        ground_truth_traj = zeros(Float32, T, d)
        for i in 1:N
            ground_truth_traj .+= assigned_mask[i] .* trajectories[i]
        end

        push!(dataset, (trajectories, ego_index, initial_states, goals, ground_truth_traj))
    end
    
    return dataset
end

###############################################################################
# DataLoader
###############################################################################
struct DataLoader
    dataset::Vector{Tuple{Dict{Int,Array{Float32,3}}, Int, Array{Float32,1}, Array{Float32,1}, Array{Float32,2}}}
    batch_size::Int
    indices::Vector{Int}  
end

function DataLoader(dataset::Vector{Tuple{Dict{Int,Array{Float32,3}}, Int, Array{Float32,1}, Array{Float32,1}, Array{Float32,2}}}, batch_size::Int)
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
        trajectories, ego_index, initial_states, goals, ground_truth_traj = dl.dataset[idx]

        input_vec = prepare_input(trajectories, ego_index) |> device  
        push!(batch_inputs, input_vec)
        push!(batch_targets, ground_truth_traj |> device)  
    end

    return (hcat(batch_inputs...), hcat(batch_targets...)), state + dl.batch_size
end
