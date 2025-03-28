"Utility for unpacking trajectory."
function unpack_trajectory(flat_trajectories; dynamics::ProductDynamics)
    horizon = Int(
        length(flat_trajectories[Block(1)]) /
        (state_dim(dynamics, 1) + control_dim(dynamics, 1)),
    )
    trajs = map(1:num_players(dynamics), blocks(flat_trajectories)) do ii, traj
        num_states = state_dim(dynamics, ii) * horizon
        X = reshape(traj[1:num_states], (state_dim(dynamics, ii), horizon))
        U = reshape(traj[(num_states + 1):end], (control_dim(dynamics, ii), horizon))

        (; xs = eachcol(X) |> collect, us = eachcol(U) |> collect)
    end

    stack_trajectories(trajs)
end

"Utility for packing trajectory."
function pack_trajectory(traj)
    trajs = unstack_trajectory(traj)
    mapreduce(vcat, trajs) do τ
        [reduce(vcat, τ.xs); reduce(vcat, τ.us)]
    end
end

"Pack an initial state and set of other params into a single parameter vector."
function pack_parameters(initial_state, other_params_per_player)
    mortar(map((x, θ) -> [x; θ], blocks(initial_state), blocks(other_params_per_player)))
end

"Unpack parameters into initial state and other parameters."
function unpack_parameters(params; dynamics::ProductDynamics)
    initial_state = mortar(map(1:num_players(dynamics), blocks(params)) do ii, θi
        θi[1:state_dim(dynamics, ii)]
    end)
    other_params = mortar(map(1:num_players(dynamics), blocks(params)) do ii, θi
        θi[((state_dim(dynamics, ii) + 1):end)]
    end)

    (; initial_state, other_params)
end

"Generate a BitVector mask with 0 entries corresponding to the initial states."
function parameter_mask(; dynamics::ProductDynamics, params_per_player)
    x = mortar([ones(state_dim(dynamics, i)) for i in 1:num_players(dynamics)])
    x̃ = 2mortar([ones(state_dim(dynamics, i)) for i in 1:num_players(dynamics)])
    θ = 3mortar([ones(kk) for kk in params_per_player])

    pack_parameters(x, θ) .== pack_parameters(x̃, θ)
end

"Convert a TrajectoryGame to a PrimalDualMCP."
function build_parametric_game(;
    game::TrajectoryGame,
    horizon = 10,
    params_per_player = 0, # not including initial state, which is always a param
)
    (;
        K_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds,
        dims,
        problems,
        shared_equality,
        shared_inequality,
    ) = build_mcp_components(; game, horizon, params_per_player)

    mcp = MixedComplementarityProblems.PrimalDualMCP(
        K_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds,
    )
    MixedComplementarityProblems.ParametricGame(
        problems,
        shared_equality,
        shared_inequality,
        dims,
        mcp,
    )
end

"Construct MCP components from game components."
function build_mcp_components(;
    game::TrajectoryGame,
    horizon = 10,
    params_per_player = 0, # not including initial state, which is always a param
)
    N = num_players(game)

    # Construct costs.
    function player_cost(τ, θi, ii)
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        ts = Iterators.eachindex(xs)
        map(xs, us, ts) do x, u, t
            game.cost.discount_factor^(t - 1) * game.cost.stage_cost[ii](x, u, t, θi)
        end |> game.cost.reducer
    end

    objectives = map(1:N) do ii
        (τ, θi) -> player_cost(τ, θi, ii)
    end

    # Shared equality constraints.
    shared_equality = (τ, θ) -> let
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        (; initial_state) = unpack_parameters(θ; game.dynamics)

        # Initial state constraint.
        g̃1 = xs[1] - initial_state

        # Dynamics constraints.
        ts = Iterators.eachindex(xs)
        g̃2 = mapreduce(vcat, ts[2:end]) do t
            xs[t] - game.dynamics(xs[t - 1], us[t - 1])
        end

        vcat(g̃1, g̃2)
    end

    # Shared inequality constraints.
    shared_inequality =
        (τ, θ) -> let
            (; xs, us) = unpack_trajectory(τ; game.dynamics)

            # Collision-avoidance constriant.
            h̃1 = game.coupling_constraints(xs, us, θ)

            # Environment boundaries.
            env_constraints = TrajectoryGamesBase.get_constraints(game.env)
            h̃2 = mapreduce(vcat, xs) do x
                env_constraints(x)
            end

            # Actuator/state limits.
            actuator_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                control_bounds(game.dynamics),
            )
            h̃3 = mapreduce(vcat, us) do u
                actuator_constraint(u)
            end

            state_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                state_bounds(game.dynamics),
            )
            h̃4 = mapreduce(vcat, xs) do x
                state_constraint(x)
            end

            vcat(h̃1, h̃2, h̃3, h̃4)
        end

    primal_dims = [
        horizon * (state_dim(game.dynamics, ii) + control_dim(game.dynamics, ii)) for
        ii in 1:N
    ]

    problems = map(
        f -> MixedComplementarityProblems.OptimizationProblem(; objective = f),
        objectives,
    )

    components = MixedComplementarityProblems.game_to_mcp(;
        test_point = BlockArray(zeros(sum(primal_dims)), primal_dims),
        test_parameter = mortar([
            zeros(state_dim(game.dynamics, ii) + params_per_player) for ii in 1:N
        ]),
        problems,
        shared_equality,
        shared_inequality,
    )

    (; problems, shared_equality, shared_inequality, components...)
end

"Generate an initial guess for primal variables following a zero input sequence."
function zero_input_trajectory(;
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    initial_state,
)
    rollout_strategy =
        map(1:num_players(game)) do ii
            (x, t) -> zeros(control_dim(game.dynamics, ii))
        end |> TrajectoryGamesBase.JointStrategy

    TrajectoryGamesBase.rollout(game.dynamics, rollout_strategy, initial_state, horizon)
end

"Generate an initial guess for primal variables following a custom input sequence."
function custom_input_trajectory(;
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    initial_state,
    parameters,
)
    rollout_strategy =
        map(1:num_players(game)) do ii
            (x, t) -> zeros(control_dim(game.dynamics, ii))
        end |> TrajectoryGamesBase.JointStrategy

    TrajectoryGamesBase.rollout(game.dynamics, rollout_strategy, initial_state, horizon)
end

"Solve a parametric trajectory game, where the parameter is just the initial state."
function TrajectoryGamesBase.solve_trajectory_game!(
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    parameter_value,
    strategy;
    parametric_game = build_parametric_game(;
        game,
        horizon,
        params_per_player = Int(
            (length(parameter_value) - state_dim(game)) / num_players(game),
        ),
    ),
    target,
)
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == :solved
        solution = MixedComplementarityProblems.solve(
            parametric_game,
            parameter_value;
            solver_type = MixedComplementarityProblems.InteriorPoint(),
            x₀ = strategy.last_solution.variables.x,
            y₀ = strategy.last_solution.variables.y,
        )
    else
        (; initial_state) = unpack_parameters(parameter_value; game.dynamics)
        solution = MixedComplementarityProblems.solve(
            parametric_game,
            parameter_value;
            solver_type = MixedComplementarityProblems.InteriorPoint(),
            x₀ = [
                pack_trajectory(zero_input_trajectory(; game, horizon, initial_state))
                zeros(sum(parametric_game.dims.λ) + parametric_game.dims.λ̃)
            ],
        )
    end

    # Update warm starting info.
    if solution.status == :solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    function gradient_test(parameter_value)
        if !isnothing(strategy.last_solution) && strategy.last_solution.status == :solved
            solution = MixedComplementarityProblems.solve(
                parametric_game,
                parameter_value;
                solver_type = MixedComplementarityProblems.InteriorPoint(),
                x₀ = strategy.last_solution.variables.x,
                y₀ = strategy.last_solution.variables.y,
            )
        else
            (; initial_state) = unpack_parameters(parameter_value; game.dynamics)
            solution = MixedComplementarityProblems.solve(
                parametric_game,
                parameter_value;
                solver_type = MixedComplementarityProblems.InteriorPoint(),
                x₀ = [
                    pack_trajectory(zero_input_trajectory(; game, horizon, initial_state))
                    zeros(sum(parametric_game.dims.λ) + parametric_game.dims.λ̃)
                ],
            )
        end
        # loss_similarity = sum(norm(solution.primals[1][j:j+1] - strategy.target[j:j+1]) for j in 1:2:120) / 60
        loss_similarity = sum(norm(solution.primals[1][j:j+1] - strategy.target[j:j+1]) for j in (horizon-input_horizon)*d+1 : d : horizon*d) / (horizon - input_horizon)
        loss_safety = minimum(norm(solution.primals[1][j:j+1] - strategy.target[(player_id-1) * horizon * d + j : (player_id-1) * horizon * d + j + 1]) for j in (horizon-input_horizon) * d + 1 : d : horizon * d for player_id in 2:N)
        loss_parameter_binary = sum(0.5 .- abs.(0.5 .- parameter_value[7+1:7+(N-1)])) / (N-1)
        loss_parameter_sum = sum(parameter_value[7+1:7+(N-1)]) / (N-1)
        
        loss = 10 * loss_similarity + 1.5 * loss_parameter_sum + 1 * loss_parameter_binary - loss_safety

        return loss
    end

    try
        gradient = only(Zygote.gradient(gradient_test, parameter_value))
    catch e
        println("Gradient computation failed, using random gradient.")
        gradient = randn(length(parameter_value))
    end
    gradient = gradient[7+1:7+(N-1)]
    loss = gradient_test(parameter_value)
    # # Pack solution into OpenLoopStrategy.
    trajs = unstack_trajectory(unpack_trajectory(mortar(solution.primals); game.dynamics))
    JointStrategy(map(traj -> OpenLoopStrategy(traj.xs, traj.us), trajs)), gradient, loss

end

"Receding horizon strategy that supports warm starting."
Base.@kwdef mutable struct WarmStartRecedingHorizonStrategy
    game::TrajectoryGame
    parametric_game::MixedComplementarityProblems.ParametricGame
    receding_horizon_strategy::Any = nothing
    time_last_updated::Int = 0
    turn_length::Int
    horizon::Int
    last_solution::Any = nothing
    parameters::Any = nothing
    solution_status::Any = nothing
    gradient::Any = nothing
    target::Any = nothing
    initial_states::Any = nothing
    loss::Any = nothing
end

function (strategy::WarmStartRecedingHorizonStrategy)(state, time)
    plan_exists = !isnothing(strategy.receding_horizon_strategy)
    time_along_plan = time - strategy.time_last_updated + 1
    plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

    update_plan = !plan_exists || !plan_is_still_valid
    if update_plan
        strategy.receding_horizon_strategy, strategy.gradient, strategy.loss = TrajectoryGamesBase.solve_trajectory_game!(
            strategy.game,
            strategy.horizon,
            pack_parameters(state, strategy.parameters),
            strategy;
            strategy.parametric_game,
            strategy.target,
        )

        strategy.time_last_updated = time
        time_along_plan = 1
    end
    strategy.receding_horizon_strategy(state, time_along_plan), strategy.gradient, strategy.loss
end

###############################################################################
# Game Setup
###############################################################################
"Utility to create the road environment."
function setup_road_environment(; length)
    vertices = [
        [-0.5length, -0.5length],
        [0.5length, -0.5length],
        [0.5length, 0.5length],
        [-0.5length, 0.5length],
    ]
    (; environment = PolygonEnvironment(vertices))
end

"Utility to set up a trajectory game."
function setup_trajectory_game(; environment, N)
    cost = let
        stage_costs = map(1:N) do ii
            (x, u, t, θi) -> let
            goal = θi[end-(N+1):end-N]
            mask = θi[end-(N-1):end]
                norm_sqr(x[Block(ii)][1:2] - goal) + norm_sqr(x[Block(ii)][3:4]) + 0.1 * norm_sqr(u[Block(ii)]) + 2 * sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
            end
        end

        function reducer(stage_costs)
            reduce(+, stage_costs) / length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(
            stage_costs,
            reducer,
            GeneralSumCostStructure(),
            1.0,
        )
    end

    function coupling_constraints(xs, us, θ)
        mapreduce(vcat, xs) do x
            player_states = blocks(x)  # Extract exactly N player states
            [ 1 ]
            # [
            #     norm_sqr(player_states[1][1:2] - player_states[2][1:2]) - 1 * θ[7] * θ[8],
            # ]
        end
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -2, -2], ub = [Inf, Inf, 2, 2]),
        control_bounds = (; lb = [-1, -1], ub = [1, 1]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:N])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end


###############################################################################
# Initialize GPU (if available)
###############################################################################
# device = cpu
# println("Using CPU.")

###############################################################################
# Neural Network Model
###############################################################################

function build_model()
    model = Chain(
        Dense(input_size, 256, relu),
        Dense(256, 64, relu),
        Dense(64, 16, relu),
        Dense(16, N-1, sigmoid)
    )
    return model
end

###############################################################################
# Data Processing Functions
###############################################################################
function prepare_input(trajectories::Dict{Int, Matrix{Float64}}, input_horizon, input_state_dim)
    flat_traj = vcat([vec(trajectories[i][1:input_horizon, 1:input_state_dim]) for i in 1:N]...)
    return flat_traj  # ✅ No `.device`
end

###############################################################################
# Integrate the Solver (Correct Input Format)
###############################################################################
function run_solver(game, parametric_game, target, initial_states, goals, N, horizon, num_sim_steps, mask)
    grad, loss = run_example(
        game = game,
        parametric_game = parametric_game,
        initial_states = initial_states,
        goals = goals,
        N = N,
        horizon = horizon,
        num_sim_steps = num_sim_steps,
        mask = mask,
        target = target,
    )
    return grad, loss  # Return as a single array
end

###############################################################################
# Load JSON Data
###############################################################################
function load_all_json_data(directory::String)
    json_files = glob("simulation_results_*.json", directory)
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
                    trajectories[i] = reshape(data_matrix, (horizon, d))  # Ensure correct shape
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
        ego_index = 1  # Ensure ego-agent is an active player

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
    batch_initial_states = []
    batch_goals = []
    batch_indices = []  # new vector for indices

    for i in state:min(state + dl.batch_size - 1, length(dl.dataset))
        idx = dl.indices[i]
        push!(batch_indices, idx)  # store the actual dataset index
        
        trajectories, ego_index, initial_states, goals = dl.dataset[idx]
        input_vec = prepare_input(trajectories, input_horizon, input_state_dim)
        push!(batch_inputs, input_vec)
        push!(batch_initial_states, initial_states)
        push!(batch_goals, goals)
        
        ground_truth_traj = Float64[]
        for j in 1:N
            append!(ground_truth_traj, vec(trajectories[j]))
        end
        push!(batch_targets, ground_truth_traj)
    end

    return ((hcat(batch_inputs...), hcat(batch_targets...), hcat(batch_initial_states...), hcat(batch_goals...), batch_indices), state + dl.batch_size)
end


###############################################################################
# Problem and Data Dimensions
###############################################################################
const N = 4      # Number of players
const horizon = 30     # Time steps in past trajectory
const d = 4      # State dimension per player
const input_horizon = 10  # Number of time steps in input trajectory
const input_state_dim = 4  # State dimension per player in input trajectory
const input_size = N * input_horizon * input_state_dim  # Input size for neural network
const num_sim_steps = 1  # Number of simulation steps

# Generate all possible masks dynamically (binary representation from 1 to 2^N - 1)
const masks = [bitstring(i)[end-N+1:end] |> x -> parse.(Int, collect(x)) for i in 1:(2^N - 1)]

###############################################################################
# Load Game
###############################################################################
(; environment) = setup_road_environment(; length = 10)
game = setup_trajectory_game(; environment, N = N)
parametric_game = build_parametric_game(; game, horizon=horizon, params_per_player = N + 2)

###############################################################################
# Load Dataset
###############################################################################
println("Loading dataset...")
train_dir = "/home/tq877/Tianyu/player_selection/MCP/data_train_$N/"
train_dataset = load_all_json_data(train_dir)
val_dir = "/home/tq877/Tianyu/player_selection/MCP/data_val_$N/"
val_dataset = load_all_json_data(val_dir)
test_dir = "/home/tq877/Tianyu/player_selection/MCP/data_test_$N/"
test_dataset = load_all_json_data(test_dir)
println("Training Dataset loaded successfully. Total samples: ", length(train_dataset))
println("Validation Dataset loaded successfully. Total samples: ", length(val_dataset))
println("Testing Dataset loaded successfully. Total samples: ", length(test_dataset))

# # Set batch size and initialize DataLoader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size)
val_dataloader = DataLoader(val_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

train_batches = length(train_dataset) / batch_size
val_batches = length(val_dataset) / batch_size
test_batches = length(test_dataset) / batch_size

epochs = 150  # Number of training epochs
global learning_rate = 0.01  # Learning rate for the optimizer 0.01 for bs=16, 0.005 for bs=4

###############################################################################
# Early Stopping Hyperparameters
###############################################################################
patience = epochs            # Number of epochs to wait for improvement
global patience_counter = 0
global best_val_loss = Inf      # Initialize best validation loss

###############################################################################
# Set Random Seed for Reproducibility
###############################################################################
using Random
seed = 2
Random.seed!(seed)  # Set the seed to a fixed value

# ihs = input_horizon, isd = input_state_dim, h = horizon
global record_name = "bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed _pat_$patience _N_$N _h_$horizon _ih$input_horizon _isd_$input_state_dim"