###############################################################################
# Test Code: Load the Best Model and Evaluate on Test Data
###############################################################################
using BSON
using Flux

###############################################################################
# Metrics
###############################################################################

function mask_computation(input_traj, trajectory, control, mode, sim_step, mode_parameter)
    mask = zeros(N-1)
    if mode == "All"
        mask = ones(N-1)
    end
    return mask
end

time_list = []

for N in 2:10
    println("N = $N")
    game = setup_trajectory_game(; environment, N = N)
    parametric_game = build_parametric_game(; game, horizon=horizon, params_per_player = N + 2)
    for mode in evaluation_modes
        for mode_parameter in mode_parameters[mode]
            # println("\nTesting mode: $mode with parameter: $mode_parameter")
            for scenario_id in 40:40
                file_path = joinpath(test_dir, "scenario_$scenario_id.csv")
                data = CSV.read(file_path, DataFrame)
                goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
                initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
                trajectory = Dict{Int64, Vector{Float64}}()
                receding_horizon_result = Dict()
                for player_id in 1:N
                    trajectory[player_id] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
                    receding_horizon_result["Player $player_id Trajectory"] = [initial_states[4 * (player_id - 1) + 1:4 * player_id]]
                    receding_horizon_result["Player $player_id Initial State"] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
                    receding_horizon_result["Player $player_id Goal"] = goals[2 * (player_id - 1) + 1:2 * player_id]
                    receding_horizon_result["Player $player_id Control"] = []
                end
                latest_control = []
                receding_horizon_result["Player 1 Mask"] = []
                global tstart = time()
                for sim_step in 1:10
                    if sim_step == 2
                        global tstart = time()
                    end
                    # println("Sim Step: $sim_step")
                    input_traj = []
                    current_mask = mask_computation(input_traj, trajectory, latest_control, mode, sim_step, mode_parameter)
                    pred_mask = vcat([1], current_mask)
                    push!(receding_horizon_result["Player 1 Mask"], pred_mask)
                    target = [0 for _ in 1:N * horizon * d]
                    results = run_example(
                        game = game,
                        parametric_game = parametric_game,
                        initial_states = initial_states,
                        goals = goals,
                        N = N,
                        horizon = horizon,
                        num_sim_steps = num_sim_steps,
                        mask = pred_mask,
                        target = target,
                        save = true
                    )
                    initial_states = vec(vcat([results["Player $player_id Latest Initial State"] for player_id in 1:N]...))
                    for player_id in 1:N
                        push!(receding_horizon_result["Player $player_id Trajectory"], results["Player $player_id Latest Initial State"])
                        push!(receding_horizon_result["Player $player_id Control"], results["Player $player_id Latest Control"])
                        push!(latest_control, results["Player $player_id Latest Control"])
                    end              
                end
                end
            end
        end
    tend = time()
    println("Time taken for N = $N: ", tend - tstart)
    push!(time_list, tend - tstart)
end

# Save the time_list to a file
open("time_list.json", "w") do file
    JSON.print(file, time_list)
end
