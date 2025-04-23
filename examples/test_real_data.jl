###############################################################################
# Test Code: Load the Best Model and Evaluate on Test Data
###############################################################################
using BSON
using Flux

###############################################################################
# Metrics
###############################################################################

function rank_array_from_small_to_large(array)
    sorted_indices = sortperm(array)
    return sorted_indices
end

function rank_array_from_large_to_small(array)
    sorted_indices = sortperm(array, rev=true)
    return sorted_indices
end

function mask_computation(input_traj, trajectory, control, mode, sim_step, mode_parameter)
    mask = zeros(N-1)
    if mode == "All"
        mask = ones(N-1)
    elseif mode == "Neural Network Threshold"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, control, "Distance Threshold", sim_step, 2)
        else
            mask = best_model(input_traj)
            mask = map(x -> x > mode_parameter ? 1 : 0, mask)
        end
    elseif mode == "Neural Network Partial Threshold"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, control, "Distance Threshold", sim_step, 2)
        else
            mask = best_model(input_traj)
            mask = map(x -> x > mode_parameter ? 1 : 0, mask)
        end
    elseif mode == "Distance Threshold"
        mask = zeros(N-1)
        for player_id in 2:N
            distance = norm(trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2])
            if distance <= mode_parameter
                mask[player_id-1] = 1
            else
                mask[player_id-1] = 0
            end
        end
    elseif mode == "Nearest Neighbor"
        mask = zeros(N-1)
        distances = []
        for player_id in 2:N
            distance = norm(trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2])
            push!(distances, distance)
        end
        ranked_indices = rank_array_from_small_to_large(distances)
        for i in 1:mode_parameter-1
            mask[ranked_indices[i]] = 1
        end
    elseif mode == "Neural Network Rank"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else
            model_mask = best_model(input_traj)
            ranked_indices = rank_array_from_large_to_small(model_mask)
            mask = zeros(N-1)
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Cost Evolution"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # can't compute cost evolution at sim_step 1
        else
            mask = zeros(N-1)
            mu = 1 # hard coded for now
            cost_evolution_values = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2]) # ex: pi_{k,x} - pj_{k,x}
                D = sum(state_differences .^ 2) # denominator of mu/norm(xi_k - xj_k)^2
        
                # x_k-1 values (prior states) for player_id
                state_differences_prev = (trajectory[1][end-7:end-6] - trajectory[player_id][end-7:end-6]) # state difference for previous sim_step
                D_prev = sum(state_differences_prev .^ 2) # denominator of mu/norm(xi_k-1 - xj_k-1)^2
                
                cost_evolution_values[player_id-1] = mu / D - mu / D_prev # cost evolution value for player_id
            end
            ranked_indices = rank_array_from_large_to_small(cost_evolution_values) # rank the players based on the norm of the jacobian
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Barrier Function"
        mask = zeros(N-1)
        bf_values = zeros(N-1)
        R = 0.5 # may need to change
        kappa = 5.0 # may need to change
        for playerid in 2:N
            position_difference = trajectory[1][end-3:end-2] - trajectory[playerid][end-3:end-2]# xi - xj
            velocity_difference = trajectory[1][end-1:end] - trajectory[playerid][end-1:end] # vi - vj
            h = sum(position_difference.^2) - R^2 # ||xi - xj||^2 - R^2
            h_dot = 2 * position_difference' * velocity_difference # 2 * (xi - xj)' * (vi - vj)
            f_BF = h_dot+kappa*h
            bf_values[playerid-1] = f_BF # f_BF for player_id
        end
        ranked_indices = rank_array_from_small_to_large(bf_values) # rank the players based on BF values (small = danger)
        for i in 1:mode_parameter-1
            mask[ranked_indices[i]] = 1
        end
    else
        error("Invalid mode: $mode")
    end

    return mask
end

# (; environment) = setup_road_environment(; length = 75)
(; environment) = setup_real_environment(; xmin = 18.5, xmax = 26, ymin = 2, ymax = 23.5)
game = setup_real_game(; environment, N = N)
parametric_game = build_parametric_game(; game, horizon=horizon, params_per_player = N + 2)


###############################################################################
# Evaluation Code
###############################################################################

println("\nLoading best model for testing...")
best_model_data = BSON.load("C:/UT Austin/Research/MCP/examples/logs/bs_2 _ep_100 _lr_0.005 _sd_3 _pat_100 _N_10 _h_30 _ih10 _isd_2 _w_[11.0, 1.5, 1.0]/best_model.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

# Initialize the dictionary with empty integer arrays for each key
# time_dict = []

time_dict = [115, 81, 60, 107, 126, 102] # Example values for each scenario

results_dir = "C:/UT Austin/Research/MCP/data_ped/"

for scenario_id in 1:6
    for mode in real_evaluation_modes
        for mode_parameter in mode_parameters[mode]
            file_path = joinpath(results_dir, "scenario$scenario_id.csv")
            data = CSV.read(file_path, DataFrame)
            goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
            initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
            trajectory = Dict{Int64, Vector{Float64}}()
            partial_trajectory = Dict{Int64, Vector{Float64}}()
            receding_horizon_result = Dict()
            for player_id in 1:N
                trajectory[player_id] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
                partial_trajectory[player_id] = initial_states[4 * (player_id - 1) + 1:4 * (player_id - 1) + 2]
                receding_horizon_result["Player $player_id Trajectory"] = [initial_states[4 * (player_id - 1) + 1:4 * player_id]]
                receding_horizon_result["Player $player_id Initial State"] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
                receding_horizon_result["Player $player_id Goal"] = goals[2 * (player_id - 1) + 1:2 * player_id]
                receding_horizon_result["Player $player_id Control"] = []
            end
            latest_control = []
            receding_horizon_result["Player 1 Mask"] = []
            for sim_step in 1:time_dict[scenario_id]
                # println("Sim Step: $sim_step")
                println("Scenario: $scenario_id, Mode: $mode, Parameter: $mode_parameter, Sim Step: $sim_step")
                input_traj = []
                if sim_step <= 10
                    if sim_step > 1
                        for player_id in 1:N
                            trajectory[player_id] = vcat(trajectory[player_id], initial_states[4 * (player_id - 1) + 1:4 * player_id])
                            partial_trajectory[player_id] = vcat(partial_trajectory[player_id], initial_states[4 * (player_id - 1) + 1:4 * (player_id - 1) + 2])
                        end
                    end            
                else
                    trajectory = Dict(player_id => vcat(trajectory[player_id][5:end], reshape(initial_states[4 * (player_id - 1) + 1:4 * player_id], :)) for player_id in 1:N)
                    partial_trajectory = Dict(player_id => vcat(partial_trajectory[player_id][3:end], reshape(initial_states[4 * (player_id - 1) + 1:4 * (player_id - 1) + 2], :)) for player_id in 1:N)
                    if mode == "Neural Network Partial Threshold" || mode == "Neural Network Partial Rank"
                        input_traj = vec(vcat([reshape(partial_trajectory[player_id], :) for player_id in 1:N]...))
                    else
                        input_traj = vec(vcat([reshape(trajectory[player_id], :) for player_id in 1:N]...))
                    end
                end
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

            results_path = "$results_dir/trajectories_[$scenario_id]_[$mode]_[$mode_parameter].json"
            open(results_path, "w") do io
                JSON.print(io, receding_horizon_result)
            end
        end
    end
end
