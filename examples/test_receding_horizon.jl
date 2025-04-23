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
    elseif mode == "Neural Network Partial Rank"
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
    elseif mode == "Jacobian"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else
            mask = zeros(N-1)
            delta_t = 0.1 # hard coded for now
            norm_costs = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end] - trajectory[player_id][end-3:end]) # ex: pi_{k,x} - pj_{k,x}
                delta_px = (state_differences[1] + delta_t * state_differences[3]) ^ 2 # ex: (pi_{k,x} - pj_{k,x} + delta_t * (vi_{k,x} - vj_{k,x})) ^ 2
                delta_py = (state_differences[2] + delta_t * state_differences[4]) ^ 2
                delta_vx = (state_differences[3] + delta_t * control[player_id][1]) ^ 2
                delta_vy = (state_differences[4] + delta_t * control[player_id][2]) ^ 2
                D = delta_px + delta_py + delta_vx + delta_vy # denominator of l_col between player i and j = player_id
                J1 = 1/(D ^ 2) * 2 * delta_vx * delta_t # partial derivative of l_col with respect to aj_{k,x}
                J2 = 1/(D ^ 2) * 2 * delta_vy * delta_t # partial derivative of l_col with respect to aj_{k,y}
                norm_costs[player_id-1] = norm([J1, J2]) # [J1, J2] is the jacobian of the cost function with respect to the control of player_id
            end
            ranked_indices = rank_array_from_large_to_small(norm_costs) # rank the players based on the norm of the jacobian
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Hessian"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # mode_parameter for initial nearest neighbor is all players for now
        else
            mask = zeros(N-1)
            delta_t = 0.1 # hard coded for now
            norm_costs = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end] - trajectory[player_id][end-3:end]) # ex: pi_{k,x} - pj_{k,x}
                delta_px = (state_differences[1] + delta_t * state_differences[3]) ^ 2 # ex: (pi_{k,x} - pj_{k,x} + delta_t * (vi_{k,x} - vj_{k,x})) ^ 2
                delta_py = (state_differences[2] + delta_t * state_differences[4]) ^ 2
                delta_vx = (state_differences[3] + delta_t * control[player_id][1]) ^ 2
                delta_vy = (state_differences[4] + delta_t * control[player_id][2]) ^ 2
                D = delta_px + delta_py + delta_vx + delta_vy # denominator of l_col between player i and j = player_id
                H11 = 2 * delta_t ^ 2 / D ^ 3 * (4*delta_vx ^ 2 - D) # terms of the hessian
                H12 = 8 * delta_t ^ 2 / D ^ 3 * delta_vx * delta_vy
                H22 = 2 * delta_t ^ 2 / D ^ 3 * (4*delta_vy ^ 2 - D)
                norm_costs[player_id-1] = norm([H11 H12; H12 H22]) # Frobenius norm of the Hessian
            end
            ranked_indices = rank_array_from_large_to_small(norm_costs) # rank the players based on the norm of the jacobian
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
    elseif mode == "Control Barrier Function"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # can't compute cost evolution at sim_step 1
        else
            mask = zeros(N-1)
            cbf_values = zeros(N-1)
            R = 0.5 # may need to change
            kappa = 5.0 # may need to change
            for playerid in 2:N
                position_difference = trajectory[1][end-3:end-2] - trajectory[playerid][end-3:end-2] # xi - xj
                velocity_difference = trajectory[1][end-1:end] - trajectory[playerid][end-1:end] # vi - vj
                acceleration_difference = control[1] - control[playerid] # ai - aj
                h = sum(position_difference.^2) - R^2 # ||xi - xj||^2 - R^2
                h_dot = 2 * position_difference' * velocity_difference # 2 * (xi - xj)' * (vi - vj)
                h_ddot = 2 * (velocity_difference' * velocity_difference + position_difference' * acceleration_difference) # 2 * (vi - vj)' * (vi - vj) + 2 * (xi - xj)' * (ai - aj)
                f_CBF = h_ddot + 2 * kappa * h_dot + kappa^2 * h # CBF value for ego w.r.t player_id
                cbf_values[playerid-1] = f_CBF # f_CBF for player_id
            end
            ranked_indices = rank_array_from_small_to_large(cbf_values) # rank the players based on CBF values (small = danger)
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    else
        error("Invalid mode: $mode")
    end

    return mask
end

###############################################################################
# Evaluation Code
###############################################################################

println("\nLoading best model for testing...")
# Use the same record_name as in training
# best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/trained_model.bson")
best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/best_model.bson")
# best_model_data = BSON.load("C:/UT Austin/Research/MCP/examples/logs/bs_32 _ep_100 _lr_0.01 _sd_3 _pat_100 _N_4 _h_30 _ih10 _isd_4/trained_model.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

for mode in evaluation_modes
    for mode_parameter in mode_parameters[mode]
        println("\nTesting mode: $mode with parameter: $mode_parameter")
        for scenario_id in 40:47
            println("Scenario $scenario_id")
            file_path = joinpath(test_dir, "scenario_$scenario_id.csv")
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
            for sim_step in 1:50
                println("Sim Step: $sim_step")
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

            # results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/similarity_safety_scores_$mode.json"
            results_path = "$test_dir/receding_horizon_trajectories_[$scenario_id]_[$mode]_[$mode_parameter].json"
            open(results_path, "w") do io
                JSON.print(io, receding_horizon_result)
            end
        end
    end
end
