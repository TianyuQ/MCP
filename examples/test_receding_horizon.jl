###############################################################################
# Test Code: Load the Best Model and Evaluate on Test Data
###############################################################################
using BSON
using Flux

###############################################################################
# Metrics
###############################################################################

function similarity_analysis(results, target)
    diff = sum(norm(results[j:j+1] - target[j:j+1]) for j in (horizon-input_horizon)*d + 1 : d : horizon*d)
    return diff
end

function safety_analysis(results, target)
    safeness = minimum(norm(results[j:j+1] - target[(player_id-1) * horizon * d + j : (player_id-1) * horizon * d + j + 1]) for j in (horizon-input_horizon) * d + 1 : d : horizon * d for player_id in 2:N)
    return safeness
end

function mask_computation(input_traj, trajectory, mode, sim_step)
    if mode == "All"
        mask = ones(N-1)
    elseif mode == "Neural Network"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, "Distance Threshold", sim_step)
        else
            mask = best_model(input_traj)
            println("Pred Mask: ", round.(mask, digits=4))
            mask = map(x -> x > 0.05 ? 1 : 0, mask)
            # println("Pred Mask: ", mask)
        end
    elseif mode == "Distance Threshold"
        mask = zeros(N-1)
        for player_id in 2:N
            distance = norm(trajectory[1][end-3:end] - trajectory[player_id][end-3:end])
            if distance <= 2.5
                mask[player_id-1] = 1
            else
                mask[player_id-1] = 0
            end
        end
        # println("Pred Mask: ", mask)
    end
    return mask
end

###############################################################################
# Evaluation Code
###############################################################################

println("\nLoading best model for testing...")
# Use the same record_name as in training
# best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/trained_model.bson")
best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/logs/bs_16 _ep_100 _lr_0.01 _sd_3 _pat_100 _N_4 _h_30 _ih10 _isd_4/trained_model.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

for mode in evaluation_modes
    for scenario_id in 160:191
        println("Scenario $scenario_id")
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
        receding_horizon_result["Player 1 Mask"] = []
        for sim_step in 1:50
            input_traj = []
            # println("Sim Step: ", sim_step)
            if sim_step <= 10
            #     current_mask = mask_computation(input_traj, trajectory, mode, sim_step)
                if sim_step > 1
            #         # Update the trajectory for each player
                    for player_id in 1:N
                        trajectory[player_id] = vcat(trajectory[player_id], initial_states[4 * (player_id - 1) + 1:4 * player_id])
                    end
                end            
            else
                trajectory = Dict(player_id => vcat(trajectory[player_id][5:end], reshape(initial_states[4 * (player_id - 1) + 1:4 * player_id], :)) for player_id in 1:N)
                input_traj = vec(vcat([reshape(trajectory[player_id], :) for player_id in 1:N]...))
            end
            current_mask = mask_computation(input_traj, trajectory, mode, sim_step)
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
            # Update initial_states for the next iteration in the desired format
            initial_states = vec(vcat([results["Player $player_id Latest Initial State"] for player_id in 1:N]...))
            for player_id in 1:N
                push!(receding_horizon_result["Player $player_id Trajectory"], results["Player $player_id Latest Initial State"])
                push!(receding_horizon_result["Player $player_id Control"], results["Player $player_id Latest Control"])
            end
            # println("Initial States: ", initial_states)
        end

        # results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/similarity_safety_scores_$mode.json"
        results_path = "$test_dir/receding_horizon_trajectories_[$scenario_id]_$mode.json"
        open(results_path, "w") do io
            JSON.print(io, receding_horizon_result)
        end
        println("Receding_horizon_trajectories saved to $results_path")
    end
end
