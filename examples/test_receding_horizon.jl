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

function mask_computation(input_traj, mode)
    if mode == "All"
        mask = ones(N-1)
    elseif mode == "Neural Network"
        mask = best_model(input_traj)
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
    for (test_inputs, test_targets, test_initial_states, test_goals, test_indices) in test_dataloader
        for i in 1:batch_size
            initial_states = test_initial_states[:, i]
            input_traj = test_inputs[:, i]
            trajectory = Dict{Int64, Vector{Float64}}()
            for player_id in 1:N
                trajectory[player_id] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
            end
            for sim_step in 1:50
                println("Sim Step: ", sim_step)
                if sim_step <= 10
                    current_mask = mask_computation(input_traj, "All")
                    if sim_step > 1
                        # Update the trajectory for each player
                        for player_id in 1:N
                            trajectory[player_id] = vcat(trajectory[player_id], initial_states[4 * (player_id - 1) + 1:4 * player_id])
                        end
                    end            
                else
                    trajectory = Dict(player_id => vcat(trajectory[player_id][5:end], reshape(initial_states[4 * (player_id - 1) + 1:4 * player_id], :)) for player_id in 1:N)
                    current_mask = mask_computation(input_traj, mode)
                end
                pred_mask = vcat([1], current_mask)
                # println("Pred Masks: ", round.(pred_mask, digits=4))
                results = run_example(
                    game = game,
                    parametric_game = parametric_game,
                    initial_states = initial_states,
                    goals = test_goals[:, i],
                    N = N,
                    horizon = horizon,
                    num_sim_steps = num_sim_steps,
                    mask = pred_mask,
                    target = test_targets[:, i],
                    save = true
                )
                # Update initial_states for the next iteration in the desired format
                initial_states = vec(vcat([results["Player $player_id Latest Initial State"] for player_id in 1:N]...))
                println("Initial States: ", initial_states)
            end

            converted_values = [Float64.(vcat(x...)) for x in results["Player 1 Trajectory"]]  # Flatten each sublist
            data_matrix = hcat(converted_values...)'  # Stack into matrix and transpose to (T, d)
            trajectories = reshape(data_matrix, (horizon, d))  # Ensure correct shape
            results = vec(trajectories)

        end
    end

    # Save the results to a file
    # results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/similarity_safety_scores_$mode.json"
    results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/logs/bs_16 _ep_100 _lr_0.01 _sd_3 _pat_100 _N_4 _h_30 _ih10 _isd_4/similarity_safety_scores_$mode.json"
    open(results_path, "w") do io
        JSON.print(io, Dict("similarity_scores" => similarity_score_list, "safety_scores" => safety_score_list))
    end
    println("Similarity and safety scores saved to $results_path")
end