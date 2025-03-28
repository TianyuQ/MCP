###############################################################################
# Test Code: Load the Best Model and Evaluate on Test Data
###############################################################################
using BSON
using Flux

###############################################################################
# Evaluation Code
###############################################################################

function similarity_analysis(results, target)
    diff = sum(norm(results[j:j+1] - target[j:j+1]) for j in (horizon-input_horizon)*d + 1 : d : horizon*d)
    return diff
end

function safety_analysis(results, target)
    safeness = minimum(norm(results[j:j+1] - target[(player_id-1) * horizon * d + j : (player_id-1) * horizon * d + j + 1]) for j in (horizon-input_horizon) * d + 1 : d : horizon * d for player_id in 2:N)
    return safeness
end

# Ensure that variables such as batch_size, N, horizon, num_sim_steps, game, 
# parametric_game, and test_dataloader are defined or imported in this context.
# You may need to include a common configuration file if these are defined elsewhere.

println("\nLoading best model for testing...")
# Use the same record_name as in training
record_name = "bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed"
best_model_data = BSON.load("trained_model_bs_16 _ep_150 _lr_0.01 _sd_2.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

similarity_score_list = []
safety_score_list = []

for (test_inputs, test_targets, test_initial_states, test_goals, test_indices) in test_dataloader
    for i in 1:batch_size
        initial_states = test_initial_states[:, i]
        input_traj = test_inputs[:, i]
        for sim_step in 1:50
            if sim_step <= 5
                current_mask = ones(N-1)
            else
                current_mask = best_model(input_traj)
            end
            pred_mask = vcat([1], current_masks)
            println("\nPred Masks: ", round.(pred_mask, digits=4))
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
            initial_states = results["Initial State"] # Update initial states for the next iteration
            input_traj = results["Player 1 Trajectory"] # Update input trajectory for the next iteration

        end

        converted_values = [Float64.(vcat(x...)) for x in results["Player 1 Trajectory"]]  # Flatten each sublist
        data_matrix = hcat(converted_values...)'  # Stack into matrix and transpose to (T, d)
        trajectories = reshape(data_matrix, (horizon, d))  # Ensure correct shape
        results = vec(trajectories)

        similarity_score = similarity_analysis(results, test_targets[:, i])
        safety_score = safety_analysis(results, test_targets[:, i])
        push!(similarity_score_list, similarity_score)
        push!(safety_score_list, safety_score)
    end
end

# Save the results to a file
results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/similarity_safety_scores.json"
open(results_path, "w") do io
    JSON.print(io, Dict("similarity_scores" => similarity_score_list, "safety_scores" => safety_score_list))
end
println("Similarity and safety scores saved to $results_path")