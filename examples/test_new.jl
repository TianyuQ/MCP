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
# record_name = "bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed"
best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/trained_model.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

similarity_score_list = []
safety_score_list = []

for (test_inputs, test_targets, test_initial_states, test_goals, test_indices) in test_dataloader
    current_masks = [best_model(test_inputs[:, i]) for i in 1:batch_size]
    println("\nPred Masks: ", [round.(mask, digits=4) for mask in current_masks])
    current_masks = [map(x -> x > 0.05 ? 1 : 0, mask) for mask in current_masks]
    pred_masks = [vcat([1], mask) for mask in current_masks]
    println("\nPred Masks: ", [round.(mask, digits=4) for mask in pred_masks])
    
    for i in 1:batch_size
        results = run_example(
            game = game,
            parametric_game = parametric_game,
            initial_states = test_initial_states[:, i],
            goals = test_goals[:, i],
            N = N,
            horizon = horizon,
            num_sim_steps = num_sim_steps,
            mask = pred_masks[i],
            target = test_targets[:, i],
            save = true
        )

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
results_path = "/home/tq877/Tianyu/player_selection/MCP/examples/logs/$record_name/similarity_safety_scores.json"
open(results_path, "w") do io
    JSON.print(io, Dict("similarity_scores" => similarity_score_list, "safety_scores" => safety_score_list))
end
println("Similarity and safety scores saved to $results_path")

