###############################################################################
# Test Code: Load the Best Model and Evaluate on Test Data
###############################################################################
using BSON
using Flux

# Ensure that variables such as batch_size, N, horizon, num_sim_steps, game, 
# parametric_game, and test_dataloader are defined or imported in this context.
# You may need to include a common configuration file if these are defined elsewhere.

println("\nLoading best model for testing...")
# Use the same record_name as in training
record_name = "bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed"
best_model_data = BSON.load("/home/tq877/Tianyu/player_selection/MCP/examples/trained_model_$record_name.bson")
best_model = best_model_data[:model]
println("Best model loaded successfully!")

global test_loss = 0.0
for (test_inputs, test_targets, test_initial_states, test_goals, test_indices) in test_dataloader
    current_masks = [best_model(test_inputs[:, i]) for i in 1:batch_size]
    pred_masks = [vcat([1], mask) for mask in current_masks]
    println("\nPred Masks: ", [round.(mask, digits=4) for mask in pred_masks])
    
    for i in 1:batch_size
        results = run_solver(
            game,
            parametric_game,
            test_targets[:, i],
            test_initial_states[:, i],
            test_goals[:, i],
            N, horizon, num_sim_steps,
            pred_masks[i]
        )
        loss_val = results[2]
        global test_loss
        test_loss += mean(loss_val)
    end
end

test_loss /= test_batches
println("Test Loss: $test_loss")
