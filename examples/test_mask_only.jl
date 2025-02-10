###############################################################################
# Load Trained Model
###############################################################################
println("Loading trained model...")
loaded_data = BSON.load("trained_model.bson")
model = loaded_data[:model] |> device  # Move model to GPU if available
println("Model loaded successfully!")

###############################################################################
# Testing the Model
###############################################################################
function test_model(dataset, model)
    # Select a random test sample
    sample_idx = rand(1:length(dataset))
    trajectories, ego_index, initial_states, goals, ground_truth_traj = dataset[sample_idx]

    # Prepare input for the model
    input_vec = prepare_input(trajectories, ego_index) |> device

    # Predict mask
    predicted_mask = model(input_vec)

    # Convert predicted mask to binary
    bin_mask = Float64.(predicted_mask .>= 0.5) |> cpu

    # Run solver with predicted mask
    computed_traj = run_solver(bin_mask, initial_states, goals, N) |> cpu

    # Compute MSE loss
    mse_loss = mse(computed_traj, ground_truth_traj)

    # Print Results
    println("\n===== Test Results =====")
    println("Ego-Agent Index: $ego_index")
    println("Predicted Mask:")
    println(predicted_mask |> cpu)
    println("Binary Mask (Thresholded at 0.5):")
    println(bin_mask)
    println("Ground Truth Trajectory:")
    println(ground_truth_traj)
    println("Computed Trajectory:")
    println(computed_traj)
    println("Mean Squared Error (MSE): $mse_loss")

    return mse_loss
end

###############################################################################
# Run Testing
###############################################################################
dataset = load_all_json_data("/home/tq877/Tianyu/player_selection/MCP/data_bak/")
test_loss = test_model(dataset, model)
println("Testing complete. Final MSE loss: $test_loss")
