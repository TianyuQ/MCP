###############################################################################
# Load Trained Model
###############################################################################
println("Loading trained model...")
loaded_data = BSON.load("trained_model.bson")
model = loaded_data[:model]
println("Model loaded successfully!")

###############################################################################
# Testing the Model
###############################################################################
"""
    test_model(dataset)

Runs a single test sample from the dataset to evaluate the model's performance.

Outputs:
- Ground truth mask
- Predicted mask
- Ground truth trajectory
- Computed trajectory from solver
"""
function test_model(dataset)
    # Select a random test sample
    sample_idx = rand(1:length(dataset))
    trajectories, ego_index, initial_states, goals, ground_truth_traj = dataset[sample_idx]

    # Prepare input for the model
    input_vec = prepare_input(trajectories, ego_index)

    # Predict mask using the trained model
    predicted_mask = model(input_vec)

    # Convert predicted mask to binary (threshold at 0.5)
    bin_mask = Float32.(predicted_mask .>= 0.5)

    # Run solver with predicted mask
    computed_traj = run_solver(bin_mask, initial_states, goals, N)

    # Display Results
    println("\n===== Test Results =====")
    println("Ego-Agent Index: $ego_index")
    println("Predicted Mask (values near 1 indicate importance):")
    println(predicted_mask)
    println("Binary Mask (Thresholded at 0.5):")
    println(bin_mask)
    println("Ground Truth Trajectory:")
    println(ground_truth_traj)
    println("Computed Trajectory from Solver:")
    println(computed_traj)

    # Compute and print MSE loss
    mse_loss = mse(computed_traj, ground_truth_traj)
    println("Mean Squared Error (MSE): $mse_loss")

    return mse_loss
end

###############################################################################
# Run Model Testing
###############################################################################
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_all_json_data(directory)

# Run test
test_loss = test_model(dataset)
println("Testing complete. Final MSE loss: $test_loss")
