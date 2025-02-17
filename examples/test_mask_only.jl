###############################################################################
# Load Trained Model
###############################################################################
println("Loading trained model...")

# Load model from file
model_file = "trained_generative_model_bs_15 ep_5.bson"  # Ensure it matches training settings
loaded_data = BSON.load(model_file)
model = loaded_data[:model]  

println("Model loaded successfully!")

###############################################################################
# Testing the Model on Entire Dataset
###############################################################################
function test_model(dataset, model)
    total_loss = 0.0
    num_samples = length(dataset)

    progress = Progress(num_samples, desc="Testing dataset")

    for (sample_idx, data) in enumerate(dataset)
        trajectories, ego_index, initial_states, goals = data

        # Prepare input for the model
        input_vec = prepare_input(trajectories, ego_index)

        # Ensure correct data type
        input_vec = Float64.(input_vec)

        # Predict mask
        predicted_mask = model(input_vec)

        predicted_mask = vec(predicted_mask)

        # Convert predicted mask to binary (threshold at 0.5)
        bin_mask = Float64.(predicted_mask .>= 0.5)

        println(initial_states)
        println(goals)
        # println(bin_mask)
        println(predicted_mask)

        # Run solver with predicted mask
        computed_traj = run_solver(
            game,
            parametric_game, 
            predicted_mask,  
            initial_states, 
            goals, 
            4, 30, 1  # Ensure time horizon matches training
        )

        # Compute MSE loss
        mse_loss = mse(computed_traj, ground_truth_traj)
        total_loss += mse_loss

        # Print sample-wise results every 100 samples
        # if sample_idx % 100 == 0
        println("\n===== Test Sample $sample_idx =====")
        println("Ego-Agent Index: $ego_index")
        println("\n🔹 Predicted Mask:")
        println(predicted_mask |> cpu)  # Move to CPU if necessary
        println("\n🔹 Binary Mask (Thresholded at 0.5):")
        println(bin_mask)
        println
        println("\n🔹 Mean Squared Error (MSE): $mse_loss")
        # end

        # Update progress bar
        next!(progress)
    end

    # Compute average loss
    avg_loss = total_loss / num_samples
    return avg_loss
end

###############################################################################
# Run Testing on Entire Dataset
###############################################################################
println("\nLoading test dataset...")
test_dataset = load_all_json_data("/home/tq877/Tianyu/player_selection/MCP/data/")
println("Test dataset loaded successfully! Total samples: ", length(test_dataset))

# Run the test on the entire dataset
avg_test_loss = test_model(test_dataset, model)

# Display final result
println("\n🚀 Testing complete. Average MSE loss over entire dataset: $avg_test_loss")
