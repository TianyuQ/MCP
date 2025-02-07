###############################################################################
# Load Dataset
###############################################################################
println("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_all_json_data(directory)  # Load all training data
println("Dataset loaded successfully. Total samples: ", length(dataset))

# Set batch size and initialize DataLoader
batch_size = 10
dataloader = DataLoader(dataset, batch_size)

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
model = build_model()  # Build and move model to GPU if available
opt = ADAM(0.001)  # Adam optimizer
epochs = 20  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Training Loop
###############################################################################
println("\nStarting training...")
for epoch in 1:epochs
    total_loss = 0.0  # Track loss per epoch

    for (batch_inputs, batch_targets) in dataloader
        # Compute loss and gradients
        loss_val, grads = Flux.withgradient(params(model)) do
            pred_masks = model(batch_inputs)  # Get predicted masks
            bin_mask = Float32.(pred_masks .>= 0.5) |> cpu  # Convert to binary
            computed_traj = run_solver(bin_mask, dataset[1][3], dataset[1][4], N)  # Compute trajectory
            traj_error = mse(computed_traj |> device, batch_targets)  # Compute trajectory error
            mask_reg = sum(pred_masks)  # Regularization term
            traj_error + λ * mask_reg  # Total loss
        end

        # Apply gradients to update model
        Flux.Optimise.update!(opt, params(model), grads)

        # Accumulate loss
        total_loss += loss_val |> cpu  # Move loss to CPU for printing
    end

    # Print average loss per epoch
    println("Epoch $epoch, Average Loss: $(total_loss / length(dataloader.dataset))")
end

###############################################################################
# Save Trained Model
###############################################################################
println("\nSaving trained model...")
BSON.bson("trained_model.bson", Dict(:model => model |> cpu))  # Save model on CPU
println("Model saved successfully!")
