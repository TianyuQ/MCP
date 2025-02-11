###############################################################################
# Load Dataset
###############################################################################
println("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_all_json_data(directory)  # Load all training data
println("Dataset loaded successfully. Total samples: ", length(dataset))

# Set batch size and initialize DataLoader
batch_size = 1
dataloader = DataLoader(dataset, batch_size)

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global model = build_model()  # Declare `model` as global
# For forwarddiff-based update, we use a simple learning rate
learning_rate = 0.001
epochs = 2  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Training Loop with Progress Bar
###############################################################################
println("Starting training...")

for epoch in 1:epochs
    println("Epoch $epoch")
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch Training Progress")

    for (batch_inputs, batch_targets) in dataloader
        # Ensure we reference the global model
        global model

        # ---- Flatten the Model Parameters ----
        # Flux.destructure converts your model’s parameters into a single vector,
        # and returns a function that can rebuild the model from such a vector.
        θ, re = Flux.destructure(model)
        
        # ---- Define the Loss Function in Terms of θ ----
        loss_fun = θ -> begin
            # Reconstruct the model from θ.
            model_reconstructed = re(θ)
            # Forward pass: compute predicted masks.
            pred_masks = model_reconstructed(batch_inputs)
            # Convert predictions to binary masks.
            bin_mask = Int.(pred_masks .>= 0.5)
            
            # Compute trajectories for each example in the batch.
            batch_computed_trajs = []
            for i in 1:batch_size
                # println("Mask for batch ", i, ": ", pred_masks[:, i])
                block_sizes = fill(4, 4)
                initial_states = BlockVector(dataset[i][3], block_sizes)
                goals = BlockVector(dataset[i][4], fill(2, 4))
                traj = run_solver(bin_mask[:, i], initial_states, goals, 4, 10, 1)
                println("Computed results!")
                # Combine the trajectories for all players
                traj = hcat(traj...)
                push!(batch_computed_trajs, traj)
            end
            # Combine the trajectories into one matrix.
            computed_trajs = hcat(batch_computed_trajs...)
            # Compute loss components.
            traj_error = mse(computed_trajs', batch_targets)
            mask_reg = sum(bin_mask)
            loss_val = traj_error + 0.1 * mask_reg
            return loss_val
        end

        # ---- Compute Loss and Its Gradient Using ForwardDiff ----
        # Zygote.forwarddiff takes a vector input and runs a block in forward-mode.
        # It returns both the loss value and the gradient (as a vector with the same size as θ).
        loss_val = loss_fun(θ)
        gradθ = ForwardDiff.gradient(loss_fun, θ)
        
        # ---- Update the Parameters ----
        # Perform a simple gradient descent step.
        θ_new = θ - learning_rate * gradθ
        # Rebuild the model from the updated parameter vector.
        global model = re(θ_new)
        
        total_loss += loss_val
        next!(progress)
    end

    avg_loss = total_loss / length(dataloader.dataset)
    println("\nEpoch $epoch, Average Loss: $avg_loss")
end

###############################################################################
# Save Trained Model
###############################################################################
println("\nSaving trained model...")
BSON.bson("trained_model.bson", Dict(:model => model))  # Save the updated model
println("Model saved successfully!")
