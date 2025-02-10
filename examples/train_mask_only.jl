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
global model = build_model()  # ✅ Declare `model` as global
opt = Flux.Adam(0.001)  
global state = Optimisers.setup(opt, model)  # ✅ Declare `state` as global
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
        # ✅ Compute Gradients and Loss Together
        loss_val, grads = Flux.withgradient(() -> begin
            # ✅ Forward Pass - Compute Predicted Masks
            pred_masks = model(batch_inputs)  # Get predicted masks
            # bin_mask = Int.(pred_masks .>= 0.5)  # Convert to binary (0s & 1s)
            # ✅ Compute Trajectories for Each Batch
            batch_computed_trajs = []
            for i in 1:batch_size
                println("Mask for batch ", i, ": ", pred_masks[:, i])
                traj = run_solver(pred_masks[:, i], dataset[i][3], dataset[i][4], 4, 10, 1)
                println("test")
                push!(batch_computed_trajs, traj)
            end
            # ✅ Convert `batch_computed_trajs` to a matrix correctly
            computed_trajs = vcat(batch_computed_trajs...)  
            # ✅ Compute Loss
            traj_error = mse(computed_trajs, batch_targets)  # Compute trajectory error
            mask_reg = sum(bin_mask)  # Regularization term
            loss_val = traj_error + 0.1 * mask_reg  # ✅ Ensure return is always `Float64`

            return loss_val
        end, Flux.params(model))  # ✅ Correctly compute gradients

        # ✅ Handle `nothing` case for gradients
        if grads === nothing
            println("Error: Gradients are `nothing`, skipping model update.")
            continue  # Skip this batch to avoid breaking the training loop
        end

        println("Computed Loss: ", loss_val, " | Type: ", typeof(loss_val))
        println("Gradients computed: ", grads)

        # ✅ Update Model Only if Gradients are Valid
        try
            global state = Optimisers.update!(state, model, grads)
        catch e
            println("Error: Failed to update model due to ", e)
            continue  # Skip this iteration if update fails
        end

        total_loss += loss_val  # ✅ No need for `.cpu` conversion
        next!(progress)
    end

    println("\nEpoch $epoch, Average Loss: $(total_loss / length(dataloader.dataset))")
end


###############################################################################
# Save Trained Model
###############################################################################
println("\nSaving trained model...")
BSON.bson("trained_model.bson", Dict(:model => model))  # ✅ No `.cpu`
println("Model saved successfully!")
