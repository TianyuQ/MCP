###############################################################################
# Required Packages and Helper Functions
###############################################################################
using Flux, BlockArrays, BSON, ProgressMeter, Zygote, Optimisers, Statistics

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global model = build_model()  # Declare `model` as global
opt_state = Flux.setup(Flux.Adam(learning_rate), Flux.trainable(model))
epochs = 40  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Initialize Mask
###############################################################################
# Initialize the mask outside the training loop
global mask = ones(Float32, N-1)  # Assuming N is the length of the mask
println("Initial mask: ", mask)

###############################################################################
# Training Loop with Mask Update
###############################################################################
println("Starting training...")
training_losses = Dict()
for epoch in 1:epochs
    println("Epoch $epoch")
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch Training Progress")

    for (batch_inputs, batch_targets, batch_initial_states, batch_goals, batch_indices) in dataloader
        global model  # Declare global if needed
        
        # Use the current mask for the forward pass
        pred_mask = mask  # Start with the mask from the previous epoch
        pred_mask = vcat([1], pred_mask)
        
        results = run_solver(
            game, 
            parametric_game, 
            batch_targets[:,1], 
            batch_initial_states[:,1], 
            batch_goals[:,1], 
            N, 
            horizon, 
            num_sim_steps, 
            pred_mask
        )
        
        upstream_grads = results[1]
        loss_val = results[2]
        println("Loss Values: ", loss_val)
        total_loss += mean(loss_val)

        jacobian = Flux.jacobian(x -> model(x), batch_inputs)
        gradient = transpose(upstream_grads) * jacobian[1]

        # ---------------------------------------------------------------------
        # Update the model parameters using Flux's optimizer.
        # ---------------------------------------------------------------------
        params = Flux.params(model)
        for (p, g) in zip(params, gradient)
            p .-= learning_rate * g
        end
        global mask = model(batch_inputs[:,1])  # Update mask for the next epoch
        next!(progress)
        
    end

    training_losses[epoch] = total_loss
    println("Epoch $epoch: Loss = $total_loss")
    
    ###############################################################################
    # Mask Update Strategy
    ###############################################################################
    # Update the mask based on the loss or other criteria
    # Example: Update mask as a function of model predictions or gradients
    
    # mask = vcat([1], mask)  # Ensure mask shape consistency
    # println("Updated Mask for Next Epoch: ", mask)
end

###############################################################################
# Save Trained Generative Model
###############################################################################
println("\nSaving trained generative model...")
BSON.bson("trained_generative_model_bs_$batch_size _ep_$epochs _lr_$learning_rate.bson", Dict(:model => model))  # Save model
println("Model saved successfully!")

println("\nSaving training loss records...")
open("training_losses_bs_$batch_size _ep_$epochs _lr_$learning_rate.json", "w") do f
    JSON.print(f, sort(collect(training_losses)), 4)  # Sort by epoch and pretty print with indentation of 4 spaces
end
println("Training loss saved successfully to training_losses_bs_$batch_size _ep_$epochs _lr_$learning_rate.json!")
