###############################################################################
# Set Random Seed for Reproducibility
###############################################################################
using Random
seed = 2
Random.seed!(seed)  # Set the seed to a fixed value

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global learning_rate = 0.01  # Learning rate for the optimizer
# Make sure to pass the required arguments (e.g. input_size, N) to build_model.
global model = build_model()  # Declare `model` as global
epochs = 150  # Number of training epochs
println("Model initialized successfully!")

global record_name = "bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed"

###############################################################################
# Import TensorBoardLogger for realtime logging
###############################################################################
using TensorBoardLogger

###############################################################################
# Initialize TensorBoard Logger
###############################################################################
tb_logger = TBLogger("logs/$record_name")  # Logs will be saved under logs/record_name

###############################################################################
# Training Loop with Mask Update and TensorBoard Logging
###############################################################################
println("Starting training...")
training_losses = Dict()

for epoch in 1:epochs
    println("Epoch $epoch")
    total_loss = 0.0
    epoch_masks = []
    epoch_gradients = []
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch Training Progress")
    
    for (batch_inputs, batch_targets, batch_initial_states, batch_goals, batch_indices) in dataloader
        # For neural network optimization, compute masks each batch
        current_masks = [model(batch_inputs[:, i]) for i in 1:batch_size] 
        
        # Ensure binary format for each mask
        pred_masks = [vcat([1], mask) for mask in current_masks]
        rounded_pred_masks = [round.(mask, digits=4) for mask in pred_masks]
        println("\nPred Masks: ", rounded_pred_masks)
        
        batch_loss = 0.0
        batch_grads = []

        # Loop over each example in the batch
        for i in 1:batch_size
            pred_mask = pred_masks[i]
            
            # Call the solver for each example
            results = run_solver(
                game, 
                parametric_game, 
                batch_targets[:, i], 
                batch_initial_states[:, i], 
                batch_goals[:, i], 
                N, 
                horizon, 
                num_sim_steps, 
                pred_mask
            )
            
            upstream_grads = results[1]
            upstream_grads = clamp.(upstream_grads, -10, 10)
            # println("Upstream Grads: ", upstream_grads)
            loss_val = results[2]
            batch_loss += mean(loss_val)
            
            _, back = Zygote.pullback(() -> model(batch_inputs[:, i]), Flux.params(model))
            grads_example = back(upstream_grads)
            push!(batch_grads, grads_example)
        end
        
        # Average the loss over the batch
        # batch_loss /= batch_size
        total_loss += batch_loss
        
        params_set = Flux.params(model)
        accum_grads = IdDict{Any, Any}()
        for p in params_set
            accum_grads[p] = zeros(size(p))
        end
        
        for grads_example in batch_grads
            for p in params_set
                accum_grads[p] .+= grads_example[p]
            end
        end
        
        # Compute the mean gradient for each parameter and update.
        for p in params_set
            mean_grad = accum_grads[p] ./ batch_size
            p .-= learning_rate * mean_grad
        end
        
        push!(epoch_masks, current_masks)
        push!(epoch_gradients, batch_grads)
        next!(progress)
    end

    training_losses[epoch] = total_loss
    println("\nEpoch $epoch: Loss = $total_loss")
    
    # Log the loss to TensorBoard for realtime visualization
    log_value(tb_logger, "loss", total_loss, step=epoch)

    # Additional mask update strategy can be implemented here if needed.
end

###############################################################################
# Save Trained Generative Model
###############################################################################
println("\nSaving trained generative model...")
# BSON.bson("trained_model_bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed.bson", Dict(:model => model))
BSON.bson("trained_model_$record_name.bson", Dict(:model => model))
println("Model saved successfully!")

println("\nSaving training loss records...")
open("training_losses_$record_name.json", "w") do f
    JSON.print(f, sort(collect(training_losses)), 4)
end
# println("Training loss saved successfully to training_losses_bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed.json!")
println("Training loss saved successfully to training_losses_$record_name.json!")

###############################################################################
# Close the TensorBoard Logger
###############################################################################
close(tb_logger)
