###############################################################################
# Set Random Seed for Reproducibility
###############################################################################
using Random
seed = 1
Random.seed!(seed)  # Set the seed to a fixed value

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global learning_rate = 0.001  # Learning rate for the optimizer
# Make sure to pass the required arguments (e.g. input_size, N) to build_model.
global model = build_model()  # Declare `model` as global
epochs = 30  # Number of training epochs
println("Model initialized successfully!")

global option = 1 # 0 for numerical optimization, 1 for neural network optimization

# Initial mask for numerical optimization
initial_mask = [0.8, 0.8, 0.8]

# For numerical optimization, initialize the masks outside the training loop.
if option == 0
    global current_masks = [initial_mask for _ in 1:batch_size]
end

###############################################################################
# Training Loop with Mask Update
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
        if option == 1
            current_masks = [model(batch_inputs[:, i]) for i in 1:batch_size]
        end
        
        # Ensure binary format for each mask
        pred_masks = [vcat([1], mask) for mask in current_masks]
        println("\nPred Masks: ", pred_masks)
        
        batch_loss = 0.0
        # For option==1, we accumulate gradients (returned as dictionaries) for each example.
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
            
            
            if option == 1
                # Compute the vector-Jacobian product with respect to model parameters.
                # Here we use pullback on a closure that computes the model output for the given input.
                _, back = Zygote.pullback(() -> model(batch_inputs[:, i]), Flux.params(model))
                grads_example = back(upstream_grads)
                push!(batch_grads, grads_example)
            end
            
            if option == 0
                # For numerical optimization, use upstream_grads directly.
                push!(batch_grads, upstream_grads)
            end
        end
        
        # Average the loss over the batch
        # batch_loss /= batch_size
        total_loss += batch_loss
        
        if option == 1
            # Accumulate gradients over the batch.
            # We iterate over the model parameters (from Flux.params(model)) and sum the corresponding gradients.
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
        end
        
        if option == 0
            # Declare that we are updating the global variable `current_masks`
            global current_masks
            # Update the mask for numerical optimization.
            current_masks = [current_masks[i] .- learning_rate * batch_grads[i] for i in 1:batch_size]
        end
        
        push!(epoch_masks, current_masks)
        push!(epoch_gradients, batch_grads)
        next!(progress)
    end

    training_losses[epoch] = total_loss
    println("\nEpoch $epoch: Loss = $total_loss")
    
    # Additional mask update strategy can be implemented here if needed.
end

###############################################################################
# Save Trained Generative Model
###############################################################################
println("\nSaving trained generative model...")
BSON.bson("trained_generative_model_bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed.bson", Dict(:model => model))
println("Model saved successfully!")

println("\nSaving training loss records...")
open("training_losses_bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed.json", "w") do f
    JSON.print(f, sort(collect(training_losses)), 4)
end
println("Training loss saved successfully to training_losses_bs_$batch_size _ep_$epochs _lr_$learning_rate _sd_$seed.json!")
