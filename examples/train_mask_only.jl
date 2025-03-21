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
global learning_rate = 0.1  # Learning rate for the optimizer
global model = build_model()  # Declare `model` as global
opt_state = Flux.setup(Flux.Adam(learning_rate), Flux.trainable(model))
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
            # upstream_grads = clamp.(upstream_grads, -5, 5)
            println("Upstream Grads: ", upstream_grads)
            loss_val = results[2]
            batch_loss += mean(loss_val)
            
            if option == 1
                # Calculate gradients for each example using model's jacobian
                jacobian = Flux.jacobian(x -> model(x), batch_inputs[:, i])
                gradient = transpose(upstream_grads) * jacobian[1]
                gradient = clamp.(gradient, -10, 10)
                push!(batch_grads, gradient)
            end
            
            if option == 0
                # For numerical optimization, use upstream_grads directly.
                push!(batch_grads, upstream_grads)
            end
        end
        
        # Average the loss over the batch
        batch_loss /= batch_size
        total_loss += batch_loss
        
        # Update gradients or masks based on optimization mode
        mean_grads = mean(batch_grads, dims=1)  # Average gradients over the batch
        
        if option == 1
            params = Flux.params(model)
            for (p, g) in zip(params, mean_grads)
                p .-= learning_rate * g
            end
        end
        
        if option == 0
            # Declare that we are updating the global variable `current_masks`
            global current_masks
            # Update the mask for numerical optimization
            current_masks = [current_masks[i] .- learning_rate * batch_grads[i] for i in 1:batch_size]
        end
        
        push!(epoch_masks, current_masks)
        push!(epoch_gradients, mean_grads)
        next!(progress)
    end

    training_losses[epoch] = total_loss
    println("Epoch $epoch: Loss = $total_loss")
    
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
