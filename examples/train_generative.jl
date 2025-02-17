using Flux, BlockArrays, BSON, ProgressMeter, Distributions, JSON

###############################################################################
# Initialize Generative Model & Optimizer
###############################################################################
println("Initializing generative model...")
global model = build_model()  # Declare `model` as global
epochs = 5  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Define Bernoulli Loss Function (Fully Differentiable)
###############################################################################
function loss_fun(batch_targets, pred_probs, computed_trajs)
    traj_error = mse(computed_trajs, batch_targets)  # ✅ Fully differentiable
    mask_loss = sum(pred_probs)  # ✅ Fully differentiable
    # return 100 * traj_error  # ✅ Gradient propagates correctly
    return 100 * traj_error + mask_loss # ✅ Gradient propagates correctly
end

###############################################################################
# Bernoulli Sampling Function (Non-Differentiable)
###############################################################################
function sample_bernoulli(probabilities)
    return Float64.(rand.(Bernoulli.(probabilities)))  # 🚫 Non-differentiable (Do not use inside loss)
end

###############################################################################
# **Backpropagate `grads` to Model Parameters**
###############################################################################
###############################################################################
# **Backpropagate `grads` to Model Parameters (Manual Backpropagation)**
###############################################################################
function backpropagate_through_model!(model, grads, batch_inputs, lr=0.001)
    # Dictionary to store gradients of each parameter
    param_grads = Dict(p => zeros(size(p)) for p in Flux.params(model))

    # Forward pass to store activations
    activations = [batch_inputs]  # Store input as first activation
    x = batch_inputs
    for layer in model.layers
        x = layer(x)
        push!(activations, x)
    end

    # Initialize gradient of output layer (grads is dL/dy_hat)
    delta = grads  # This is ∂L/∂ŷ, already computed from loss

    # Backpropagate manually through dense layers
    for layer_idx in length(model.layers):-1:1
        layer = model[layer_idx]

        if isa(layer, Dense)
            input_activation = activations[layer_idx]  # Activation from previous layer

            # Compute weight and bias gradients
            weight_grad = delta * input_activation'  # dL/dW
            bias_grad = sum(delta, dims=2)  # dL/db

            # Store computed gradients
            param_grads[layer.weight] .= weight_grad
            param_grads[layer.bias] .= bias_grad

            # Compute delta for previous layer (dL/dA_prev)
            delta = (layer.weight' * delta) .* (input_activation .> 0)  # Apply ReLU derivative

        end
    end

    # Apply gradient updates (Gradient Descent)
    for (p, grad) in param_grads
        p .-= lr .* grad  # Update parameter using learning rate
    end
end

###############################################################################
# Training Loop
###############################################################################
###############################################################################
# Training Loop
###############################################################################
println("Starting training...")

training_losses = Dict()  # Store losses per epoch
learning_rate = 0.001  # Learning rate for manual updates

for epoch in 1:epochs
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch ")

    for (batch_inputs, batch_targets, batch_indices) in dataloader
        # ✅ Compute `pred_probs` from the model
        pred_probs = model(batch_inputs)  # Model forward pass

        println("Predicted Probabilities: ", pred_probs)  # Debugging

        # ✅ Compute `computed_trajs` using `pred_probs`
        batch_computed_trajs = [run_solver(
            game,
            parametric_game, 
            pred_probs[:, i],  
            deepcopy(BlockVector(dataset[idx][3], fill(4, 4))), 
            deepcopy(BlockVector(dataset[idx][4], fill(2, 4))), 
            4, 30, 1
        ) for (i, idx) in enumerate(batch_indices)]
        
        computed_trajs = hcat(batch_computed_trajs...)  # Ensure (160, batch_size)

        # ✅ Compute loss
        loss = loss_fun(batch_targets, pred_probs, computed_trajs)

        # ✅ Compute Gradients (Finite Difference Approximation)
        grads = zeros(size(pred_probs))
        for k in 1:size(pred_probs, 2)
            for j in 1:4
                pred_probs[j, k] += 1e-2  # Perturbation
                batch_computed_trajs_perturb = [run_solver(
                    game,
                    parametric_game, 
                    pred_probs[:, i],  
                    deepcopy(BlockVector(dataset[idx][3], fill(4, 4))), 
                    deepcopy(BlockVector(dataset[idx][4], fill(2, 4))), 
                    4, 30, 1
                ) for (i, idx) in enumerate(batch_indices)]
                computed_trajs_perturb = hcat(batch_computed_trajs_perturb...)
                loss_perturb = loss_fun(batch_targets, pred_probs, computed_trajs_perturb)
                grads[j, k] = (loss_perturb - loss) / 1e-2  # Finite difference gradient
            end
        end

        # println("Gradient: ", grads)  # Debugging

        # ✅ Backpropagate gradients through model
        backpropagate_through_model!(model, grads, batch_inputs, learning_rate)

        total_loss += loss  # Accumulate loss
    end
    next!(progress)

    avg_loss = total_loss
    training_losses[string(epoch)] = round(avg_loss, digits=6)  # Store epoch loss
    println("Epoch $epoch, Average Loss: ", training_losses[string(epoch)])
end

###############################################################################
# Save Trained Generative Model
###############################################################################
println("\nSaving trained generative model...")
BSON.bson("trained_generative_model_bs_$batch_size ep_$epochs.bson", Dict(:model => model))  # Save model
println("Model saved successfully!")

println("\nSaving training loss records...")
open("training_losses.json", "w") do f
    JSON.print(f, training_losses, 4)  # Pretty print with indentation of 4 spaces
end
println("Training loss saved successfully to training_losses.json!")
