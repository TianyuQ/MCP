###############################################################################
# Required Packages and Custom Binary Mask Function
###############################################################################
using Flux, Zygote, BlockArrays, BSON, ProgressMeter

# Define a binary mask function that computes the hard threshold.
function binary_mask(x)
    return Int.(x .>= 0.5)
end

# Define a custom gradient (adjoint) for binary_mask using a surrogate derivative.
Zygote.@adjoint function binary_mask(x)
    y = binary_mask(x)  # Forward pass: hard threshold
    function pullback(Δ)
         # Use a surrogate derivative based on the sigmoid.
         σx = 1 ./(1 .+ exp.(-x))
         surrogate = σx .* (1 .- σx)
         return (Δ .* surrogate,)
    end
    return y, pullback
end

###############################################################################
# Load Dataset
###############################################################################
println("Loading dataset...")
directory = "/home/tq877/Tianyu/player_selection/MCP/data_bak/"
dataset = load_all_json_data(directory)  # Load all training data
println("Dataset loaded successfully. Total samples: ", length(dataset))

# Set batch size and initialize DataLoader
batch_size = 15
dataloader = DataLoader(dataset, batch_size)

###############################################################################
# Load Game
###############################################################################
(; environment) = setup_road_environment(; length = 7)
game = setup_trajectory_game(; environment, N = 4)
parametric_game = build_parametric_game(; game, horizon=10, params_per_player = 6)

###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global model = build_model()  # Declare `model` as global
# For our update we use a simple learning rate
learning_rate = 0.001
epochs = 2  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Training Loop with Progress Bar
###############################################################################
println("Starting training...")

for epoch in 1:epochs
    println("Epoch $epoch")
    # Reinitialize the dataloader each epoch to reshuffle indices.
    dataloader = DataLoader(dataset, batch_size)
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch Training Progress")
    cnt = 0

    for (batch_inputs, batch_targets, batch_indices) in dataloader
        global model
        θ, re = Flux.destructure(model)
        cnt += 1
        println("Processing batch: ", cnt)
        
        loss_fun = θ -> begin
            model_reconstructed = re(θ)
            pred_masks = model_reconstructed(batch_inputs)
            # Use the custom binary mask function with surrogate gradient.
            bin_mask = binary_mask(pred_masks)
            
            # Instead of using push! to accumulate results, use a comprehension:
            batch_computed_trajs = [ begin
                    idx = batch_indices[i]  # get the actual dataset index
                    block_sizes = fill(4, 4)  # for a 16-dim initial_states split into 4 blocks of 4
                    initial_states = BlockVector(dataset[idx][3], block_sizes)
                    goals = BlockVector(dataset[idx][4], fill(2, 4))  # for an 8-dim goals vector split into 4 blocks of 2
                    
                    traj = run_solver(game, bin_mask[:, i], initial_states, goals, 4, 10, 1)
                    println("Computed results for sample: ", idx)
                    # If run_solver returns multiple trajectories, combine them
                    traj = hcat(traj...)
                    traj  # The comprehension collects this value
                end for i in 1:batch_size ]
            
            # Combine the computed trajectories from all samples
            computed_trajs = vcat(batch_computed_trajs...)
            traj_error = mse(computed_trajs', batch_targets)
            mask_reg = sum(bin_mask)
            loss_val = traj_error + 0.1 * mask_reg
            return loss_val
        end

        # Use Zygote's reverse-mode gradient (which uses our custom adjoint)
        loss_val = loss_fun(θ)
        gradθ = Zygote.gradient(loss_fun, θ)[1]
        
        θ_new = θ - learning_rate * gradθ
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
