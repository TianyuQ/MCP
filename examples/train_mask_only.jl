###############################################################################
# Required Packages and Helper Functions
###############################################################################
using Flux, BlockArrays, BSON, ProgressMeter

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
optimizer = Flux.Adam(0.001)  # Use Adam optimizer
epochs = 2  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Define Loss Function
###############################################################################
function loss_fun(batch_inputs, batch_targets, batch_indices)
    pred_masks = model(batch_inputs)  # Get predictions

    batch_computed_trajs = map(1:batch_size) do i
        idx = batch_indices[i]  # Get actual dataset index
        block_sizes = fill(4, 4)  # 16-dim initial_states split into 4 blocks
        initial_states = BlockVector(dataset[idx][3], block_sizes)
        goals = BlockVector(dataset[idx][4], fill(2, 4))  # 8-dim goals vector split into 4 blocks

        traj = run_solver(game, pred_masks[:, i], deepcopy(initial_states), deepcopy(goals), 4, 10, 1)
        hcat(traj...)  # Ensure correct format
    end

    computed_trajs = vcat(batch_computed_trajs...)
    traj_error = mse(computed_trajs', batch_targets)
    mask_reg = sum(pred_masks)  # Regularization term
    return traj_error + 0.1 * mask_reg
end

###############################################################################
# Training Loop with Flux Optimizer
###############################################################################
println("Starting training...")

for epoch in 1:epochs
    println("Epoch $epoch")
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch Training Progress")

    for (batch_inputs, batch_targets, batch_indices) in dataloader
        # Compute loss and gradients using Flux.train!
        loss_val, grads = Flux.withgradient(() -> loss_fun(batch_inputs, batch_targets, batch_indices), Flux.params(model))

        # Update model parameters
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)

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
