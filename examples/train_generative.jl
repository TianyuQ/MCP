###############################################################################
# Required Packages and Helper Functions
###############################################################################
using Flux, BlockArrays, BSON, ProgressMeter, Distributions, JSON

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
# Initialize Generative Model & Optimizer
###############################################################################
println("Initializing generative model...")
global model = build_model()  # Declare `model` as global
optimizer = Flux.Adam(0.001)  # Use Adam optimizer
epochs = 50  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Define Bernoulli Loss Function
###############################################################################
function loss_fun(model, batch_inputs, batch_targets, pred_probs, computed_trajs)
    traj_error = mse(computed_trajs, batch_targets)

    # Bernoulli loss (regularization)
    mask_loss = sum(pred_probs)

    # return 1000 * (traj_error + 0.001 * bernoulli_loss)
    return traj_error + 0.001 * mask_loss
end

###############################################################################
# Bernoulli Sampling Function
###############################################################################
function sample_bernoulli(probabilities)
    return Float64.(rand.(Bernoulli.(probabilities)))  # Convert samples to binary values
end

###############################################################################
# Training Loop
###############################################################################
println("Starting training...")

flat_model, reassemble = Flux.destructure(model)  # Extract model parameters as a single vector
optim_state = Flux.setup(Flux.Adam(0.001), model)

training_losses = Dict()  # Dictionary to store losses per epoch

for epoch in 1:epochs
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch ")

    for (batch_inputs, batch_targets, batch_indices) in dataloader
        # 🟢 **Compute `pred_masks` from the model**
        pred_probs = model(batch_inputs)  # Predict Bernoulli probabilities
        pred_masks = sample_bernoulli(pred_probs)  # Sample binary masks
        # 🛑 **Use `pred_masks` inside `run_solver`**
        batch_computed_trajs = [run_solver(
            game, 
            pred_masks[:, i],  # ✅ Use predicted masks, not random ones
            deepcopy(BlockVector(dataset[idx][3], fill(4, 4))), 
            deepcopy(BlockVector(dataset[idx][4], fill(2, 4))), 
            4, 10, 1
        ) for (i, idx) in enumerate(batch_indices)]

        computed_trajs = hcat(batch_computed_trajs...)  # Ensure (160, batch_size)

        # 🟢 Define loss function that does **not** call `run_solver`
        loss_function = θ -> loss_fun(reassemble(θ), batch_inputs, batch_targets, pred_probs, computed_trajs)

        # Compute loss and gradients using ForwardDiff
        loss_val = loss_function(flat_model)  # ✅ Calls loss_fun **once**
        grads = Zygote.gradient(loss_function, flat_model)  # ✅ No multiple calls to run_solver
        println(grads)
        # 🟢 Properly update model parameters
        Flux.update!(optim_state, flat_model, grads)  # ✅ FIXED

        total_loss += loss_val
    end
    next!(progress)

    avg_loss = total_loss / length(dataloader.dataset)
    training_losses[string(epoch)] = round(avg_loss, digits=6)  # Store epoch loss
    println("Epoch $epoch, Average Loss: ", training_losses[string(epoch)])
end

# Reconstruct the trained model with updated parameters
model = reassemble(flat_model)

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
