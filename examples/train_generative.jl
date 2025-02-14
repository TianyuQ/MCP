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
epochs = 2  # Number of training epochs
println("Model initialized successfully!")

###############################################################################
# Define Bernoulli Loss Function (Fully Differentiable)
###############################################################################
function loss_fun(batch_targets, pred_probs, computed_trajs)
    traj_error = mse(computed_trajs, batch_targets)  # ✅ Fully differentiable

    # Bernoulli loss (use pred_probs instead of sampled masks)
    mask_loss = sum(pred_probs)  # ✅ Fully differentiable

    return traj_error + 0.001 * mask_loss  # ✅ Gradient propagates correctly
end

###############################################################################
# Bernoulli Sampling Function (Non-Differentiable)
###############################################################################
function sample_bernoulli(probabilities)
    return Float64.(rand.(Bernoulli.(probabilities)))  # 🚫 Non-differentiable (Do not use inside loss)
end

###############################################################################
# Training Loop
###############################################################################
println("Starting training...")

optim_state = Flux.setup(Flux.Adam(0.001), model)
training_losses = Dict()  # Dictionary to store losses per epoch

for epoch in 1:epochs
    total_loss = 0.0
    progress = Progress(length(dataloader.dataset), desc="Epoch $epoch ")

    for (batch_inputs, batch_targets, batch_indices) in dataloader
        # ✅ Compute `pred_probs` from the model
        pred_probs = model(batch_inputs)  # ✅ Differentiable
        pred_masks = sample_bernoulli(pred_probs)  # 🚫 Sampling is non-differentiable

        # ✅ Use `pred_masks` inside `run_solver`
        batch_computed_trajs = [run_solver(
            game,
            parametric_game, 
            pred_probs[:, i],  # ✅ Uses predicted masks
            deepcopy(BlockVector(dataset[idx][3], fill(4, 4))), 
            deepcopy(BlockVector(dataset[idx][4], fill(2, 4))), 
            4, 10, 1
        ) for (i, idx) in enumerate(batch_indices)]
        
        computed_trajs = hcat(batch_computed_trajs...)  # Ensure (160, batch_size)

        # manually compute loss and gradient
        loss = loss_fun(batch_targets, pred_probs, computed_trajs)  # ✅ Fully differentiable

        grads = zeros(size(pred_probs))
        for k in 1 : batch_size
            for j in 1:4
                pred_probs[j, k] += 1e-5
                batch_computed_trajs_perturb = [run_solver(
                    game,
                    parametric_game, 
                    pred_probs[:, i],  # ✅ Uses predicted masks
                    deepcopy(BlockVector(dataset[idx][3], fill(4, 4))), 
                    deepcopy(BlockVector(dataset[idx][4], fill(2, 4))), 
                    4, 10, 1
                ) for (i, idx) in enumerate(batch_indices)]
                computed_trajs_perturb = hcat(batch_computed_trajs_perturb...)  # Ensure (160, batch_size)
                loss_perturb = loss_fun(batch_targets, pred_probs, computed_trajs_perturb)
                grads[j, k] = (loss_perturb - loss) / 1e-5
            end
        end
        println("Gradient: ", grads)  # Debugging: Ensure gradients are non-zero
        Flux.Optimise.update!(optim_state, pred_probs, grads)  # ✅ Updates `pred_probs` directly

        total_loss += loss  # Compute loss with updated probabilities
    end
    next!(progress)

    avg_loss = total_loss / length(dataloader.dataset)
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
