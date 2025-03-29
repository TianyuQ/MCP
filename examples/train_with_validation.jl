###############################################################################
# Initialize Model & Optimizer
###############################################################################
println("Initializing model...")
global model = build_model()     # Ensure build_model() accepts required arguments
println("Model initialized successfully!")

###############################################################################
# Import Required Packages
###############################################################################
using TensorBoardLogger
using ProgressMeter
using BSON
using JSON
using Zygote
using Flux

###############################################################################
# Initialize TensorBoard Logger
###############################################################################
tb_logger = TBLogger("logs/$record_name")  # Logs will be saved under logs/record_name

###############################################################################
# Training Loop with Validation and Early Stopping
###############################################################################
println("Starting training...")
training_losses = Dict()

for epoch in 1:epochs
    println("Epoch $epoch")
    global total_loss = 0.0
    progress = Progress(length(train_dataloader.dataset), desc="Epoch $epoch Training Progress")
    
    # ---------------------------
    # Training Phase
    # ---------------------------
    for (batch_inputs, batch_targets, batch_initial_states, batch_goals, batch_indices) in train_dataloader
        current_masks = [model(batch_inputs[:, i]) for i in 1:batch_size]
        pred_masks = [vcat([1], mask) for mask in current_masks]
        println("\nTrain Masks: ", [round.(mask, digits=4) for mask in pred_masks])
        
        # batch_loss = 0.0
        batch_grads = []
        
        # Process each example in the batch
        for i in 1:batch_size
            results = run_solver(
                game,
                parametric_game,
                batch_targets[:, i],
                batch_initial_states[:, i],
                batch_goals[:, i],
                N, horizon, num_sim_steps,
                pred_masks[i]
            )
            upstream_grads = clamp.(results[1], -10, 10)
            loss_val = results[2]
            total_loss += mean(loss_val)
            
            # Compute gradients via pullback
            _, back = Zygote.pullback(() -> model(batch_inputs[:, i]), Flux.params(model))
            grads_example = back(upstream_grads)
            push!(batch_grads, grads_example)
        end
        
        # total_loss += batch_loss
        
        # Gradient accumulation and parameter update
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
        
        for p in params_set
            mean_grad = accum_grads[p] ./ batch_size
            p .-= learning_rate * mean_grad
        end
        
        next!(progress)
    end
    global total_loss
    total_loss /= train_batches
    training_losses[epoch] = total_loss
    println("\nEpoch $epoch: Training Loss = $total_loss")
    log_value(tb_logger, "train_loss", total_loss, step=epoch)
    
    # ---------------------------
    # Validation Phase
    # ---------------------------
    global val_loss = 0.0
    for (val_inputs, val_targets, val_initial_states, val_goals, val_indices) in val_dataloader
        current_masks = [model(val_inputs[:, i]) for i in 1:batch_size]
        pred_masks = [vcat([1], mask) for mask in current_masks]
        # println("\nVal Masks: ", [round.(mask, digits=4) for mask in pred_masks])
        
        for i in 1:batch_size
            results = run_solver(
                game,
                parametric_game,
                val_targets[:, i],
                val_initial_states[:, i],
                val_goals[:, i],
                N, horizon, num_sim_steps,
                pred_masks[i]
            )
            loss_val = results[2]
            val_loss += mean(loss_val)
        end
    end
    global val_loss
    val_loss /= val_batches
    println("Epoch $epoch: Validation Loss = $val_loss")
    log_value(tb_logger, "val_loss", val_loss, step=epoch)
    
    # ---------------------------
    # Early Stopping Check
    # ---------------------------
    global best_val_loss
    if val_loss < best_val_loss
        best_val_loss = val_loss
        global patience_counter
        patience_counter = 0
        # Save the best model so far
        BSON.bson("logs/$record_name/best_model.bson", Dict(:model => model))
    else
        global patience_counter
        patience_counter += 1
        if patience_counter >= patience
            println("Early stopping triggered at epoch $epoch")
            break
        end
    end
end

###############################################################################
# Save Final Trained Model & Training Loss Records
###############################################################################
println("\nSaving trained generative model...")
BSON.bson("logs/$record_name/trained_model.bson", Dict(:model => model))
println("Model saved successfully!")

println("\nSaving training loss records...")
open("logs/$record_name/training_losses.json", "w") do f
    JSON.print(f, sort(collect(training_losses)), 4)
end
println("Training loss saved successfully to logs/$record_name/training_losses.json!")

###############################################################################
# Close the TensorBoard Logger
###############################################################################
close(tb_logger)
