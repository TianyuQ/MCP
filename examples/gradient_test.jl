using CSV
using DataFrames

# Initialize an empty DataFrame to store the loss values
loss_df = DataFrame(mask_3=Float64[], mask_4=Float64[], loss=Float64[])

for (batch_inputs, batch_targets, batch_initial_states, batch_goals, batch_indices) in dataloader
    # for mask_2 in 1:1
        for mask_3 in 0:0.1:1
            for mask_4 in 0:0.1:1
                # pred_mask = [1, mask_2, mask_3, mask_4]
                pred_mask = [1, 1, mask_3, mask_4]
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
                loss_val = results[2]
                println("Loss Values: ", loss_val)
                total_loss = mean(loss_val)
                
                # Round the total_loss to 4 digits and add to DataFrame
                push!(loss_df, (mask_3, mask_4, round(total_loss, digits=4)))
            end
        end
    # end
end

# Save the DataFrame to a CSV file
CSV.write("/home/tq877/Tianyu/player_selection/MCP/examples/loss_values_[1111]_01.csv", loss_df)

using Plots

# Load the loss values from the CSV file
loss_df = CSV.read("/home/tq877/Tianyu/player_selection/MCP/examples/loss_values_[1111]_01.csv", DataFrame)

# Pivot the DataFrame to get mask_3, mask_4, and loss values in a matrix form
pivot_df = unstack(loss_df, :mask_4, :mask_3, :loss)

# Extract the mask_3 and mask_4 values
mask_3_values = unique(loss_df.mask_3)
mask_4_values = unique(loss_df.mask_4)

# Extract the loss matrix
loss_matrix = Matrix{Float64}(pivot_df[:, 2:end])

# Plot the heatmap
heatmap(mask_3_values, mask_4_values, loss_matrix, xlabel="Mask 3", ylabel="Mask 4", title="Loss Heatmap", color=:viridis)
savefig("/home/tq877/Tianyu/player_selection/MCP/examples/loss_heatmap_1111_01.png")




# # Initialize an empty DataFrame to store the loss values
# loss_df = DataFrame(mask_2=Float64[], mask_3=Float64[], mask_4=Float64[], loss=Float64[])

# for (batch_inputs, batch_targets, batch_initial_states, batch_goals, batch_indices) in dataloader
#     for mask_2 in 0:0.1:1
#         for mask_3 in 0:0.1:1
#             for mask_4 in 0:0.1:1
#                 pred_mask = [1, mask_2, mask_3, mask_4]
#                 results = run_solver(
#                     game, 
#                     parametric_game, 
#                     batch_targets[:,1], 
#                     batch_initial_states[:,1], 
#                     batch_goals[:,1], 
#                     N, 
#                     horizon, 
#                     num_sim_steps, 
#                     pred_mask
#                 )
#                 loss_val = results[2]
#                 println("Loss Values: ", loss_val)
#                 total_loss = mean(loss_val)
                
#                 # Round the total_loss to 4 digits and add to DataFrame
#                 push!(loss_df, (mask_2, mask_3, mask_4, round(total_loss, digits=4)))
#             end
#         end
#     end
# end

# # Save the DataFrame to a CSV file
# CSV.write("/home/tq877/Tianyu/player_selection/MCP/examples/loss_values_2.csv", loss_df)

# using CSV
# using DataFrames
# using Plots

# # Load the loss values from the CSV file
# loss_df = CSV.read("/home/tq877/Tianyu/player_selection/MCP/examples/loss_values_2.csv", DataFrame)

# # Get unique values for each mask variable
# mask_2_values = unique(loss_df.mask_2)
# mask_3_values = unique(loss_df.mask_3)
# mask_4_values = unique(loss_df.mask_4)

# # Loop over each value of mask_2 and create a 3D surface plot
# for mask_2 in mask_2_values
#     # Filter the DataFrame for the current mask_2 value
#     df_subset = filter(row -> row.mask_2 == mask_2, loss_df)
    
#     # Pivot the DataFrame to get loss values in matrix form
#     pivot_df = unstack(df_subset, :mask_4, :mask_3, :loss)
#     loss_matrix = Matrix{Float64}(pivot_df[:, 2:end])
    
#     # Create the 3D surface plot
#     plot(
#         mask_3_values,
#         mask_4_values,
#         loss_matrix,
#         st = :surface,
#         xlabel = "Mask 3",
#         ylabel = "Mask 4",
#         zlabel = "Loss",
#         title = "3D Loss Surface (Mask 2 = $mask_2)",
#         color = :viridis
#     )
    
#     # Save each plot as a separate image
#     savefig("/home/tq877/Tianyu/player_selection/MCP/examples/loss_3d_surface_mask2_$mask_2.png")
# end