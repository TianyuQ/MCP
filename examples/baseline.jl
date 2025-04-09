using JSON3, LinearAlgebra

fp = "receding_horizon_trajectories_[170]_All.json"
data = JSON3.read(open(fp, "r"))

function get_trajectory(data, N)
    trajectories = Dict{Int, Any}()
    goals = Dict{Int, Any}()
    controls = Dict{Int, Any}()

    # loop through parameters in the dictionary
    for playerid in 1:N
        # Get trajectories, goals, control of relevant players (only first sim_step for now)
        trajectories[playerid] = data["Player $playerid Trajectory"]
        goals[playerid] = data["Player $playerid Goal"]
        controls[playerid] = data["Player $playerid Control"]
    end
    return trajectories, goals, controls
end

function mask_computation(input_traj, trajectory, control, mode, sim_step, mode_parameter)
    mask = zeros(N-1)
    if mode == "All"
        mask = ones(N-1)
    elseif mode == "Neural Network Threshold"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, control, "Distance Threshold", sim_step, 2)
        else
            mask = best_model(input_traj)
            # println("Pred Mask: ", round.(mask, digits=4))
            mask = map(x -> x > mode_parameter ? 1 : 0, mask)
            # println("Pred Mask: ", mask)
        end
    elseif mode == "Distance Threshold"
        mask = zeros(N-1)
        for player_id in 2:N
            distance = norm(trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2])
            if distance <= mode_parameter
                mask[player_id-1] = 1
            else
                mask[player_id-1] = 0
            end
        end
    elseif mode == "Nearest Neighbor"
        mask = zeros(N-1)
        distances = []
        for player_id in 2:N
            distance = norm(trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2])
            push!(distances, distance)
        end
        ranked_indices = rank_array_from_small_to_large(distances)
        for i in 1:mode_parameter-1
            mask[ranked_indices[i]] = 1
        end
    elseif mode == "Neural Network Rank"
        if sim_step <= 10
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else
            model_mask = best_model(input_traj)
            ranked_indices = rank_array_from_large_to_small(model_mask)
            mask = zeros(N-1)
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Jacobian"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # mode_parameter for initial nearest neighbor is all players for now
        else
            mask = zeros(N-1)
            delta_t = 0.1 # hard coded for now
            norm_costs = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end] - trajectory[player_id][end-3:end]) # ex: pi_{k,x} - pj_{k,x}
                delta_px = (state_differences[1] + delta_t * state_differences[3]) ^ 2 # ex: (pi_{k,x} - pj_{k,x} + delta_t * (vi_{k,x} - vj_{k,x})) ^ 2
                delta_py = (state_differences[2] + delta_t * state_differences[4]) ^ 2

                delta_vx = (state_differences[3] + delta_t * control[player_id][1]) ^ 2
                delta_vy = (state_differences[4] + delta_t * control[player_id][2]) ^ 2
                D = delta_px + delta_py + delta_vx + delta_vy # denominator of l_col between player i and j = player_id
                J1 = 1/(D ^ 2) * 2 * delta_vx * delta_t # partial derivative of l_col with respect to aj_{k,x}
                J2 = 1/(D ^ 2) * 2 * delta_vy * delta_t # partial derivative of l_col with respect to aj_{k,y}
                norm_costs[player_id-1] = norm([J1, J2]) # [J1, J2] is the jacobian of the cost function with respect to the control of player_id
            end
            ranked_indices = rank_array_from_large_to_small(norm_costs) # rank the players based on the norm of the jacobian
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Hessian"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # mode_parameter for initial nearest neighbor is all players for now
        else
            mask = zeros(N-1)
            delta_t = 0.1 # hard coded for now
            norm_costs = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end] - trajectory[player_id][end-3:end]) # ex: pi_{k,x} - pj_{k,x}
                delta_px = (state_differences[1] + delta_t * state_differences[3]) ^ 2 # ex: (pi_{k,x} - pj_{k,x} + delta_t * (vi_{k,x} - vj_{k,x})) ^ 2
                delta_py = (state_differences[2] + delta_t * state_differences[4]) ^ 2
                delta_vx = (state_differences[3] + delta_t * control[player_id][1]) ^ 2
                delta_vy = (state_differences[4] + delta_t * control[player_id][2]) ^ 2
                D = delta_px + delta_py + delta_vx + delta_vy # denominator of l_col between player i and j = player_id
                H11 = 2 * delta_t ^ 2 / D ^ 3 * (4*delta_vx ^ 2 - D) # terms of the hessian
                H12 = 8 * delta_t ^ 2 / D ^ 3 * delta_vx * delta_vy
                H22 = 2 * delta_t ^ 2 / D ^ 3 * (4*delta_vy ^ 2 - D)
                norm_costs[player_id-1] = norm([H11 H12; H12 H22]) # Frobenius norm of the Hessian
            end
            ranked_indices = rank_array_from_large_to_small(norm_costs) # rank the players based on the norm of the jacobian
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    elseif mode == "Cost Evolution"
        if sim_step == 1
            mask = mask_computation(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter) # can't compute cost evolution at sim_step 1
        else
            mask = zeros(N-1)
            mu = 1 # hard coded for now
            cost_evolution_values = zeros(N-1)
            for player_id in 2:N
                state_differences = (trajectory[1][end-3:end] - trajectory[player_id][end-3:end]) # ex: pi_{k,x} - pj_{k,x}
                D = sum(state_differences .^ 2) # denominator of mu/norm(xi_k - xj_k)^2
        
                # x_k-1 values (prior states) for player_id
                state_differences_prev = (trajectory[1][end-7:end-4] - trajectory[player_id][end-7:end-4]) # state difference for previous sim_step
                D_prev = sum(state_differences_prev .^ 2) # denominator of mu/norm(xi_k-1 - xj_k-1)^2
                
                cost_evolution_values[player_id-1] = mu / D - mu / D_prev # cost evolution value for player_id
            end
            ranked_indices = rank_array_from_large_to_small(norm_costs) # rank the players based on the norm of the jacobian
            for i in 1:mode_parameter-1
                mask[ranked_indices[i]] = 1
            end
        end
    else
        error("Invalid mode: $mode")
    end

    return mask
end

# delta_vx = (state_differences[3] + delta_t * control[player_id][sim_step][1]) ^ 2    TEMPORARY, JUST SO MINE WORKS
# delta_vy = (state_differences[4] + delta_t * control[player_id][sim_step][2]) ^ 2




# test
trajectories, goals, controls = get_trajectory(data, 4)
flattened_trajectories = Dict{Int, Vector{Float64}}()
for player_id in 1:4
    flattened_trajectories[player_id] = vcat(trajectories[player_id]...)
end
#println(controls[2][10][1])
mask = mask_computation(flattened_trajectories, flattened_trajectories, controls, "Jacobian", 10, 2)
# mask = nearest_neighbors(trajectories, 1, 3)


# for later
# stage_costs = map(1:N) do ii
#     (x, u, t, θi) -> let
#     goal = θi[end-(N+1):end-N]
#     mask = θi[end-(N-1):end]
#         norm_sqr(x[Block(ii)][1:2] - goal) + norm_sqr(x[Block(ii)][3:4]) + 0.1 * norm_sqr(u[Block(ii)]) + 2 * sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
#     end
# end


#function distance_threshold(trajectories, ego_player_id, threshold, player_num, TOTAL_PLAYERS=4) # player_num is how many players we care about
    #     mask = zeros(TOTAL_PLAYERS) # mask of size player_num
    #     mask[ego_player_id] = 1 # ego player is always in the mask
        
    #     # loop through the remaining players and check distance
    #     for playerid in 1:player_num
    #         if playerid != ego_player_id
    #             if norm(trajectories[string(ego_player_id)][1:2] - trajectories[string(playerid)][1:2]) < threshold
    #                 mask[playerid] = 1
    #             end
    #         end
    #     end
    #     return mask
    # end
    
    # function nearest_neighbors(trajectories, ego_player_id, player_num, TOTAL_PLAYERS=4) # player_num is how many players we care about
    #     mask = zeros(TOTAL_PLAYERS) # mask of size TOTAL_PLAYERS
    #     mask[ego_player_id] = 1 # ego player is always in the mask
    #     distances = zeros(TOTAL_PLAYERS)
        
    #     # loop through the remaining players and check distance
    #     for playerid in 1:TOTAL_PLAYERS
    #         if playerid != ego_player_id
    #             distances[playerid] = norm(trajectories[string(ego_player_id)][1:2] - trajectories[string(playerid)][1:2])
    #         end
            
    #     end
        
    #     # find the (player_num) nearest neighbors. if player_num < TOTAL_PLAYERS, then the rest will be ignored
    #     for i in 1:player_num
    #         min_distance = minimum(distances)
    #         min_index = argmin(distances)
    #         mask[min_index] = 1
    #         distances[min_index] = Inf
    #     end
    
    #     return mask
    # end
