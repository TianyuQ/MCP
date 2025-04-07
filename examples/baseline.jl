
using JSON3, LinearAlgebra

fp = "receding_horizon_trajectories_[170]_All.json"
data = JSON3.read(open(fp, "r"))

function get_trajectory(data, N)
    trajectories = Dict{String, Any}()
    goals = Dict{String, Any}()
    controls = Dict{String, Any}()

    # loop through parameters in the dictionary
    for playerid in 1:N
        # Get trajectories, goals, control of relevant players (only first sim_step for now)
        trajectories[string(playerid)] = data["Player $playerid Trajectory"]
        goals[string(playerid)] = data["Player $playerid Goal"]
        controls[string(playerid)] = data["Player $playerid Control"]
    end
    return trajectories, goals, controls
end

function distance_threshold(trajectories, ego_player_id, threshold, player_num, TOTAL_PLAYERS=4) # player_num is how many players we care about
    mask = zeros(TOTAL_PLAYERS) # mask of size player_num
    mask[ego_player_id] = 1 # ego player is always in the mask
    
    # loop through the remaining players and check distance
    for playerid in 1:player_num
        if playerid != ego_player_id
            if norm(trajectories[string(ego_player_id)][1:2] - trajectories[string(playerid)][1:2]) < threshold
                mask[playerid] = 1
            end
        end
    end
    return mask
end

function nearest_neighbors(trajectories, ego_player_id, player_num, TOTAL_PLAYERS=4) # player_num is how many players we care about
    mask = zeros(TOTAL_PLAYERS) # mask of size TOTAL_PLAYERS
    mask[ego_player_id] = 1 # ego player is always in the mask
    distances = zeros(TOTAL_PLAYERS)
    
    # loop through the remaining players and check distance
    for playerid in 1:TOTAL_PLAYERS
        if playerid != ego_player_id
            distances[playerid] = norm(trajectories[string(ego_player_id)][1:2] - trajectories[string(playerid)][1:2])
        end
        
    end
    
    # find the (player_num) nearest neighbors. if player_num < TOTAL_PLAYERS, then the rest will be ignored
    for i in 1:player_num
        min_distance = minimum(distances)
        min_index = argmin(distances)
        mask[min_index] = 1
        distances[min_index] = Inf
    end

    return mask
end

function jacobian(input_traj, trajectory, mode, sim_step, mode_parameter)
    mask = zeros(N-1)
    for player_id in 2:N
        del_px = trajectory[1][end-3:end-2] - trajectory[player_id][end-3:end-2]
    end

end


print(trajectories["1"][1][end-3:end-2])









# test
# trajectories, goals, controls = get_trajectory(data, 4)
# mask = distance_threshold(trajectories, 1, 4, 4)
# mask = nearest_neighbors(trajectories, 1, 3)


# for later
# stage_costs = map(1:N) do ii
#     (x, u, t, θi) -> let
#     goal = θi[end-(N+1):end-N]
#     mask = θi[end-(N-1):end]
#         norm_sqr(x[Block(ii)][1:2] - goal) + norm_sqr(x[Block(ii)][3:4]) + 0.1 * norm_sqr(u[Block(ii)]) + 2 * sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
#     end
# end


