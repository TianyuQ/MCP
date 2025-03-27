
using JSON3, LinearAlgebra

fp = "baseline_data\\data_test\\simulation_results_0[1, 1, 1, 1].json"
data = JSON3.read(open(fp, "r"))

function get_trajectory(data, N)
    trajectories = Dict{String, Any}()
    goals = Dict{String, Any}()
    controls = Dict{String, Any}()

    # loop through parameters in the dictionary
    for playerid in 1:N
        # Get trajectories, goals, control of relevant players (only first sim_step for now)
        trajectories[string(playerid)] = data["Player $playerid Trajectory"][1]
        goals[string(playerid)] = data["Player $playerid Goal"]
        controls[string(playerid)] = data["Player $playerid Control"]
    end
    return trajectories, goals, controls
end

function distance_threshold(trajectories, ego_player_id, threshold, player_num)
    mask = zeros(player_num) # mask of size player_num
    mask[ego_player_id] = 1 # ego player is always in the mask
    
    # loop through the remaining players and check distance
    for playerid in 1:player_num
        if playerid != ego_player_id
            if norm(trajectories[string(ego_player_id)][1:2] - trajectories[string(playerid)][1:2]) < threshold
                mask[playerid] = 1
            else
                mask[playerid] = 0
            end
        end
    end
    return mask
end

# test
trajectories, goals, controls = get_trajectory(data, 4)
mask = distance_threshold(trajectories, 1, 4, 4)



# for later
# stage_costs = map(1:N) do ii
#     (x, u, t, θi) -> let
#     goal = θi[end-(N+1):end-N]
#     mask = θi[end-(N-1):end]
#         norm_sqr(x[Block(ii)][1:2] - goal) + norm_sqr(x[Block(ii)][3:4]) + 0.1 * norm_sqr(u[Block(ii)]) + 2 * sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
#     end
# end