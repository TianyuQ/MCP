function run_example(;
    game,
    parametric_game,
    initial_states,
    goals,
    N,
    horizon = 3,
    num_sim_steps,
    mask,
    target,
    weight,
    save = false
)
    results = Dict()
    # Prepare parameters for the game simulation
    parameters = mortar([vcat(goals[2 * (i - 1) + 1:2 * i], i == 1 ? mask : [1, 1, 1, 1]) for i in 1:N])
    # parameters = mortar([vcat(goals[2 * (i - 1) + 1:2 * i], mask) for i in 1:N])

    # Progress bar for simulation steps
    progress = ProgressMeter.Progress(num_sim_steps)

    # Define the strategy
    strategy = nothing
    strategy = WarmStartRecedingHorizonStrategy(;
        game,
        parametric_game,
        turn_length = 1,
        horizon,
        parameters = parameters,
        target = target,
        initial_states = initial_states,
    )

    _, gradient, sim_steps = rollout(
        game.dynamics,
        strategy,
        initial_states,
        num_sim_steps;
        get_info = (γ, x, t) -> (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
    )
    # println("strategy: ", strategy.receding_horizon_strategy.substrategies[1].xs)
    # println("sim_steps[1]: ", sim_steps[1])
    # println("sim_steps[2]: ", sim_steps[2])
    # println("sim_steps: ", sim_steps)
    # println("sim_steps: ", sim_steps[step].substrategies[player_id].xs for step in 1:num_sim_steps)
    # diff = sim_steps[step].substrategies[player_id].xs - strategy.receding_horizon_strategy.substrategies[player_id].xs
    #  for step in 1:num_sim_steps
    

    # mean_gradient = (true_gradient1 .+ true_gradient2 .+ true_gradient3 .+ true_gradient4) ./ 4

    # println("sim_steps: ", sim_steps.us)
    # println("ground_truth_strategy.loss: ", ground_truth_strategy.gradient)
    # max_iteration = length(sim_steps)

    # Add initial states and goals to the results dictionary
    if save
        for player_id in 1:N
            results["Player $player_id Initial State"] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
            results["Player $player_id Goal"] = goals[2 * (player_id - 1) + 1:2 * player_id]
            # results["Player $player_id Trajectory"] = [sim_steps[step].substrategies[player_id].xs for step in 1:num_sim_steps]
            # results["Player $player_id Control"] = [sim_steps[step].substrategies[player_id].us for step in 1:num_sim_steps]
            results["Player $player_id Trajectory"] = [strategy.receding_horizon_strategy.substrategies[player_id].xs for step in 1:num_sim_steps]
            results["Player $player_id Control"] = [strategy.receding_horizon_strategy.substrategies[player_id].us for step in 1:num_sim_steps]
        end

        return results
    else
        # Collect trajectories for all players and flatten into a single long vector
        # flattened_traj = flatten_trajectory(sim_steps, max_iteration, num_sim_steps, N)  
        # return flattened_traj, strategy  # Return only trajectories as a long vector
        # println("Loss: ", loss)
        return gradient[1][2], gradient[1][3]
    end
end

function flatten_trajectory(sim_steps, max_iteration, num_sim_steps, N)
    trajectory = []
    for player_id in 1:N
        for step in 1:num_sim_steps
            testtraj = vcat(sim_steps[max_iteration][step].substrategies[player_id].xs)
            append!(trajectory, hcat(testtraj...))
        end
    end
    # println("Finished")
    return trajectory
end
