function run_example(;
    game,
    initial_states,
    goals,
    N,
    horizon = 3,
    num_sim_steps,
    mask,
    save = false
)
    results = Dict()
    # Prepare parameters for the game simulation
    parameters = mortar([vcat(goals[2 * (i - 1) + 1:2 * i], mask) for i in 1:N])

    # Progress bar for simulation steps
    progress = ProgressMeter.Progress(num_sim_steps)

    # Define the strategy
    ground_truth_strategy = WarmStartRecedingHorizonStrategy(;
        game,
        parametric_game,
        turn_length = 3,
        horizon,
        parameters = parameters,
    )

    # Run simulation
    sim_steps = rollout(
        game.dynamics,
        ground_truth_strategy,
        initial_states,
        num_sim_steps;
        get_info = (γ, x, t) -> (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
    )

    max_iteration = length(sim_steps)

    # Add initial states and goals to the results dictionary
    if save
        for player_id in 1:N
            results["Player $player_id Initial State"] = initial_states[4 * (player_id - 1) + 1:4 * player_id]
            results["Player $player_id Goal"] = goals[2 * (player_id - 1) + 1:2 * player_id]
            results["Player $player_id Trajectory"] = [sim_steps[max_iteration][step].substrategies[player_id].xs for step in 1:num_sim_steps]
            results["Player $player_id Control"] = [sim_steps[max_iteration][step].substrategies[player_id].us for step in 1:num_sim_steps]
        end

        return results
    else
        # Collect trajectories for all players and flatten into a single long vector
        flattened_traj = flatten_trajectory(sim_steps, max_iteration, num_sim_steps, N)
        return flattened_traj  # Return only trajectories as a long vector
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
