function run_example(;
    game = game,
    initial_states = initial_states,
    goals = goals,
    N = N,
    horizon = 3,
    num_sim_steps = total_steps,
    mask,
)
    results = Dict()

    # (; environment) = setup_road_environment(; length = 7)
    # game = setup_trajectory_game(; environment, scenario_id = scenario_id, N = N)
    parameters = mortar([vcat(goals[2 * (i - 1) + 1:2 * i], mask) for i in 1:N])
    # parametric_game = build_parametric_game(; game, horizon, params_per_player = 6)
    # Simulate the ground truth.
    turn_length = 3
    sim_steps = let
        progress = ProgressMeter.Progress(num_sim_steps)
        ground_truth_strategy = WarmStartRecedingHorizonStrategy(;
            game,
            parametric_game,
            turn_length,
            horizon,
            parameters = parameters,
        )

        rollout(
            game.dynamics,
            ground_truth_strategy,
            initial_states,
            num_sim_steps;
            get_info = (γ, x, t) ->
                (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
        )
    end

    max_iteration = length(sim_steps)

    # Add initial states and goals to the results dictionary
    for player_id in 1:N
        results["Player $player_id Initial State"] = initial_states
        results["Player $player_id Goal"] = goals
        results["Player $player_id Trajectory"] = [sim_steps[max_iteration][step].substrategies[player_id].xs for step in 1:num_sim_steps]
        results["Player $player_id Control"] = [sim_steps[max_iteration][step].substrategies[player_id].us for step in 1:num_sim_steps]
    end
    return results
end