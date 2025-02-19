"Utility to create the road environment."
function setup_road_environment(; length)
    vertices = [
        [-0.5length, -0.5length],
        [0.5length, -0.5length],
        [0.5length, 0.5length],
        [-0.5length, 0.5length],
    ]
    (; environment = PolygonEnvironment(vertices))
end

"Utility to set up a trajectory game."
function setup_trajectory_game(; environment, scenario_id = 0, goals, N)
    cost = let
        stage_costs = map(1:N) do ii
            (x, u, t, θi) -> let
            goal = goals[Block(ii)]
                norm_sqr(x[Block(ii)][1:2] - goal) +
                sum(1 / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii) +
                1norm_sqr(x[Block(ii)][3:4]) + 
                0.1norm_sqr(u[Block(ii)])
            end
        end

        function reducer(stage_costs)
            reduce(+, stage_costs) / length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(
            stage_costs,
            reducer,
            GeneralSumCostStructure(),
            1.0,
        )
    end

    function coupling_constraints(xs, us, θ)
        mapreduce(vcat, xs) do x
            player_states = blocks(x)  # Extract exactly N player states
            [ 1 ]
        end
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -2, -2], ub = [Inf, Inf, 2, 2]),
        control_bounds = (; lb = [-3, -3], ub = [3, 3]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:N])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function setup_game_like_optimal_control(; environment, scenario_id = 0, goals)
    cost = let
        stage_costs = map(1:2) do ii
            (x, u, t, θi) -> let
            goal = goals[Block(ii)]
                norm_sqr(x[Block(ii)][1:2] - goal) +
                1norm_sqr(x[Block(ii)][3:4]) + 
                0.1norm_sqr(u[Block(ii)])
            end
        end

        function reducer(stage_costs)
            reduce(+, stage_costs) / length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(
            stage_costs,
            reducer,
            GeneralSumCostStructure(),
            1.0,
        )
    end

    function coupling_constraints(xs, us, θ)
        mapreduce(vcat, xs) do x
            player_states = blocks(x)  # Extract exactly N player states
            [1]
        end
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -2, -2], ub = [Inf, Inf, 2, 2]),
        control_bounds = (; lb = [-3, -3], ub = [3, 3]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:2])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function run_example(;
    initial_states = initial_states,
    goals = goals,
    N = N,
    horizon = 3,
    num_sim_steps = total_steps,
    scenario_id = 0,
    mask,
)
    # mask = Int.(mask .>= 0.5)
    # println(mask)
    results = Dict()

    #Optional Game for all the nonchosen players except the ego player
    for i in 1:N
        if mask[i] == 0
            optional_initial_states = mortar([initial_states[4 * (i - 1) + 1:4 * i],[0, 0, 0, 0]])
            optional_goals = mortar([goals[2 * (i - 1) + 1:2 * i],[0, 0]])
            (; environment) = setup_road_environment(; length = 7)
            game = setup_game_like_optimal_control(; environment, scenario_id = scenario_id, goals = optional_goals)
            optional_parameters = mortar([[0] for _ in 1:2])
            parametric_game = build_parametric_game(; game, horizon, params_per_player = 1)
        
            turn_length = 3
            sim_steps = let
                progress = ProgressMeter.Progress(num_sim_steps)
                ground_truth_strategy = WarmStartRecedingHorizonStrategy(;
                    game,
                    parametric_game,
                    turn_length,
                    horizon,
                    parameters = optional_parameters,
                )

                rollout(
                    game.dynamics,
                    ground_truth_strategy,
                    optional_initial_states,
                    num_sim_steps;
                    get_info = (γ, x, t) ->
                        (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
                )
            end
            max_iteration = length(sim_steps)
            results["Player $i Initial State"] = initial_states[4 * (i - 1) + 1:4 * i]
            results["Player $i Goal"] = goals[2 * (i - 1) + 1:2 * i]
            results["Player $i Trajectory"] = [sim_steps[max_iteration][step].substrategies[1].xs for step in 1:num_sim_steps]
            results["Player $i Control"] = [sim_steps[max_iteration][step].substrategies[1].us for step in 1:num_sim_steps]
        end
    end

    if sum(mask) == 1
        i = findfirst(mask .== 1)
        optional_initial_states = mortar([initial_states[4 * (i - 1) + 1:4 * i],[0, 0, 0, 0]])
        optional_goals = mortar([goals[2 * (i - 1) + 1:2 * i],[0, 0]])
        (; environment) = setup_road_environment(; length = 7)
        game = setup_game_like_optimal_control(; environment, scenario_id = scenario_id, goals = optional_goals)
        optional_parameters = mortar([[0] for _ in 1:2])
        parametric_game = build_parametric_game(; game, horizon, params_per_player = 1)
    
        turn_length = 3
        sim_steps = let
            progress = ProgressMeter.Progress(num_sim_steps)
            ground_truth_strategy = WarmStartRecedingHorizonStrategy(;
                game,
                parametric_game,
                turn_length,
                horizon,
                parameters = optional_parameters,
            )

            rollout(
                game.dynamics,
                ground_truth_strategy,
                optional_initial_states,
                num_sim_steps;
                get_info = (γ, x, t) ->
                    (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
            )
        end
        max_iteration = length(sim_steps)
        results["Player $i Initial State"] = initial_states[4 * (i - 1) + 1:4 * i]
        results["Player $i Goal"] = goals[2 * (i - 1) + 1:2 * i]
        results["Player $i Trajectory"] = [sim_steps[max_iteration][step].substrategies[1].xs for step in 1:num_sim_steps]
        results["Player $i Control"] = [sim_steps[max_iteration][step].substrategies[1].us for step in 1:num_sim_steps]
    elseif sum(mask) > 1
        #Main Game
        masked_N = sum(mask)
        masked_initial_states = mortar([initial_states[4 * (i - 1) + 1:4 * i] for i in 1:N if mask[i] == 1])
        masked_goals = mortar([goals[2 * (i - 1) + 1:2 * i] for i in 1:N if mask[i] == 1])

        (; environment) = setup_road_environment(; length = 7)
        game = setup_trajectory_game(; environment, scenario_id = scenario_id, goals = masked_goals, N = masked_N)
        default_parameters = mortar([[0] for _ in 1:masked_N])
        parametric_game = build_parametric_game(; game, horizon, params_per_player = 1)
        # Simulate the ground truth.
        turn_length = 3
        sim_steps = let
            progress = ProgressMeter.Progress(num_sim_steps)
            ground_truth_strategy = WarmStartRecedingHorizonStrategy(;
                game,
                parametric_game,
                turn_length,
                horizon,
                parameters = default_parameters,
            )

            rollout(
                game.dynamics,
                ground_truth_strategy,
                masked_initial_states,
                num_sim_steps;
                get_info = (γ, x, t) ->
                    (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
            )
        end

        max_iteration = length(sim_steps)

        # Add initial states and goals to the results dictionary
        for player_id in 1:N
            if mask[player_id] == 1
                masked_index = sum(mask[1:player_id])
                results["Player $player_id Initial State"] = initial_states[4 * (masked_index - 1) + 1:4 * masked_index]
                results["Player $player_id Goal"] = goals[2 * (masked_index - 1) + 1:2 * masked_index]
                results["Player $player_id Trajectory"] = [sim_steps[max_iteration][step].substrategies[masked_index].xs for step in 1:num_sim_steps]
                results["Player $player_id Control"] = [sim_steps[max_iteration][step].substrategies[masked_index].us for step in 1:num_sim_steps]
            end
        end
    end
    return results
end