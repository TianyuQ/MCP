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
function setup_trajectory_game(; environment, scenario_id = 0, N)
    cost = let
        stage_costs = map(1:N) do ii
            (x, u, t, θi) -> let
            goal = θi[end-5:end-4]
            mask = θi[end-3:end]
                norm_sqr(x[Block(ii)][1:2] - goal) +
                1norm_sqr(x[Block(ii)][3:4]) + 
                0.1norm_sqr(u[Block(ii)]) + 
                sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
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

function run_example(;
    initial_states = initial_states,
    goals = goals,
    N = N,
    horizon = 3,
    num_sim_steps = total_steps,
    scenario_id = 0,
    mask,
)
    results = Dict()

    (; environment) = setup_road_environment(; length = 7)
    game = setup_trajectory_game(; environment, scenario_id = scenario_id, N = N)
    parameters = mortar([vcat(goals[2 * (i - 1) + 1:2 * i], mask) for i in 1:N])
    parametric_game = build_parametric_game(; game, horizon, params_per_player = 6)
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