"Utility to create the road environment."
function setup_road_environment(; lane_width = 2, num_lanes = 2, height = 50)
    lane_centers = map(lane_idx -> (lane_idx - 0.5) * lane_width, 1:num_lanes)
    vertices = [
        [first(lane_centers) - 0.5lane_width, 0],
        [last(lane_centers) + 0.5lane_width, 0],
        [last(lane_centers) + 0.5lane_width, height],
        [first(lane_centers) - 0.5lane_width, height],
    ]

    (; lane_centers, environment = PolygonEnvironment(vertices))
end

"Utility to set up a (two player) trajectory game."
function setup_trajectory_game(; environment)
    cost = let
        stage_costs = map(1:2) do ii
            (x, u, t, θi) -> let
                lane_preference = last(θi)

                (x[Block(ii)][1] - lane_preference)^2 +
                0.5norm_sqr(x[Block(ii)][3:4] - [0, 2]) +
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
            x1, x2 = blocks(x)

            # Players need to stay at least 2 m away from one another.
            norm_sqr(x1[1:2] - x2[1:2]) - 4
        end
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -10, 0], ub = [Inf, Inf, 10, 10]),
        control_bounds = (; lb = [-5, -5], ub = [3, 3]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:2])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function run_lane_change_example(;
    initial_state = mortar([[1.0, 1.0, 0.0, 1.0], [3.2, 0.9, 0.0, 1.0]]),
    horizon = 10,
    height = 50.0,
    num_lanes = 2,
    lane_width = 2,
    num_sim_steps = 150,
)
    (; environment, lane_centers) =
        setup_road_environment(; num_lanes, lane_width, height)
    game = setup_trajectory_game(; environment)

    # Build a game. Each player has a parameter for lane preference.
    # P1 wants to stay in the left lane, and P2 wants to move from the
    # right to the left lane.
    lane_preferences = mortar([[lane_centers[1]], [lane_centers[1]]])
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
            parameters = lane_preferences,
        )

        rollout(
            game.dynamics,
            ground_truth_strategy,
            initial_state,
            num_sim_steps;
            get_info = (γ, x, t) ->
                (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
        )
    end

    animate_sim_steps(
        game,
        sim_steps;
        live = false,
        framerate = 20,
        show_turn = true,
        xlims = (first(lane_centers) - lane_width, last(lane_centers) + lane_width),
        ylims = (-1, height + 1),
        aspect = num_lanes * lane_width / height,
    )
end