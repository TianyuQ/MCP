using CSV
using DataFrames

# Read the CSV file
file_path = "/home/tq877/Tianyu/player_selection/MCP/scripts/agents_and_goals_0.csv"  # Replace with your actual file path
data = CSV.read(file_path, DataFrame)

N = 4

goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])

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

"Utility to set up a (two player) trajectory game."
function setup_trajectory_game(; environment)
    cost = let
        stage_costs = map(1:N) do ii
            (x, u, t, θi) -> let
            # println(θi)
            # lane_preference = last(θi)
            goal = goals[Block(ii)]

                # (x[Block(ii)][1] - goal[1])^2 + (x[Block(ii)][2] - goal[2])^2 +
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
        # mapreduce(vcat, xs) do x
        #     x1, x2 = blocks(x)

        #     # Players need to stay at least 2 m away from one another.
        #     norm_sqr(x1[1:2] - x2[1:2]) - 4
        # end

        # h = mapreduce(vcat, xs) do x
        #     # x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = blocks(x)
        #     # x1, x2, x3, x4, x5, x6 = blocks(x)
        #     x1, x2, x3, x4 = blocks(x)

        #     # Players need to stay at least 2 m away from one another.
        #     # norm_sqr(x1[1:2] - x2[1:2]) + 1
        #     [
        #         norm_sqr(x1[1:2] - x2[1:2]) - 4
        #         norm_sqr(x1[1:2] - x3[1:2]) - 4
        #         norm_sqr(x1[1:2] - x4[1:2]) - 4
        #         norm_sqr(x2[1:2] - x3[1:2]) - 4
        #         norm_sqr(x2[1:2] - x4[1:2]) - 4
        #         norm_sqr(x3[1:2] - x4[1:2]) - 4
        #     ]
        # end

        mapreduce(vcat, xs) do x
            player_states = blocks(x)  # Extract exactly N player states
    
            # Compute constraints for all unique player pairs
            # [norm_sqr(player_states[i][1:2] - player_states[j][1:2]) - 0.25
            #  for i in 1:N-1 for j in i+1:N
            # ]
            [
                1
            ]
        end
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -2, -2], ub = [Inf, Inf, 2, 2]),
        control_bounds = (; lb = [-3, -3], ub = [3, 3]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:N])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function run_lane_change_example(;
    # initial_state = mortar([[1.0, 1.0, 0.0, 1.0], [3.2, 0.9, 0.0, 1.0], [-0.2, 0.9, 0.0, 1.0]]),
    initial_state = initial_states,
    horizon = 3,
    height = 50.0,
    num_lanes = 2,
    lane_width = 2,
    num_sim_steps = 4,
)
    (; environment) =
        setup_road_environment(; length = 10)
    game = setup_trajectory_game(; environment)

    # Build a game. Each player has a parameter for lane preference.
    # P1 wants to stay in the left lane, and P2 wants to move from the
    # right to the left lane.
    lane_preferences = mortar([[0] for _ in 1:N])
    # println(lane_preferences)
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

    println("Simulation Results:")
    max_steps = length(sim_steps)
    println("Step $max_steps:")
    for i in 1:4
        println(sim_steps[max_steps][i].substrategies[1].xs[1]) 
    end

    # animate_sim_steps(
    #     game,
    #     sim_steps;
    #     live = false,
    #     framerate = 20,
    #     show_turn = true,
    #     xlims = (-6, 6),
    #     ylims = (-6, 6),
    #     aspect = 1,
    # )
end