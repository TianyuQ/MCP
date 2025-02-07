using CSV
using DataFrames
using JSON

global dir_path = "/home/tq877/Tianyu/player_selection/MCP/data_bak"

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

    cost_json = """
        norm_sqr(x[Block(ii)][1:2] - goal) +
        sum(1 / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii) +
        1norm_sqr(x[Block(ii)][3:4]) + 
        0.1norm_sqr(u[Block(ii)])
    """

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

    # game_meta_data = Dict()
    # # Save the meta data of the game, including dynamics, costs, and constraints
    # meta_data = Dict(
    #     "Dynamics" => dynamics_json,
    #     "Cost" => cost_json,
    #     "Constraints" => coupling_constraints_json,
    #     "Mask" => mask,
    #     "Goals" => goals,
    #     "Initial States" => initial_states,
    # )
    # # meta_data_path = "/home/tq877/Tianyu/player_selection/MCP/examples/game_meta_data.json"
    # # meta_data_path = joinpath(dir_path, "game_meta_data.json")
    # meta_data_path = joinpath(dir_path, "game_meta_data_$scenario_id.json")

    # game_meta_data["Meta Data"] = meta_data

    # open(meta_data_path, "w") do io
    #     JSON.print(io, game_meta_data)
    # end

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
    else
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
            # else
            #     results["Player $player_id Initial State"] = []
            #     results["Player $player_id Goal"] = []
            #     results["Player $player_id Trajectory"] = []
            #     results["Player $player_id Control"] = []
            end
        end
    end
    # result_path = "/home/tq877/Tianyu/player_selection/MCP/examples/simulation_results.json"
    # result_path = joinpath(dir_path, "simulation_results_$scenario_id.json")
    result_path = joinpath(dir_path, "simulation_results_$scenario_id$mask.json")
    open(result_path, "w") do io
        JSON.print(io, results)
    end
end

function generate_results(;
    N = 4, #number of total players
    horizon = 10,
    scenario_num = 100,
    total_steps = 1,
)
    for scenario_id in 0:scenario_num-1
        println("Scenario $scenario_id")
        file_path = joinpath(dir_path, "scenario_$scenario_id.csv")
        data = CSV.read(file_path, DataFrame)
        goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
        initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
        masks = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        for mask in masks
            println("Mask: $mask")
            run_example(
                initial_states = initial_states,
                goals = goals,
                N = N,
                horizon = horizon,
                num_sim_steps = total_steps,
                scenario_id = scenario_id,
                mask = mask,
            )
        end
    end
end