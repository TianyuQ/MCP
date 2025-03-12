using CSV
using DataFrames
using JSON

global dir_path = "/home/tq877/Tianyu/player_selection/MCP/data_circle/"

function generate_results(;
    N = 4, #number of total players
    horizon = 30,
    scenario_num = 100,
    total_steps = 1,
)
    # (; environment) = setup_road_environment(; length = 7)
    # game = setup_trajectory_game(; environment, N = 4)
    # parametric_game = build_parametric_game(; game, horizon=horizon, params_per_player = 6)
    target = [0 for _ in 1:N * horizon * d]
    for scenario_id in 0:99
        println("Scenario $scenario_id")
        file_path = joinpath(dir_path, "scenario_$scenario_id.csv")
        data = CSV.read(file_path, DataFrame)
        goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
        initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
        masks = [
            [1, 0, 0, 0],
            # [0, 1, 0, 0],
            # [0, 0, 1, 0],
            # [0, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            # [0, 1, 1, 0],
            # [0, 1, 0, 1],
            # [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            # [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        for mask in masks
            println("Mask: $mask")
            result = run_example(;
                game,
                parametric_game,
                initial_states,
                goals,
                N,
                horizon = horizon,
                num_sim_steps = total_steps,
                mask,
                target,
                save = true
            )
            result_path = joinpath(dir_path, "simulation_results_$scenario_id$mask.json")
            open(result_path, "w") do io
                JSON.print(io, result)
            end
        end
    end
end