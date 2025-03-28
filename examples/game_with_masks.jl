using CSV
using DataFrames
using JSON

global dir_path = "C:/UT Austin/Research/MCP/data_vel_0_10"

# function generate_results(;
# N = 10, #number of total players
# horizon = 30,
scenario_num = 200
total_steps = 1
target = [0 for _ in 1:N * horizon * d]
masks = [ones(N)]

for scenario_id in 0:199
    println("Scenario $scenario_id")
    file_path = joinpath(dir_path, "scenario_$scenario_id.csv")
    data = CSV.read(file_path, DataFrame)
    goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
    initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
    # masks = [
    #     # [[1; collect(masks)] for masks in Iterators.product((0:1 for _ in 2:N)...)]
    # ]
    # masks = [[1; collect(bits)] for bits in Iterators.product(fill((0,1), N-1)...)]
    # masks = [ones(N)]
    for mask in masks
        # println("Mask: $mask")
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
# end