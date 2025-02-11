using CSV
using DataFrames

mask = [1, 0, 1, 1]
horizon = 3
num_sim_steps = 1
N = 4

file_path = "/home/tq877/Tianyu/player_selection/MCP/data_bak/scenario_0.csv"
data = CSV.read(file_path, DataFrame)
goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])
println(typeof(initial_states))
println(typeof(goals))

# computed_results = run_example(
#     initial_states = initial_states,
#     goals = goals,
#     N = N,
#     horizon = horizon,
#     num_sim_steps = num_sim_steps,
#     mask = mask
# )

# grad = only(Zygote.gradient(f, θ))


# open("traj_results.json", "w") do io
#     JSON.print(io, computed_results)
# end

function f(mask)
    # run_solver(mask, initial_states, goals, N, horizon, num_sim_steps)
    t = run_solver(mask, initial_states, goals, N, horizon, num_sim_steps)
    # t0 = run_example(
    #     initial_states = initial_states,
    #     goals = goals,
    #     N = N,
    #     horizon = horizon,
    #     num_sim_steps = num_sim_steps,
    #     mask = [1,0,0,1]
    # )
    # return t["Player 1 Trajectory"][1][1][1]
end
test = f(mask)
println(test)
# grad = Zygote.forwarddiff((mask ->f(mask)), mask)
# println(grad)
println("done")