using CSV
using DataFrames

# mask = [1, 0, 1, 1]
mask = [0.9, 0.9, 0.8, 0.1]
horizon = 2
num_sim_steps = 1
N = 4

file_path = "/home/tq877/Tianyu/player_selection/MCP/data_bak/scenario_0.csv"
data = CSV.read(file_path, DataFrame)
goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])

(; environment) = setup_road_environment(; length = 7)
game = setup_trajectory_game(; environment, N = 4)
parametric_game = build_parametric_game(; game, horizon=horizon, params_per_player = 6)

computed_results = run_solver(
        game,
        parametric_game,
        mask,
        initial_states,
        goals,
        N,
        horizon,
        num_sim_steps,
    )

# # println("computed_results", computed_results[1]) 
# # println(fieldnames(typeof(computed_results)))
# # println("computed_results", computed_results.last_solution.variables.x)

# # println(typeof(computed_results.last_solution.variables.x))
# # println(typeof(mask))
# # println("computed_results.last_solution.variables.x[1]: ", computed_results.last_solution.variables.x[1])

# function f(mask)
#     mask = ForwardDiff.value.(mask)  # ✅ Convert `Dual` numbers to `Float64`
#     # println("mask: ", mask)
#     computed_results = run_example(
#         game = game,
#         parametric_game = parametric_game,
#         initial_states = initial_states,
#         goals = goals,
#         N = N,
#         horizon = horizon,
#         num_sim_steps = num_sim_steps,
#         mask = mask,  # ✅ Ensure Float64 values are passed
#         save = false
#     )
#     return computed_results.last_solution.variables.x[1]  # Ensure numeric output
# end

# # ✅ Compute Jacobian safely
# # J = ForwardDiff.jacobian(f, mask)
# # J = ForwardDiff.gradient(f, mask)
# # J = only(Zygote.forwarddiff(f, mask))

# # println("Jacobian matrix:")
# # println(J)