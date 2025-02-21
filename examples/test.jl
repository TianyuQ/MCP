using CSV
using DataFrames

mask = [1, 1, 1, 1]
# mask = [0.9, 0.9, 0.8, 0.1]

file_path = "/home/tq877/Tianyu/player_selection/MCP/data_bak/scenario_0.csv"
data = CSV.read(file_path, DataFrame)
goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])

batch_targets = [0 for i in 1:N * horizon * d]

loss = run_solver(
    game, 
    parametric_game, 
    batch_targets, 
    initial_states, 
    goals, 
    N, 
    horizon, 
    num_sim_steps, 
    mask, 
)

# println("Loss: ", loss)

# parameter_value = pack_parameters(initial_states, mortar([vcat(goals[2 * (i - 1) + 1:2 * i], mask) for i in 1:N]))

# function gradient_test(parameter_value)
    
#     solution = MixedComplementarityProblems.solve(
#                 parametric_game,
#                 parameter_value;
#                 solver_type = MixedComplementarityProblems.InteriorPoint(),
#                 x₀ = [
#                     pack_trajectory(zero_input_trajectory(; game, horizon, initial_state=initial_states))
#                     zeros(sum(parametric_game.dims.λ) + parametric_game.dims.λ̃)
#                 ],
#             )

#     loss = solution.variables.x[1]
#     return loss
# end

# grad = only(Zygote.gradient(gradient_test, parameter_value))
# # grad = Zygote.forwarddiff(gradient_test, parameter_value)
# println("Gradient: ", grad)