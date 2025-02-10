using CSV
using DataFrames

mask = [1, 0, 0, 1]
horizon = 10
num_sim_steps = 1
N = 4

file_path = "/home/tq877/Tianyu/player_selection/MCP/data_bak/scenario_0.csv"
data = CSV.read(file_path, DataFrame)
goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])
initial_states = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])])



computed_traj = run_solver(mask, initial_states, goals, N, horizon, num_sim_steps)