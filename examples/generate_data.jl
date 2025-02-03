using CSV
using DataFrames
using Plots

# Read the CSV file
file_path = "C:\\Users\\mouan\\MCP\\scripts\\agents_and_goals.csv"  # Replace with your actual file path
data = CSV.read(file_path, DataFrame)

N = 10

goals = mortar([[row.goal_x, row.goal_y] for row in eachrow(data[1:N,:])])

"Utility to create the road environment."
# function setup_road_environment(; lane_width = 2, num_lanes = 2, height = 50)
#     lane_centers = map(lane_idx -> (lane_idx - 0.5) * lane_width, 1:num_lanes)
#     vertices = [
#         [first(lane_centers) - 0.5lane_width, 0],
#         [last(lane_centers) + 0.5lane_width, 0],
#         [last(lane_centers) + 0.5lane_width, height],
#         [first(lane_centers) - 0.5lane_width, height],
#     ]

#     (; lane_centers, environment = PolygonEnvironment(vertices))
# end

function setup_environment(; length)
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
                # lane_preference = last(θi)

                norm_sqr(x[Block(ii)][1:2] - goals[Block(ii)]) +
                0.5norm_sqr(x[Block(ii)][3:4]) +
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
        #     x1, x2, x3 = blocks(x)
        #     # Players need to stay at least 2 m away from one another.
        #     norm_sqr(x1[1:2] - x2[1:2]) - 4
        # end
        constraints = []
        mapreduce(vcat, xs) do x
            player_states = [block for block in blocks(x)]
        #     # Players need to stay at least 2 m away from one another.
            for i in 1:N-1
                for j in i+1:N
                    push!(constraints, norm_sqr(player_states[i][1:2] - player_states[j][1:2]) - 4)
                end
            end 
        end
        return constraints
    end

    agent_dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -10, 0], ub = [Inf, Inf, 10, 10]),
        control_bounds = (; lb = [-5, -5], ub = [3, 3]),
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:N])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function run_lane_change_example(;
    # initial_state = mortar([[1.0, 1.0, 0.0, 1.0], [3.2, 0.9, 0.0, 1.0]]),
    initial_state = mortar([[row.x, row.y, row.vx, row.vy] for row in eachrow(data[1:N, :])]),
    horizon = 5,
    num_sim_steps = 100,
)
    # (; environment, lane_centers) =
    #     setup_road_environment(; num_lanes, lane_width, height)
    (; environment) = setup_environment(; length = 15)
    game = setup_trajectory_game(; environment)

    # Build a game. Each player has a parameter for lane preference.
    # P1 wants to stay in the left lane, and P2 wants to move from the
    # right to the left lane.
    lane_preferences = mortar([[10] for _ in 1:N])
    parametric_game = build_parametric_game(; game, horizon, params_per_player = 0)

    # Simulate the ground truth.
    turn_length = 2
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

    
    vis = true
    goal_xs = goals[1:2:end]
    goal_ys = goals[2:2:end]

    if vis    

        # Determine the global x and y limits
        all_x = []
        all_y = []

        for step in 1:num_sim_steps
            for i in 1:N
                pos = sim_steps[max_steps][step].substrategies[i].xs[1]
                push!(all_x, pos[1])
                push!(all_y, pos[2])
            end
        end

        x_min, x_max = minimum(all_x), maximum(all_x)
        y_min, y_max = minimum(all_y), maximum(all_y)

        # Create an animation with fixed axes
        anim = @animate for step in 1:num_sim_steps
            plt = plot(xlims=(x_min, x_max), ylims=(y_min, y_max), title="Trajectories")
            scatter!(plt, goal_xs, goal_ys, color=:black, marker=:star, label="Goal", markersize=6) # plot goals
            # Iterate over all players
            for i in 1:N
                x_traj = []
                y_traj = []

                # Collect trajectory up to the current step
                for s in 1:step
                    pos = sim_steps[max_steps][s].substrategies[i].xs[1]
                    push!(x_traj, pos[1])
                    push!(y_traj, pos[2])
                end
                
                plot!(plt, x_traj, y_traj, label="Player $i", marker=:circle)
            end
        end

        # Save the animation as a GIF
        gif(anim, "trajectory_animation.gif", fps=20)

    end
    
    # this is for an image 

    # # Initialize a plot
    # plt = plot()
    
    # # Iterate over players
    # for i in 1:N
    #     x_traj = []  # Store x positions
    #     y_traj = []  # Store y positions
        
    #     # Iterate over simulation steps
    #     for step in 1:num_sim_steps
    #         pos = sim_steps[max_steps][step].substrategies[i].xs[1]
    #         push!(x_traj, pos[1])
    #         push!(y_traj, pos[2])
    #     end
        
    #     # Plot the trajectory of the player
    #     plot!(plt, x_traj, y_traj, label="Player $i", marker=:circle)
    # end
    
    # # Show the plot
    # display(plt)
    


    #substrategies[i].xs[j] contains the state of player i's trajectory at time j (j=1,2,...,horizon)
    #println(sim_steps[max_steps][end].substrategies[10].xs) 
end
