import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
N=4

# first import the json file containing relevant players' trajectories
f = open("simulation_results_0[1, 1, 0, 0].json")

data = json.load(f)
trajectories = {}
goals = {}
# loop through parameters in the dict
for playerid in range(N):

    # remove irrelevant players and their values
    if not data[f"Player {playerid+1} Initial State"]:
        data.pop(f"Player {playerid+1} Initial State", None)
        data.pop(f"Player {playerid+1} Goal", None)
        data.pop(f"Player {playerid+1} Trajectory", None)
        data.pop(f"Player {playerid+1} Control", None)
        continue

    # get trajectories and goals of relevant players. doing only first sim_step for now
    trajectories[f"{playerid+1}"] = data[f"Player {playerid+1} Trajectory"][0]
    goals[f"{playerid+1}"] = data[f"Player {playerid+1} Goal"]

colors = {}

# now plot trajectories
for i, (player, trajectory) in enumerate(trajectories.items()):
    x_vals = [step[0] for step in trajectory]  
    y_vals = [step[1] for step in trajectory]  
    x_goal, y_goal = goals[player]  
    
    # assign a color from a colormap
    if player not in colors:
        colors[player] = plt.get_cmap("tab10")(i)  # Get distinct colors
    
    # plot player trajectory
    plt.plot(x_vals, y_vals, marker='o', color=colors[player], label=f'Player {player}')
    
    # plot goal with the same color
    plt.plot(x_goal, y_goal, '*', color=colors[player], markersize=10, label=f'Player {player} goal')

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Trajectories")
plt.legend()
plt.grid(True)

animate = True
# animation (thank you chat)
if animate:
    # Define colors for each player
    colors = {}
    fig, ax = plt.subplots(figsize=(8, 6))

    # Store line objects for each player
    lines = {}
    goal_markers = {}

    # Find the maximum trajectory length for animation frames
    max_steps = max(len(trajectory) for trajectory in trajectories.values())

    # Initialize plot elements
    for i, (player, trajectory) in enumerate(trajectories.items()):
        # Assign color
        if player not in colors:
            colors[player] = plt.get_cmap("tab10")(i)

        # Initialize empty plot line for player trajectory
        line, = ax.plot([], [], 'o-', color=colors[player], label=f'Player {player}')
        lines[player] = line

        # Plot goal position as a static star marker
        x_goal, y_goal = goals[player]
        goal_marker, = ax.plot(x_goal, y_goal, '*', color=colors[player], markersize=10, label=f'Player {player} goal')
        goal_markers[player] = goal_marker

    # Set plot limits (adjust based on data range)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Animated Player Trajectories")
    ax.legend()
    ax.grid(True)

    # Animation update function
    def update(frame):
        for player, trajectory in trajectories.items():
            # Get current trajectory slice up to the current frame
            x_vals = [step[0] for step in trajectory[:frame+1]]
            y_vals = [step[1] for step in trajectory[:frame+1]]
            
            # Update the player's trajectory line
            lines[player].set_data(x_vals, y_vals)
        return lines.values()

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=250, blit=True)

    # Show animation
    plt.show()
