import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PIL
import os
N=4

def get_trajectory(data):
    trajectories = {}
    goals = {}
    # loop through parameters in the dict
    for playerid in range(N): # in range(N)

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
    
    return trajectories, goals

def plot_traj(trajectories, goals, fname):
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
    plt.title(f"Trajectories for {fname}")
    #plt.legend()
    plt.grid(True)
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.gca().set_aspect('equal')
    plt.savefig(f"trajectory_{fname}.png")
    plt.close()

animate = False
def animate_traj(trajectories, goals, fname):
    # animation (thank you chat)
    if animate:
        # Define colors for each player
        colors = {}
        fig, ax = plt.subplots(figsize=(6, 6))

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

        ax.set_aspect("equal", adjustable="box")  # Keeps a square grid

        # Set plot limits (adjust based on data range)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Animated Player Trajectories")
        #ax.legend(loc="best")
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
        ani.save(f"trajectory_animation_{fname}.gif", writer="pillow", fps=10)

        # Show animation
        plt.show()
def lighten_color(color, factor=0.5):
    # Lighten the color by a factor. factor = 1 returns white
    r, g, b, a = color  # extract RGB values
    new_r = r + (1 - r) * factor
    new_g = g + (1 - g) * factor
    new_b = b + (1 - b) * factor
    return (new_r, new_g, new_b, a)

# old function to plot all gt trajectories in a directory
def plot_all(directory):
    for fname in os.listdir(directory):
        # first import the json file containing relevant players' trajectories
        if not fname.endswith('[1, 1, 1, 1].json'):
            continue
        f = open(directory +'/'+ fname)
        data = json.load(f)

        trajectories, goals = get_trajectory(data)
        plot_traj(trajectories, goals, fname)

# compare gt and predicted trajectories
def compare_gt_pred(gt_file, pred_file):

    # get gt and predicted trajectories
    f = open(gt_file)
    data = json.load(f)
    gt_trajectories, gt_goals = get_trajectory(data)
    f = open(pred_file)
    data = json.load(f)
    pred_trajectories, pred_goals = get_trajectory(data)

    colors = {} # stores unique colors for each player

    # plot gt and predicted trajectories
    plt.figure(figsize=(10, 10))  
    for i, (player, gt_trajectory) in enumerate(gt_trajectories.items()):
        # generate color for player
        if player not in colors:
            colors[player] = plt.get_cmap("tab10")(i)  # get player color
        
        pred_trajectory = pred_trajectories[player]
        x_vals = [step[0] for step in pred_trajectory]  
        y_vals = [step[1] for step in pred_trajectory]  
        x_goal, y_goal = pred_goals[player]  
        plt.plot(x_vals, y_vals, marker='o', color=colors[player], label=f'Pred Player {player}')
        plt.plot(x_goal, y_goal, '*', color=colors[player], markersize=10, label=f'gt Player {player} goal')

        lighter_color = lighten_color(colors[player]) # lightens the predicted trajectory color for gt
        x_vals = [step[0] for step in gt_trajectory]  
        y_vals = [step[1] for step in gt_trajectory]  
        x_goal, y_goal = gt_goals[player]  
        plt.plot(x_vals, y_vals, marker='o', color=lighter_color, label=f'gt Player {player}')
        
    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"GT and Predicted Trajectories. Predicted trajectory is darker.")
    plt.grid(True)
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.legend() 
    plt.gca().set_aspect('equal')
    plt.savefig(f"compare_gt_pred.png")
    plt.close()

compare_gt_pred('simulation_results_0[1, 1, 1, 1].json', 'simulation_results_0[1, 0, 0, 1].json')