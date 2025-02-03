import json
import matplotlib.pyplot as plt

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

# Show plot
plt.show()