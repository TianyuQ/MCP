import csv
import random
import math

def generate_agents_and_goals(N, bounds=(-5, 5), min_distance=2, max_velocity=0.5):
    agents = []
    goals = []

    def is_far_enough(pos, others, min_dist):
        return all(math.dist(pos, other) >= min_dist for other in others)

    while len(agents) < N:
        # Generate random initial position
        position = [round(random.uniform(bounds[0], bounds[1]), 4) for _ in range(2)]

        # Ensure it's far enough from existing agents
        if is_far_enough(position, [a[1:3] for a in agents], min_distance):
            # Generate random initial velocity
            velocity_magnitude = random.uniform(0, max_velocity)
            velocity_angle = random.uniform(0, 2 * math.pi)
            velocity = [
                round(velocity_magnitude * math.cos(velocity_angle), 4),
                round(velocity_magnitude * math.sin(velocity_angle), 4),
            ]

            # Add the agent's initial state with ID
            agent_id = len(agents) + 1
            agents.append([agent_id] + position + velocity)

    while len(goals) < N:
        # Generate a random goal position
        goal = [round(random.uniform(bounds[0], bounds[1]), 4) for _ in range(2)]

        # Ensure it's far enough from existing goals
        if is_far_enough(goal, goals, min_distance):
            goals.append(goal)

    return agents, goals

def save_to_csv(agents, goals, filename="agents_and_goals.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["id", "x", "y", "vx", "vy", "goal_x", "goal_y"])
        # Write data
        for agent, goal in zip(agents, goals):
            writer.writerow(agent + goal)

if __name__ == "__main__":
    # Parameters
    N = 10  # Number of agents
    bounds = (-5, 5)  # Area bounds
    min_distance = 2  # Minimum distance between agents and goals
    max_velocity = 0.5  # Maximum magnitude of initial velocity

    # Generate agents and goals
    agents, goals = generate_agents_and_goals(N, bounds, min_distance, max_velocity)

    # Save to CSV
    save_to_csv(agents, goals)
    print(f"Generated {N} agents and goals and saved to 'agents_and_goals.csv'")
