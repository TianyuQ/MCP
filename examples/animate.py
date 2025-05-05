import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_method(
    method_file: str,
    N: int,
    get_trajectory,
    figsize=(6,6),
    interval: int = 200,
    save_path: str = None
):
    """
    Create and return a matplotlib.animation.FuncAnimation for one method JSON.

    Args:
        method_file: Path to the JSON file for this method.
        N: Number of players (for get_trajectory).
        get_trajectory: function(data, sim_steps="all") -> (trajectories, goals, masks).
        figsize: tuple, size of the figure.
        interval: int, ms between frames.
        save_path: if provided, path to write out the animation (e.g. .mp4 or .gif).

    Returns:
        anim: FuncAnimation object.
    """
    # Load data
    with open(method_file, 'r') as f:
        data = json.load(f)
    trajectories, goals, masks = get_trajectory(data, sim_steps="all")
    total_steps = len(trajectories["1"])

    # Precompute axis limits
    all_x = [pt[0] for traj in trajectories.values() for pt in traj]
    all_y = [pt[1] for traj in trajectories.values() for pt in traj]
    padding = 1.0
    x_min, x_max = min(all_x) - padding, max(all_x) + padding
    y_min, y_max = min(all_y) - padding, max(all_y) + padding

    # set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    title = ax.set_title("")

    # storage for our artists so blit=True can work
    artists = []

    # Colors (light shades) and mask logic reused
    color_ego       = "#66B3FF"
    color_other_on  = "#FF9999"
    color_other_off = "#99FF99"

    def init():
        # clear any existing artists
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        title.set_text(f"t = {frame/10:.1f} s")

        this_artists = []
        for p_str, traj in trajectories.items():
            p_id = int(p_str)

            # Determine current color based on mask
            midx = min(frame, len(masks)-1)
            if masks[midx][p_id-1] == 1:
                curr_color = color_ego if p_id==1 else color_other_on
            else:
                curr_color = color_ego if p_id==1 else color_other_off

            # Plot last 10 segments
            start_idx = max(0, frame-9)
            for idx in range(start_idx, frame):
                midx = min(idx, len(masks)-1)
                seg_color = (color_ego if p_id==1 else color_other_on) if masks[midx][p_id-1]==1 else (color_ego if p_id==1 else color_other_off)
                ln, = ax.plot(
                    [traj[idx][0], traj[idx+1][0]],
                    [traj[idx][1], traj[idx+1][1]],
                    color=seg_color, lw=2
                )
                this_artists.append(ln)

            # plot current point
            pt, = ax.plot(
                traj[frame][0], traj[frame][1],
                marker='o', color=curr_color, markersize=8
            )
            this_artists.append(pt)

        return this_artists

    anim = FuncAnimation(
        fig, update, frames=range(total_steps),
        init_func=init, interval=interval, blit=True
    )

    if save_path:
        # Requires ffmpeg for mp4 or imagemagick for gif
        anim.save(save_path, dpi=250)

    return anim
