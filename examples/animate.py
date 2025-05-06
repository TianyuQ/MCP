import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

def animate_method(
    method_file: str,
    N: int,
    get_trajectory,
    figsize=(6, 6),
    interval: int = 200,
    save_path: str = None
):
    """
    Create and return a FuncAnimation showing receding-horizon trajectories.

    Args:
        method_file: path to the JSON file for this method.
        N: number of players.
        get_trajectory: function(data, sim_steps="all") -> (trajectories, goals, masks).
        figsize: size of the figure (width, height).
        interval: milliseconds between frames.
        save_path: if provided, saves the animation to this path (mp4 or gif).

    Returns:
        anim: FuncAnimation object.
    """
    # 1) Load data & extract
    with open(method_file, 'r') as f:
        data = json.load(f)
    trajectories, goals, masks = get_trajectory(data, sim_steps="all")
    total_steps = len(trajectories["1"])

    # 2) Compute axis limits (with padding)
    all_x = [pt[0] for traj in trajectories.values() for pt in traj]
    all_y = [pt[1] for traj in trajectories.values() for pt in traj]
    pad = 1.0
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    # 3) Setup figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    title = ax.set_title("")

    # 4) Prepare per-player unique colors for IDs/goals
    player_ids = sorted(int(k) for k in trajectories.keys())
    cmap = plt.cm.get_cmap('tab10', len(player_ids))
    player_colors = {pid: cmap(i) for i, pid in enumerate(player_ids)}

    # 5) Mask‐color definitions
    color_ego       = "#66B3FF"  # light blue for Player 1
    color_other_on  = "#FF9999"  # light red
    color_other_off = "#99FF99"  # light green

    def init():
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        title.set_text("")
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        title.set_text(f"t = {frame/10:.1f} s")

        artists = []

        for p_str, traj in trajectories.items():
            p_id = int(p_str)
            p_color = player_colors[p_id]

            # 5a) current‐step mask color
            mstep = min(frame, len(masks)-1)
            if masks[mstep][p_id-1] == 1:
                curr_col = color_ego if p_id==1 else color_other_on
            else:
                curr_col = color_ego if p_id==1 else color_other_off

            # 5b) plot last 10 segments
            start_idx = max(0, frame-9)
            for idx in range(start_idx, frame):
                midx = min(idx, len(masks)-1)
                seg_col = (color_ego if p_id==1 else color_other_on) \
                          if masks[midx][p_id-1]==1 else (color_ego if p_id==1 else color_other_off)
                ln, = ax.plot(
                    [traj[idx][0], traj[idx+1][0]],
                    [traj[idx][1], traj[idx+1][1]],
                    color=seg_col, lw=2
                )
                artists.append(ln)

            # 5c) plot current position
            if frame < len(traj):

                #annotate player ID at current position
                txt = ax.annotate(
                    str(p_id),
                    xy=(traj[frame][0], traj[frame][1]),
                    xytext=(3, 3),                # offset 3 points right & up
                    textcoords="offset points",
                    fontsize=12,
                    color=p_color,
                    ha='left',
                    va='bottom'
                )
                artists.append(txt)

            # 5d) plot goal & label
            gx, gy = goals[p_str]
            star, = ax.plot(
                gx, gy,
                marker='*', markersize=12,
                color=p_color
            )
            artists.append(star)
            gtxt = ax.annotate(
                str(p_id),
                xy=(gx, gy),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=12,
                color=p_color,
                ha='left',
                va='bottom'
            )
            artists.append(gtxt)
            pt, = ax.plot(
                    traj[frame][0], traj[frame][1],
                    marker='o', color=curr_col, markersize=8
                )
            artists.append(pt)

        return artists

    anim = FuncAnimation(
        fig, update,
        frames=range(total_steps),
        init_func=init,
        interval=interval,
        blit=True
    )

    if save_path:
        anim.save(save_path, dpi=250, fps=15)

    return anim
