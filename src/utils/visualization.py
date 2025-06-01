
"""Visualization utilities for NFL Big Data Bowl 2025."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_football_field(
    ax=None,
    figsize=(12, 6.33),
    yard_lines=True,
    hash_marks=True,
    endzones=True
):
    """
    Plot a football field as a backdrop for visualizations.

    Args:
        ax (matplotlib.axes, optional): Axes to draw on
        figsize (tuple, optional): Figure size if ax is not provided
        yard_lines (bool): Whether to draw yard lines
        hash_marks (bool): Whether to draw hash marks
        endzones (bool): Whether to draw endzones

    Returns:
        matplotlib.axes: The axes with the field drawn
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Field dimensions
    field_width = 53.3  # yards
    field_length = 120  # yards (including endzones)

    # Draw the field outline
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_facecolor('forestgreen')

    # Draw the yard lines
    if yard_lines:
        for yard in range(10, 110, 10):
            ax.axvline(yard, color='white', lw=2)
            # Add yard number markers
            if yard > 0 and yard < 100:
                label = str(yard - 10 if yard < 60 else 100 - yard)
                ax.text(
                    yard, 
                    5, 
                    label, 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=12, 
                    fontweight='bold'
                )
                ax.text(
                    yard,
                    field_width - 5,
                    label, 
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=12,
                    fontweight='bold'
                )

    # Draw the hash marks
    if hash_marks:
        hash_width = 0.5
        hash_distance = 1

        # NCAA hash marks are 40 feet apart (13.33 yards)
        # NFL hash marks are 18 feet, 6 inches apart (6.11 yards)
        nfl_hash_yards = 6.11

        hash_top = (field_width / 2) + (nfl_hash_yards / 2)
        hash_bottom = (field_width / 2) - (nfl_hash_yards / 2)

        for yard in range(10, 110):
            ax.plot(
                [yard, yard],
                [hash_top - hash_distance, hash_top],
                color='white',
                lw=1
            )
            ax.plot(
                [yard, yard],
                [hash_bottom, hash_bottom + hash_distance],
                color='white',
                lw=1
            )

    # Draw the endzones
    if endzones:
        if yard_lines:
            left_patch = Rectangle(
                (0, 0),
                10,
                field_width,
                facecolor='navy',
                alpha=0.2,
                zorder=0
            )
            right_patch = Rectangle(
                (110, 0),
                10,
                field_width,
                facecolor='navy',
                alpha=0.2,
                zorder=0
            )
            ax.add_patch(left_patch)
            ax.add_patch(right_patch)

    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the axes frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def plot_receiver_defender_distances(
    distances_df,
    open_threshold=3.0,
    figsize=(12, 6),
    title=None
):
    """
    Plot the distance between a receiver and nearest defender over time.

    Args:
        distances_df (DataFrame): DataFrame with frameId and min_distance
        open_threshold (float): Threshold to consider receiver "open"
        figsize (tuple): Figure size
        title (str, optional): Plot title

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the distances
    ax.plot(
        distances_df['frameId'],
        distances_df['min_distance'],
        marker='o',
        linestyle='-',
        color='blue',
        alpha=0.7
    )

    # Add the open threshold
    ax.axhline(
        open_threshold,
        color='red',
        linestyle='--',
        label=f'Open Threshold ({open_threshold} yards)'
    )

    # Calculate SCI metrics to add to the plot
    from src.sci import calculate_sci
    metrics = calculate_sci(distances_df, open_threshold)

    # Add metrics in the legend
    ax.plot(
        [], 
        [], 
        ' ',
        label=f"Mean: {metrics['mean_separation']:.2f} yards"
    )
    ax.plot(
        [], 
        [], 
        ' ',
        label=f"Std: {metrics['std_separation']:.2f} yards"
    )
    ax.plot(
        [], 
        [], 
        ' ',
        label=f"% Open: {metrics['pct_time_open']*100:.1f}%"
    )
    ax.plot(
        [], 
        [], 
        ' ',
        label=f"SCI Score: {metrics['sci_score']:.2f}"
    )

    # Add open/closed shading
    ax.fill_between(
        distances_df['frameId'],
        distances_df['min_distance'],
        open_threshold,
        where=(distances_df['min_distance'] >= open_threshold),
        color='green',
        alpha=0.3,
        label='Open'
    )
    ax.fill_between(
        distances_df['frameId'],
        distances_df['min_distance'],
        open_threshold,
        where=(distances_df['min_distance'] < open_threshold),
        color='red',
        alpha=0.3,
        label='Covered'
    )

    # Customize the plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Receiver-Defender Separation Over Time', fontsize=14)

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Separation Distance (yards)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_player_route(
    tracking_df,
    player_id,
    field_ax=None,
    start_frame=None,
    end_frame=None,
    color='blue',
    alpha=0.8,
    field_kwargs=None
):
    """
    Plot a player's route on a football field.

    Args:
        tracking_df (DataFrame): Tracking data
        player_id (int): NFL ID of the player to plot
        field_ax (matplotlib.axes, optional): Axes with a field already drawn
        start_frame (int, optional): Starting frame ID
        end_frame (int, optional): Ending frame ID
        color (str): Color for the route line
        alpha (float): Alpha for the route line
        field_kwargs (dict, optional): Arguments for the field creation

    Returns:
        matplotlib.axes: The axes with the route drawn
    """
    # Filter to the player's trajectory
    player_track = tracking_df[tracking_df['nflId'] == player_id].copy()

    if start_frame is not None:
        player_track = player_track[player_track['frameId'] >= start_frame]
    if end_frame is not None:
        player_track = player_track[player_track['frameId'] <= end_frame]

    if len(player_track) == 0:
        raise ValueError(f"No tracking data found for player {player_id}")

    # Create field if not provided
    if field_ax is None:
        if field_kwargs is None:
            field_kwargs = {}
        field_ax = plot_football_field(**field_kwargs)

    # Get play direction
    play_direction = player_track['playDirection'].iloc[0]

    # Plot the route
    x_vals = player_track['x'].values
    y_vals = player_track['y'].values

    # We need to flip coordinates if the play is going right to left
    if play_direction == 'left':
        x_vals = 120 - x_vals

    field_ax.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=3)

    # Mark the start and end points
    field_ax.scatter(
        x_vals[0], 
        y_vals[0], 
        color=color, 
        s=100, 
        marker='o', 
        edgecolor='black', 
        linewidth=1.5
    )
    field_ax.scatter(
        x_vals[-1], 
        y_vals[-1], 
        color=color, 
        s=100, 
        marker='*', 
        edgecolor='black', 
        linewidth=1.5
    )

    return field_ax


def plot_play_animation(
    tracking_df, 
    game_id, 
    play_id, 
    start_frame=None, 
    end_frame=None,
    interval=100
):
    """
    Create an animation of a play.

    Args:
        tracking_df (DataFrame): Tracking data
        game_id (int): Game ID
        play_id (int): Play ID
        start_frame (int, optional): First frame to show
        end_frame (int, optional): Last frame to show
        interval (int): Milliseconds between frames

    Returns:
        matplotlib.animation.FuncAnimation: The animation
    """
    from matplotlib.animation import FuncAnimation

    # Filter to the specific play
    play_track = tracking_df[
        (tracking_df['gameId'] == game_id) & 
        (tracking_df['playId'] == play_id)
    ].copy()

    if len(play_track) == 0:
        raise ValueError(f"No data found for play {play_id} in game {game_id}")

    # Get play direction
    play_direction = play_track['playDirection'].iloc[0]

    # Determine frame range
    if start_frame is None:
        start_frame = play_track['frameId'].min()
    if end_frame is None:
        end_frame = play_track['frameId'].max()

    frames = range(start_frame, end_frame + 1)

    # Create the field
    fig, ax = plt.subplots(figsize=(12, 6.33))
    field_ax = plot_football_field(ax=ax)

    # Set up colors for teams
    teams = play_track['teamAbbr'].dropna().unique()
    if len(teams) >= 2:
        team_colors = {teams[0]: 'red', teams[1]: 'blue'}
        team_colors['ball'] = 'brown'
    else:
        team_colors = {t: 'blue' for t in teams}
        team_colors['ball'] = 'brown'

    # Set up the scatter plot for players
    scatters = {}
    for team in teams:
        team_data = play_track[
            (play_track['teamAbbr'] == team) & 
            (play_track['frameId'] == start_frame)
        ]

        # Handle coordinate flip for play direction
        x_vals = team_data['x'].values
        if play_direction == 'left':
            x_vals = 120 - x_vals

        scatter = field_ax.scatter(
            x_vals,
            team_data['y'].values,
            color=team_colors[team],
            s=100,
            alpha=0.7,
            edgecolor='black'
        )
        scatters[team] = scatter

    # Add the ball
    ball_data = play_track[
        (play_track['frameId'] == start_frame) & 
        play_track['nflId'].isna()
    ]

    if len(ball_data) > 0:
        # Handle coordinate flip for play direction
        ball_x = ball_data['x'].values[0]
        if play_direction == 'left':
            ball_x = 120 - ball_x

        ball_scatter = field_ax.scatter(
            ball_x,
            ball_data['y'].values[0],
            color=team_colors['ball'],
            s=50,
            alpha=1,
            edgecolor='black'
        )
        scatters['ball'] = ball_scatter

    # Add frame counter
    frame_text = field_ax.text(
        10, 
        5, 
        f"Frame: {start_frame}", 
        fontsize=12, 
        color='white', 
        bbox=dict(
            facecolor='black', 
            alpha=0.6, 
            edgecolor='none'
        )
    )

    # Animation update function
    def update(frame):
        for team in teams:
            team_data = play_track[
                (play_track['teamAbbr'] == team) & 
                (play_track['frameId'] == frame)
            ]

            # Handle coordinate flip for play direction
            x_vals = team_data['x'].values
            if play_direction == 'left':
                x_vals = 120 - x_vals

            scatters[team].set_offsets(
                np.column_stack([x_vals, team_data['y'].values])
            )

        # Update ball position
        ball_data = play_track[
            (play_track['frameId'] == frame) & 
            play_track['nflId'].isna()
        ]

        if len(ball_data) > 0 and 'ball' in scatters:
            # Handle coordinate flip for play direction
            ball_x = ball_data['x'].values[0]
            if play_direction == 'left':
                ball_x = 120 - ball_x

            scatters['ball'].set_offsets(
                [[ball_x, ball_data['y'].values[0]]]
            )

        # Update frame counter
        frame_text.set_text(f"Frame: {frame}")

        return list(scatters.values()) + [frame_text]

    # Create animation
    animation = FuncAnimation(
        fig, 
        update, 
        frames=frames, 
        interval=interval, 
        blit=True
    )

    return animation


def plot_sci_distribution(sci_df, metric='sci_score', figsize=(10, 6)):
    """
    Plot the distribution of a Separation Consistency Index metric.

    Args:
        sci_df (DataFrame): DataFrame with SCI metrics for players
        metric (str): Metric to plot
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        sci_df[metric].dropna(),
        kde=True,
        ax=ax,
        color='navy',
        alpha=0.7
    )

    metric_display_names = {
        'mean_separation': 'Mean Separation (yards)',
        'std_separation': 'Standard Deviation of Separation (yards)',
        'pct_time_open': 'Percentage of Time Open',
        'sci_score': 'Separation Consistency Index Score'
    }

    metric_name = metric_display_names.get(metric, metric)

    ax.set_title(f'Distribution of {metric_name}', fontsize=14)
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    # Add mean line
    mean_val = sci_df[metric].mean()
    ax.axvline(
        mean_val,
        color='red',
        linestyle='--',
        label=f'Mean: {mean_val:.2f}'
    )

    ax.legend()
    plt.tight_layout()

    return fig


def plot_top_receivers(
    sci_df, 
    metric='sci_score', 
    n=10, 
    min_plays=5, 
    figsize=(12, 8)
):
    """
    Plot the top N receivers by a specific SCI metric.

    Args:
        sci_df (DataFrame): Aggregated SCI metrics by player
        metric (str): Metric to sort by
        n (int): Number of top players to show
        min_plays (int): Minimum play count to qualify
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter receivers with minimum play count
    qualified = sci_df[sci_df['play_count'] >= min_plays].copy()

    if len(qualified) == 0:
        print(f"No receivers with at least {min_plays} plays.")
        return None

    # Sort by the metric and get top N
    top_n = qualified.sort_values(metric, ascending=False).head(n)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    bars = ax.barh(
        top_n['displayName'], 
        top_n[metric],
        color='navy',
        alpha=0.7
    )

    # Add position labels to bars
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height()/2,
            f"{top_n['position'].iloc[i]} ({top_n['play_count'].iloc[i]} plays)",
            va='center',
            fontsize=10
        )

    metric_display_names = {
        'mean_separation': 'Mean Separation (yards)',
        'std_separation': 'Standard Deviation of Separation (yards)',
        'pct_time_open': 'Percentage of Time Open',
        'sci_score': 'Separation Consistency Index Score'
    }

    metric_name = metric_display_names.get(metric, metric)

    # Add a title and labels
    ax.set_title(f'Top {n} Receivers by {metric_name}', fontsize=14)
    ax.set_xlabel(metric_name, fontsize=12)

    # Add a grid for the x-axis
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig 
