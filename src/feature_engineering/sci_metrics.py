"""Separation Consistency Index (SCI) metrics and calculations."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def compute_receiver_defender_distances(
    tracking_df, 
    receiver_id, 
    possession_team, 
    include_frames=None
):
    """
    Compute distances between a receiver and all defenders for each frame.

    Args:
        tracking_df (DataFrame): Tracking data for a single play
        receiver_id (int): NFL ID of the receiver
        possession_team (str): Team abbreviation of the offense
        include_frames (list, optional): List of frame IDs to include

    Returns:
        DataFrame: DataFrame with frameId and min_distance columns
    """
    # Filter to the receiver's positions
    receiver_positions = tracking_df[tracking_df['nflId'] == receiver_id]

    if include_frames is not None:
        receiver_positions = receiver_positions[
            receiver_positions['frameId'].isin(include_frames)
        ]

    # We need positions for every frame in the data
    frames = receiver_positions['frameId'].unique()
    results = []

    for frame in frames:
        # Get receiver position for this frame
        rec_pos = receiver_positions[
            receiver_positions['frameId'] == frame
        ][['x', 'y']].values

        if len(rec_pos) == 0:
            continue

        rec_pos = rec_pos[0].reshape(1, -1)  # Reshape for cdist

        # Get all defenders positions for this frame
        defenders = tracking_df[
            (tracking_df['frameId'] == frame) & 
            (tracking_df['teamAbbr'] != possession_team) & 
            (tracking_df['nflId'].notna())
        ]

        if len(defenders) == 0:
            continue

        def_pos = defenders[['x', 'y']].values

        # Compute distance from receiver to all defenders
        distances = cdist(rec_pos, def_pos, 'euclidean')
        min_distance = distances.min()

        results.append({
            'frameId': frame,
            'min_distance': min_distance
        })

    return pd.DataFrame(results)


def calculate_sci(distances_df, open_threshold=3.0):
    """
    Calculate Separation Consistency Index from distance time series.

    Args:
        distances_df (DataFrame): DataFrame with frameId and min_distance
        open_threshold (float): Threshold in yards to consider a receiver "open"

    Returns:
        dict: Dictionary with various SCI metrics
    """
    if len(distances_df) < 2:
        return {
            'mean_separation': np.nan,
            'std_separation': np.nan,
            'min_separation': np.nan, 
            'max_separation': np.nan,
            'pct_time_open': np.nan,
            'sci_score': np.nan
        }

    distances = distances_df['min_distance'].values

    mean_separation = np.mean(distances)
    std_separation = np.std(distances)
    min_separation = np.min(distances)
    max_separation = np.max(distances)
    pct_time_open = np.mean(distances >= open_threshold)

    # Consistency score: higher is better
    # We want high mean separation and low standard deviation
    # This formula gives higher score to consistent separation
    if std_separation == 0:
        sci_score = mean_separation * 10  # Perfect consistency
    else:
        sci_score = mean_separation / std_separation

    return {
        'mean_separation': mean_separation,
        'std_separation': std_separation,
        'min_separation': min_separation,
        'max_separation': max_separation,
        'pct_time_open': pct_time_open,
        'sci_score': sci_score
    }


def get_route_frames(tracking_df, snap_frame, pass_frame):
    """
    Extract frames from ball snap to pass release (the route running phase).

    Args:
        tracking_df (DataFrame): Tracking data for a specific play
        snap_frame (int): Frame ID when ball was snapped
        pass_frame (int): Frame ID when pass was released

    Returns:
        list: List of frame IDs during route running phase
    """
    all_frames = tracking_df['frameId'].unique()
    return [f for f in all_frames if snap_frame <= f <= pass_frame]


def find_event_frame(tracking_df, event):
    """
    Find frame number where a specific event occurs.

    Args:
        tracking_df (DataFrame): Tracking data for a specific play
        event (str): Event to find (e.g., 'ball_snap', 'pass_forward')

    Returns:
        int or None: Frame ID of the event, or None if not found
    """
    event_rows = tracking_df[tracking_df['event'] == event]
    if len(event_rows) > 0:
        return event_rows['frameId'].iloc[0]
    return None


def compute_sci_for_play(
    tracking_df, 
    player_play_df,
    plays_df,
    game_id, 
    play_id
):
    """
    Compute SCI for all route-running receivers in a specific play.

    Args:
        tracking_df (DataFrame): Tracking data
        player_play_df (DataFrame): Player-play participation data
        plays_df (DataFrame): Play information
        game_id (int): Game ID
        play_id (int): Play ID

    Returns:
        DataFrame: DataFrame with SCI metrics for each receiver in the play
    """
    # Get play data
    play_tracking = tracking_df[
        (tracking_df['gameId'] == game_id) & 
        (tracking_df['playId'] == play_id)
    ]

    if len(play_tracking) == 0:
        return pd.DataFrame()

    # Get possession team
    play_info = plays_df[
        (plays_df['gameId'] == game_id) & 
        (plays_df['playId'] == play_id)
    ]

    if len(play_info) == 0:
        return pd.DataFrame()

    possession_team = play_info['possessionTeam'].iloc[0]

    # Find snap and pass frames
    snap_frame = find_event_frame(play_tracking, 'ball_snap')
    pass_frame = find_event_frame(play_tracking, 'pass_forward')

    if snap_frame is None or pass_frame is None:
        # Not a pass play or missing key events
        return pd.DataFrame()

    # Get route-running receivers
    route_runners = player_play_df[
        (player_play_df['gameId'] == game_id) &
        (player_play_df['playId'] == play_id) &
        (player_play_df['wasRunningRoute'] == 1)
    ]

    if len(route_runners) == 0:
        return pd.DataFrame()

    # Get route frames (from snap to pass)
    route_frames = get_route_frames(play_tracking, snap_frame, pass_frame)

    # Calculate SCI for each route runner
    results = []

    for _, runner in route_runners.iterrows():
        receiver_id = runner['nflId']

        # Get distances to defenders for each frame
        distances = compute_receiver_defender_distances(
            play_tracking, 
            receiver_id, 
            possession_team,
            include_frames=route_frames
        )

        if len(distances) < 2:
            continue

        # Calculate SCI metrics
        sci_metrics = calculate_sci(distances)

        # Add to results
        results.append({
            'gameId': game_id,
            'playId': play_id,
            'nflId': receiver_id,
            'teamAbbr': runner['teamAbbr'],
            'wasTargettedReceiver': runner['wasTargettedReceiver'],
            'hadPassReception': runner['hadPassReception'],
            **sci_metrics
        })

    return pd.DataFrame(results)


def compute_sci_for_week(
    tracking_df, 
    player_play_df,
    plays_df,
    max_plays=None
):
    """
    Compute SCI for all plays in the tracking data.

    Args:
        tracking_df (DataFrame): Tracking data for one week
        player_play_df (DataFrame): Player-play participation data
        plays_df (DataFrame): Play information
        max_plays (int, optional): Maximum number of plays to process

    Returns:
        DataFrame: DataFrame with SCI metrics for all receivers
    """
    # Get unique play IDs in the tracking data
    game_play_pairs = tracking_df[['gameId', 'playId']].drop_duplicates()

    if max_plays is not None:
        game_play_pairs = game_play_pairs.head(max_plays)

    results = []

    for _, (game_id, play_id) in game_play_pairs.iterrows():
        play_sci = compute_sci_for_play(
            tracking_df, 
            player_play_df,
            plays_df,
            game_id, 
            play_id
        )

        if len(play_sci) > 0:
            results.append(play_sci)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def aggregate_player_sci(sci_df, players_df=None):
    """
    Aggregate SCI metrics by player across multiple plays.

    Args:
        sci_df (DataFrame): DataFrame with SCI metrics for many plays
        players_df (DataFrame, optional): Players info for adding names

    Returns:
        DataFrame: Aggregated SCI metrics by player
    """
    # Group by player and calculate average metrics
    player_sci = sci_df.groupby('nflId').agg({
        'mean_separation': 'mean',
        'std_separation': 'mean',  # Average of per-play standard deviations
        'min_separation': 'min',   # Minimum separation across all plays
        'max_separation': 'max',   # Maximum separation across all plays
        'pct_time_open': 'mean',
        'sci_score': 'mean',
        'gameId': 'count'          # Number of plays
    }).rename(columns={'gameId': 'play_count'})

    # Add global std (consistency across plays)
    player_means = sci_df.groupby(['nflId', 'gameId', 'playId'])[
        'mean_separation'
    ].mean().reset_index()

    play_std = player_means.groupby('nflId')['mean_separation'].std()
    player_sci['separation_std_across_plays'] = play_std

    # Add player information if available
    if players_df is not None:
        player_info = players_df[['nflId', 'displayName', 'position']]
        player_sci = player_sci.reset_index().merge(
            player_info, 
            on='nflId', 
            how='left'
        )
    else:
        player_sci = player_sci.reset_index()

    return player_sci 
