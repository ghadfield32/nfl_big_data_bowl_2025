import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def _label_height_zone(pass_length: float) -> str:
    if pass_length <= 5:
        return "low"
    if pass_length <= 20:
        return "mid"
    return "high"

def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    try:
        ft, inch = height_str.split('-')
        return int(ft) * 12 + int(inch)
    except (ValueError, AttributeError):
        return np.nan


def _height_to_inches(height_str: str | float) -> float:
    if pd.isna(height_str):
        return np.nan
    try:
        ft, inch = height_str.split("-")
        return int(ft) * 12 + int(inch)
    except (ValueError, AttributeError):
        return np.nan

def map_pass_dir(pass_loc: str | float, target_x: float) -> str:
    """
    Map to 'L', 'M', 'R' based on passLocationType if present,
    else use target_x with thirds of field (0-53.3).
    """
    # prefer explicit label
    if isinstance(pass_loc, str):
        pl = pass_loc.lower()
        if 'left' in pl: return 'L'
        if 'right' in pl: return 'R'
        if 'middle' in pl or 'center' in pl: return 'M'
    # fallback on x
    if pd.notna(target_x):
        if target_x < 120/3: return 'L'
        if target_x > 2*120/3: return 'R'
        return 'M'
    return np.nan

def calc_age(birth_date: str, game_date: str) -> float:
    """
    Return age in years on the game_date (YYYY-MM-DD).
    """
    bd = pd.to_datetime(birth_date, errors='coerce')
    gd = pd.to_datetime(game_date, errors='coerce')
    if pd.isna(bd) or pd.isna(gd): return np.nan
    return (gd - bd).days / 365.25

def calc_bmi(height_in: float, weight_lb: float) -> float:
    """
    BMI = weight(lb) / (height(in)^2) * 703
    """
    if pd.isna(height_in) or pd.isna(weight_lb): return np.nan
    return weight_lb / (height_in**2) * 703

def lookup_vert_jump_pct(df: pd.DataFrame, combine_csv: str) -> pd.DataFrame:
    """
    Join a CSV of {nflId, vert_jump_inches}, compute percentile within position.
    """
    comb = pd.read_csv(combine_csv)
    comb['vert_pct'] = comb.groupby('position')['vert_jump_inches'] \
                            .rank(pct=True)
    return df.merge(comb[['nflId','vert_pct']], on='nflId', how='left')

def lookup_draft_bucket(df: pd.DataFrame, draft_csv: str) -> pd.DataFrame:
    """
    Join a CSV of {nflId, draft_round}, bucket as R1, R2-3, Day3/UDFA.
    """
    dr = pd.read_csv(draft_csv)
    def bucket(r):
        if r == 1: return 'R1'
        if r in (2,3): return 'R2-3'
        return 'Day3/UDFA'
    dr['draft_bucket'] = dr['draft_round'].apply(bucket)
    return df.merge(dr[['nflId','draft_bucket']], on='nflId', how='left')



def _label_vertical_zone(ball_z: float, *, low: float = 6.0, mid: float = 10.0) -> float:
    """
    Assign an integer code to each “vertical zone” based on the ball’s z height (in feet)
    at pass_arrived.  

    0 = low (below `low` ft), 
    1 = mid (`low` ≤ ball_z < `mid`), 
    2 = high (≥ `mid`).

    - `low` = 6 ft is roughly chest/shoulder height.
    - `mid` = 10 ft is just under crossbar; most receivers max out ~11–12 ft.

    Returns np.nan if ball_z is NaN.
    """
    if pd.isna(ball_z):
        return np.nan
    if ball_z < low:
        return 0
    if ball_z < mid:
        return 1
    return 2


if __name__ == '__main__':
    print('Downloading dataset...')
    download_dataset(force=False)
    plays, players, player_play, games = load_base_data()
