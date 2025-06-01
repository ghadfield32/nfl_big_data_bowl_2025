from typing import Optional
import pandas as pd
from src.feature_engineering.utils import (
    convert_height_to_inches,
    # calc_age,    <— WE NO LONGER import calc_age here
    calc_bmi,
    lookup_vert_jump_pct,
    lookup_draft_bucket,
)

def players_feature_engineering(players_raw: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Clean player bio data and keep the original `position` column intact.

    Returns one row per `nflId` with, at minimum:
        nflId, position, height_inches, weight_numeric, displayName, birthDate (datetime64)
    """
    df = players_raw.copy()

    # 2. Parse birthDate as datetime, and report missing/unparseable counts
    # 2. Parse birthDate as datetime, compute median, and impute missing values
    if "birthDate" in df.columns:
        # Convert strings → datetime; unparseable strings become NaT
        df["birthDate"] = pd.to_datetime(df["birthDate"], errors="coerce")

        if debug:
            raw_nulls = df["birthDate"].isna().sum()
            print(f"[DEBUG] raw birthDate missing or unparseable: {raw_nulls}")

        # ─────────────────────────────────────────────────────────────────────
        # Compute median of existing (non-null) birthDates
        median_date = df["birthDate"].median()
        if debug:
            print(f"[DEBUG] median birthDate to impute = {median_date.date()}")

        # Create a flag marking which rows will be imputed
        df["birthDate_imputed_flag"] = df["birthDate"].isna().astype(int)

        # Fill all NaT entries with the median_date
        df["birthDate"] = df["birthDate"].fillna(median_date)

        if debug:
            post_impute_nulls = df["birthDate"].isna().sum()
            print(f"[DEBUG] post-imputation birthDate null count (should be 0): {post_impute_nulls}")
    else:
        raise KeyError(
            "No 'birthDate' column found. Downstream code needs it to compute age at game time."
        )


    # 3. Height → inches
    if "height" in df.columns and "height_inches" not in df.columns:
        df["height_inches"] = df["height"].map(convert_height_to_inches)
        if debug:
            missing_h = df["height_inches"].isna().sum()
            print(f"[DEBUG] height_inches missing after conversion: {missing_h}")
            bad_h = df.loc[df["height_inches"].isna(), "height"].unique()
            print(f"[DEBUG] height strings causing NaN inches: {bad_h[:10]}")
    else:
        if debug and "height" not in df.columns:
            print("[DEBUG] WARNING: 'height' column not found; skipping height_inches creation.")

    # 4. Weight → numeric
    if "weight" in df.columns and "weight_numeric" not in df.columns:
        df["weight_numeric"] = pd.to_numeric(df["weight"], errors="coerce")
        if debug:
            missing_w = df["weight_numeric"].isna().sum()
            print(f"[DEBUG] weight_numeric missing after conversion: {missing_w}")
            bad_w = df.loc[df["weight_numeric"].isna(), "weight"].unique()
            print(f"[DEBUG] weight strings causing NaN weight_numeric: {bad_w[:10]}")
    else:
        if debug and "weight" not in df.columns:
            print("[DEBUG] WARNING: 'weight' column not found; skipping weight_numeric creation.")

    # 5. Guarantee `position` exists
    if "position" not in df.columns:
        alt_map = {"pos": "position", "playerPosition": "position"}
        found = [c for c in alt_map if c in df.columns]
        if found:
            df.rename(columns={found[0]: "position"}, inplace=True)
            if debug:
                print(f"[DEBUG] renamed column '{found[0]}' → 'position'")
        else:
            raise KeyError(
                "No `position` column found after cleaning. Make sure to keep a canonical `position` field."
            )

    # 6. (Optional) Additional engineered features go here…
    # 7. Final debug summary…
    if debug:
        pre_cols = set(players_raw.columns)
        post_cols = set(df.columns)
        dropped = sorted(pre_cols - post_cols)
        added = sorted(post_cols - pre_cols)
        print(f"[DEBUG] players_fe ▶︎ dropped: {dropped}")
        print(f"[DEBUG] players_fe ▶︎ added  : {added}")

    return df








# ── Smoke Tests Using Real Data ──────────────────────────────────────────────
if __name__ == '__main__':
    from src.load_data.load_data import download_dataset, load_base_data
    # Download and load datasets
    download_dataset(force=False)
    plays, players, player_play, games = load_base_data()
    print('plays dataset columns:', plays.columns)
    print('players dataset columns:', players.columns)
    print('player_play dataset columns:', player_play.columns)
    print('games dataset columns:', games.columns)
    players_fe = players_feature_engineering(players, debug=True)
    print(players_fe[['nflId', 'height', 'height_inches', 'weight', 'weight_numeric']].head())

    # check sum of nulls
    play_nulls = players_fe.isnull().sum()
    print('play_nulls:', play_nulls)
    print('total rows:', len(players_fe))
    print('percentage of nulls:', play_nulls.sum() / len(players_fe))
    assert play_nulls.sum() == 0
