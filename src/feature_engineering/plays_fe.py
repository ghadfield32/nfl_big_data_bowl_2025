"""
Feature engineering utilities that use ONLY the *plays* table.
"""

import pandas as pd
import numpy as np
from src.feature_engineering.utils import _label_height_zone, map_pass_dir


def plays_feature_engineering(plays_raw: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    One row per drop-back play, adding:
     - height_zone, air_yards_bin, pass_dir
     - is_play_action, is_rpo
     - preWP, wp_delta, postWP
     - epa_change
    """
    # 1️⃣ filter dropbacks
    df = plays_raw.loc[plays_raw["isDropback"]].copy()

    # 🔍 debug & drop missing passLength
    if debug:
        n_missing = df["passLength"].isna().sum()
        print(f"[DEBUG] passLength missing before binning: {n_missing}")
        print(df.loc[df["passLength"].isna(), ["gameId","playId"]].head(10))
    df = df.loc[df["passLength"].notna()]

    # 🔍 debug inputs
    if debug:
        print("[DBG] INPUT plays cols:", plays_raw.columns.tolist())
        print("[DBG] POST-FILTER cols:", df.columns.tolist())

    # 2️⃣ basic bins & flags
    # ── height_zone (low/mid/high) stays the same ──
    df["height_zone"] = df["passLength"].apply(_label_height_zone)

    # ── air_yards_bin: now explicitly handles x<0 as "behind" ──
    # negative (backward) throws → "behind", then short/mid/long as before.
    bins   = [-np.inf, 0, 5, 20, np.inf]
    labels = ['behind','short','mid','long']
    df["air_yards_bin"] = pd.cut(df["passLength"], bins=bins, labels=labels)

    if debug:
        n_missing_bin = df["air_yards_bin"].isna().sum()
        print(f"[DEBUG] air_yards_bin missing after binning: {n_missing_bin}")

    df["pass_dir"]       = df.apply(
                              lambda r: map_pass_dir(r.get("passLocationType"), r.get("targetX")),
                              axis=1)
    df["is_play_action"] = df["playAction"] == "play_action"
    df["is_rpo"]         = df.get("pff_runPassOption","").notna()

    # 3️⃣ WIN-PROB features
    df["preWP"]    = df["preSnapHomeTeamWinProbability"] / 100.0
    df["wp_delta"] = df["homeTeamWinProbabilityAdded"]  / 100.0
    df["postWP"]   = (df["preWP"] + df["wp_delta"]).clip(0,1)

    # 4️⃣ pick & rename
    keep = [
        "gameId","playId","passResult","passLength","targetX","targetY",
        "height_zone","air_yards_bin","pass_dir","is_play_action","is_rpo",
        "timeToThrow","passTippedAtLine","yardlineNumber","yardlineSide",
        "down","yardsToGo",
        # ▲ new fields here
        "preWP","wp_delta","postWP",
        "pff_passCoverage","expectedPointsAdded"
    ]
    out = (
        df[keep]
          .rename(columns={
               "yardlineNumber":"yardline_number",
               "yardlineSide":"yardline_side",
               "expectedPointsAdded":"epa_change"
           })
    )

    # ─── 3️⃣ handle spatial / direction nulls ───────────────────────────
    # flag whether there was a real target
    out["has_target"] = out["targetX"].notna()

    # fill missing target coordinates (model will see (0,0)+flag)
    out[["targetX","targetY"]] = out[["targetX","targetY"]].fillna(0.0)

    # pass_dir unknown when no target
    out["pass_dir"] = out["pass_dir"].fillna("unknown")

    # ─── 4️⃣ timing & tip flags ─────────────────────────────────────────
    # fill missing timeToThrow with median (or choose drop/mode)
    median_ttt = out["timeToThrow"].median()
    out["timeToThrow"] = out["timeToThrow"].fillna(median_ttt)
    # optional flag if you ever want to know which were imputed:
    out["time_imputed"] = out["timeToThrow"].isna()

    # missing passTippedAtLine → not tipped
    out["passTippedAtLine"] = out["passTippedAtLine"].fillna(False)


    # 5️⃣ tidy
    out["yardline_side"] = out["yardline_side"].fillna("middle")
    out["epa_change"]    = out["epa_change"].fillna(0.0)
    # 6️⃣ side_code
    side_code_unique = out["yardline_side"].unique()
    print('yardline_side unique values:', side_code_unique)

    # 6️⃣ side_code
    # fill na with unknown
    out["pff_passCoverage"] = out["pff_passCoverage"].fillna("unknown")
    print(out["pff_passCoverage"].value_counts(dropna=False))

    side_code_unique = out["pff_passCoverage"].unique()
    print('pff_passCoverage unique values:', side_code_unique)


    if debug:
        print("[DBG] OUTPUT cols:", out.columns.tolist())

    return out





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
    plays_fe = plays_feature_engineering(plays, debug=True)
    print('plays_fe columns:', plays_fe.columns.tolist())
    assert {"preWP","wp_delta","postWP"}.issubset(plays_fe.columns)

    # check sum of nulls
    play_nulls = plays_fe.isnull().sum()
    print('play_nulls:', play_nulls)
    assert play_nulls.sum() == 0
