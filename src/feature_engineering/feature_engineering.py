"""
Data loading and preprocessing for NFL Big Data Bowl 2025,
including assembly of ML-ready dataset for WR/TE/CB pass-contest analysis.
"""
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
import zipfile
from typing import Iterable
from src.load_data.load_data import load_tracking_data
from src.feature_engineering.utils import _label_height_zone, convert_height_to_inches, _height_to_inches
from src.feature_engineering.plays_fe import plays_feature_engineering
from src.feature_engineering.players_fe import players_feature_engineering
from src.feature_engineering.player_play_fe import player_play_feature_engineering
from src.feature_engineering.games_fe import tracking_feature_engineering
from src.feature_engineering.column_schema import ColumnSchema 

# Automatically locate and load the nearest .env file
dotenv_path = find_dotenv()
if not dotenv_path:
    print("WARNING: .env not found—make sure KAGGLE_USERNAME/KEY are set!")
else:
    load_dotenv(dotenv_path)
    print(f"[dotenv] loaded from {dotenv_path}")

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate Kaggle API
a_pi = KaggleApi()
a_pi.authenticate()

DATA_DIR = "data/nfl-bdb-2025"
COMPETITION = "nfl-big-data-bowl-2025"

# ── NEW: save/load helpers ───────────────────────────────────────────────
def save_fe_datasets(df: pd.DataFrame, path: str, file_format: str = 'parquet') -> None:
    """
    Save the DataFrame to disk at `path`.
    Supports 'parquet' or 'csv' via `file_format`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if file_format == 'parquet':
        df.to_parquet(path, index=False)
    elif file_format == 'csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

def load_fe_dataset(path: str, file_format: str = 'parquet') -> pd.DataFrame:
    """
    Load the DataFrame from disk at `path`.
    """
    if file_format == 'parquet':
        return pd.read_parquet(path)
    elif file_format == 'csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")
# ── end new helpers ─────────────────────────────────────────────────────

# Engineering each dataset before Join--------------------------


# feature adding functions:
def add_kinematic_features(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Add sideline_dist and placeholder pass_rush_sep.
    """
    if "y" in df.columns:
        y_col = "y"
    elif "rcv_y" in df.columns:
        y_col = "rcv_y"
    else:
        raise KeyError("Neither 'y' nor 'rcv_y' in df.columns")

    df["sideline_dist"] = np.minimum(df[y_col], 53.3 - df[y_col])
    df["pass_rush_sep"] = np.nan  # placeholder

    if debug:
        print("👟 after add_kinematic_features:", df.columns.tolist())

    return df




def add_contextual_features(df, plays_ctx, player_play, *, debug=False):
    """
    Merge play-level context (timeToThrow, tipped flag, yard-line info, etc.)
    into the main DF.  Robust to camelCase vs snake_case and to pre-existing
    yard-line cols in `df` (avoids *_x / *_y pitfalls).
    """
    if debug:
        # 1) all dropback plays with missing side in plays_fe:
        missing_in_ctx = (
            plays_ctx[plays_ctx["yardline_side"].isna()]
              .loc[:, ["gameId","playId"]]
              .drop_duplicates()
        )
        print(f"[DEBUG] plays_fe dropbacks missing side: {len(missing_in_ctx)} plays")

        # 2) which plays actually made it into this df so far:
        plays_in_df = df.loc[:, ["gameId","playId"]].drop_duplicates()
        print(f"[DEBUG] plays entering ctx‐merge:      {len(plays_in_df)} unique plays")

        # 3) how many of those missing‐side plays survived into df?
        common = plays_in_df.merge(missing_in_ctx, on=["gameId","playId"])
        print(f"[DEBUG] missing‐side plays in df:     {len(common)} plays")
        print("[DEBUG] sample of those plays:", common.sample(min(10,len(common))))

        # 4) which missing‐side plays got dropped before this point?
        dropped = missing_in_ctx.merge(plays_in_df, on=["gameId","playId"], how="left", indicator=True)
        dropped = dropped[dropped["_merge"]=="left_only"]
        print(f"[DEBUG] missing‐side plays dropped:    {len(dropped)} plays")
        print("[DEBUG] sample dropped plays:", dropped.sample(min(10,len(dropped))))

    if debug:
        print("DEBUG: df cols BEFORE ctx merge:", list(df.columns)[:25], "…")
        print("DEBUG: plays_ctx cols:", list(plays_ctx.columns)[:25], "…")

    # 1. Build clean context table…
    col_map = {
        "yardline_number": ["yardline_number", "yardlineNumber"],
        "yardline_side"  : ["yardline_side",   "yardlineSide"],
    }
    base_cols = ["gameId", "playId", "timeToThrow", "passTippedAtLine",
                 "down", "yardsToGo"]

    ctx_cols = base_cols + [
        next(c for c in aliases if c in plays_ctx.columns)
        for aliases in col_map.values()
    ]
    play_ctx = (
        plays_ctx[ctx_cols]
        .rename(columns={
            "passTippedAtLine": "pass_tipped_at_line",
            **{raw: canon
               for canon, aliases in col_map.items()
               for raw in aliases if raw in ctx_cols}
        })
    )
    play_ctx["was_tipped"] = play_ctx["pass_tipped_at_line"] == 1
    play_ctx.drop(columns="pass_tipped_at_line", inplace=True)

    # 2. Merge; coalesce yardline_*
    df = df.merge(
        play_ctx,
        on=["gameId", "playId"],
        how="left",
        suffixes=("", "_ctx")
    )
    for col in ("yardline_side", "yardline_number"):
        if col not in df.columns and f"{col}_ctx" in df.columns:
            df.rename(columns={f"{col}_ctx": col}, inplace=True)
        elif f"{col}_ctx" in df.columns:
            df[col] = df[col].combine_first(df[f"{col}_ctx"])
            df.drop(columns=f"{col}_ctx", inplace=True)

    if debug:
        print("DEBUG: df cols AFTER ctx merge:", list(df.columns)[:25], "…")

    # ── FRAME-LEVEL NaN check ─────────────────────────────────────────────
    if debug:
        total_rows_missing = df["yardline_side"].isna().sum()
        unique_plays_missing = (
            df[df["yardline_side"].isna()]
              .loc[:, ["gameId","playId"]]
              .drop_duplicates()
              .shape[0]
        )
        print(f"[DEBUG] frame-level rows with yardline_side==NaN: {total_rows_missing}")
        print(f"[DEBUG] unique plays missing side_code:          {unique_plays_missing}")

    # 3. Derived features
    df["contest_thresh"] = df["rcv_height"].apply(
        lambda h: max(1.0, 0.2 * h / 12) if pd.notna(h) else 1.0
    )
    df["is_contested"]    = df["sep_receiver_cb"] <= df["contest_thresh"]
    df["caught_flag"]     = df["passResult"].isin(("C", "S"))

    cov_cols = [c for c in df if c.startswith("cov_")]
    if cov_cols:
        df["cov_zone"] = df.get("cov_zone", 0)
        df["contested_success"] = (
            df["is_contested"] & df["caught_flag"] & (1 - 0.1 * df["cov_zone"])
        )
    else:
        df["contested_success"] = df["is_contested"] & df["caught_flag"]

    # ── FIX: guarantee qb_was_hit exists without overwriting ───────────────
    if "qb_was_hit" not in df.columns:
        df["qb_was_hit"] = False

    return df



def add_matchup_features(df, plays, player_play):
    """
    Add route and coverage one-hots, then guarantee the full cov_* set.
    """
    # 1. route one-hots (unchanged) …
    if 'routeRan' in player_play.columns:
        src = 'routeRan'
    elif 'route' in player_play.columns:
        src = 'route'
    else:
        src = None

    if src:
        routes = (
            player_play[['gameId','playId','nflId',src]]
            .rename(columns={src:'route','nflId':'receiverId'})
        )
        df = df.merge(routes, on=['gameId','playId','receiverId'], how='left')
        df = pd.concat([df, pd.get_dummies(df['route'],prefix='route')], axis=1)
    else:
        df['route'] = np.nan

    # 2. coverage one-hots
    if 'pff_passCoverage' in plays.columns:
        cov = (
            plays[['gameId','playId','pff_passCoverage']]
            .rename(columns={'pff_passCoverage':'pass_coverage'})
        )
        df = df.merge(cov, on=['gameId','playId'], how='left')
        df = pd.concat([df, pd.get_dummies(df['pass_coverage'], prefix='cov')], axis=1)
        df.drop(columns=['pass_coverage'], inplace=True)

    # ── GUARANTEE FULL coverage set ───────────────────────────
    schema = ColumnSchema()
    expected = [c for c in schema.nominal_cols() if c.startswith('cov_')]
    missing  = set(expected) - set(df.columns)
    for col in missing:
        df[col] = 0

    return df



# ---------- PATCH #2  (new helper) ----------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder temporal metrics.
    """
    df["sep_stddev"] = 0.0
    df["hypo_sep"]   = df["min_other_sep"] if "min_other_sep" in df.columns else np.nan
    return df




def get_defender_snapshots(plays_df, tracking_weeks):
    """
    Return one row per (gameId, playId, defenderId) at pass_arrived,
    with defender x,y coordinates.
    """
    # 1. Stack the pass_arrived frames
    arrived_list = []
    for wk in tracking_weeks:
        tr = load_tracking_data(week=wk)[
            ['gameId','playId','nflId','club','x','y','event']
        ]
        arrived = tr[tr['event'] == 'pass_arrived'].copy()
        arrived_list.append(arrived)
    arrived = pd.concat(arrived_list, ignore_index=True)

    # 2. Attach possessionTeam so we know offense vs defense
    ctx = plays_df[['gameId','playId','possessionTeam']]
    arrived = arrived.merge(ctx, on=['gameId','playId'], how='left')

    # 3. Filter out offensive club → keep defenders only
    defenders = arrived[arrived['club'] != arrived['possessionTeam']].copy()
    # rename columns for clarity
    defenders.rename(columns={'nflId':'defenderId','x':'def_x','y':'def_y'}, inplace=True)

    return defenders[['gameId','playId','defenderId','def_x','def_y']]



# ---------------------------------------------------------------------------
# 2.  contest flags: use correct "caught" codes
# ---------------------------------------------------------------------------
def add_contest_features(df: pd.DataFrame,
                         contest_threshold: float = 1.0,
                         caught_codes: tuple[str, ...] = ("C", "S")) -> pd.DataFrame:
    """
    - is_contested      : sep ≤ threshold (yards) **and** a defender present
    - contested_success : contested **and** pass caught  (passResult in caught_codes)
    """
    df["is_contested"] = df["sep_receiver_cb"] <= contest_threshold
    df["caught_flag"]  = df["passResult"].isin(caught_codes)
    df["contested_success"] = df["is_contested"] & df["caught_flag"]
    return df




def make_vsrg_summary(df: pd.DataFrame, min_attempts: int = 5) -> pd.DataFrame:
    """
    Build a tidy summary table of contested-catch success by receiver × height_zone.
    Drops any receiver–zone with fewer than `min_attempts`.
    """
    base = (
        df.groupby(["receiverId", "height_zone"], observed=False)
          .agg(attempts=('is_contested','sum'),
               successes=('contested_success','sum'))
          .assign(vsrg_rate=lambda d: d['successes']/d['attempts'])
          .reset_index()
    )

    base = base[base['attempts'] >= min_attempts].copy()
    base['vsrg_zone'] = (base['vsrg_rate'] * 100).round().astype(int)
    return base


def add_vsrg_overall(
    vsrg_long: pd.DataFrame,
    equal_weight: bool = False
) -> pd.DataFrame:
    """
    Collapse the long VSRG table to one overall score per receiver:
      - default: attempt-weighted average of vsrg_rate
      - equal_weight=True: simple mean of zone rates
    Returns a DataFrame with columns ['receiverId','vsrg_overall'].
    """
    if equal_weight:
        agg = lambda g: g['vsrg_rate'].mean()
    else:
        agg = lambda g: (g['vsrg_rate'] * g['attempts']).sum() / g['attempts'].sum()

    overall = (
        vsrg_long
        .groupby('receiverId')
        .apply(agg)
        .mul(100)
        .round()
        .astype(int)
        .rename('vsrg_overall')
        .reset_index()
    )
    return overall


def get_qb_snapshots(
    tracking_weeks: Iterable[int],
    players_df: pd.DataFrame,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Return one row per (gameId, playId) with QB x,y coordinates at `pass_arrived`.
    """

    # 🛑 1. quick guard: if a DataFrame slipped in, fail fast
    from pandas import DataFrame
    if isinstance(tracking_weeks, DataFrame):
        raise TypeError(
            "get_qb_snapshots() expected an iterable of week numbers "
            "but received a DataFrame.  Pass the *list* of weeks, e.g. range(1,10)."
        )

    # … everything below is unchanged …
    pos_col_candidates = ["position", "pos", "playerPosition"]
    pos_col = next((c for c in pos_col_candidates if c in players_df.columns), None)
    if pos_col is None:
        raise KeyError(f"None of {pos_col_candidates} found in players_df")

    qb_ids = set(players_df.loc[players_df[pos_col] == "QB", "nflId"])
    if debug:
        print(f"get_qb_snapshots ▶︎ using `{pos_col}` — {len(qb_ids)} QBs")

    frames = []
    for wk in tracking_weeks:
        tr = load_tracking_data(week=wk, nrows=None)[
            ["gameId", "playId", "nflId", "x", "y", "event"]
        ]
        mask = (tr["event"] == "pass_arrived") & (tr["nflId"].isin(qb_ids))
        frames.append(tr.loc[mask, ["gameId", "playId", "x", "y"]])

    qb_snap = (
        pd.concat(frames, ignore_index=True)
          .rename(columns={"x": "qb_x", "y": "qb_y"})
          .drop_duplicates(subset=["gameId", "playId"])
    )
    return qb_snap

def add_pass_rush_sep(
    df: pd.DataFrame,
    plays_df: pd.DataFrame,
    players_df: pd.DataFrame,
    tracking_weeks: Iterable[int],
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Compute min distance between any defender and QB at pass_arrived.
    Ensures a single 'pass_rush_sep' column by dropping any placeholder
    before merging in the real values.
    """
    # ── DROP placeholder to avoid pandas suffixes ─────────────────────────────
    df = df.drop(columns=["pass_rush_sep"], errors="ignore")

    # 1. get QB positions at pass_arrived
    qb_snap = get_qb_snapshots(tracking_weeks, players_df, debug=debug)
    # 2. get defender positions at pass_arrived
    defs    = get_defender_snapshots(plays_df, tracking_weeks)

    # 3. compute rush distance
    merged = defs.merge(qb_snap, on=["gameId","playId"], how="left")
    merged["rush_dist"] = np.hypot(
        merged["def_x"] - merged["qb_x"],
        merged["def_y"] - merged["qb_y"]
    )

    # 4. keep only the minimum separation per play
    min_sep = (
        merged
        .groupby(["gameId","playId"], observed=False)["rush_dist"]
        .min()
        .reset_index()
        .rename(columns={"rush_dist": "pass_rush_sep"})
    )

    # 5. merge into the main DF
    out = df.merge(min_sep, on=["gameId","playId"], how="left")

    if debug:
        print("⚡ after add_pass_rush_sep:", out.columns.tolist())

    return out




def feature_engineering(
    plays,
    players_phys,
    player_pp,
    games,
    *,
    tracking_weeks=range(1, 10),
    debug=False,
    save_path=None,
    save_format='parquet',
):
    plays_fe     = plays_feature_engineering(plays, debug=debug)
    player_pp_fe = (
        player_play_feature_engineering(player_pp, debug=debug)
        .rename(columns={"nflId": "receiverId"})
        [["gameId", "playId", "receiverId", "qb_was_hit"]]
    )
    tracking_df  = tracking_feature_engineering(tracking_weeks, debug=debug)

    if debug:
        print("▶ plays_fe cols:", plays_fe.columns.tolist())
        print("▶ player_pp_fe cols:", player_pp_fe.columns.tolist())

    # ← merge on only gameId & playId
    drops = plays_fe.merge(
        player_pp_fe,
        on=["gameId", "playId"],
        how="inner"
    )
    if debug:
        print(f"[DEBUG] after plays↔player_pp merge: {drops.shape}")

    df = drops.merge(
        tracking_df,
        on=["gameId", "playId", "receiverId"],
        how="left"
    )

    # ── NEW: depth-zone labeling ────────────────────────────────────────
    if "passLength" in df.columns:
        df["height_zone"] = df["passLength"].apply(_label_height_zone)
    else:
        raise KeyError("Expected 'passLength' to exist for height_zone labeling.")

    # 2. rename raw tracking → visual fields
    df = df.rename(
        columns={
            "rcv_x": "x",
            "rcv_y": "y",
            "club":  "rcv_club",
        }
    )
    df.rename(columns={"dist_cb": "sep_receiver_cb"}, inplace=True)

    # 3. sanity check for plotting keys
    for col in ("x", "y", "rcv_club", "cb_x", "cb_y", "cb_club"):
        if col not in df.columns:
            raise KeyError(f"After renaming, expected column '{col}' but didn't find it.")

    # 4. contest + physical + matchup + temporal
    df = add_contest_features(df)
    # ── ensure passTippedAtLine is real boolean ───────────────────────────────
    df["passTippedAtLine"] = df["passTippedAtLine"].astype(bool)

    # --- DEBUG: before receiver merge ---
    if debug:
        print(
            f"[DEBUG] before rcv merge: {df.shape[0]} rows,"
            f" {df['receiverId'].nunique()} distinct receiverIds"
        )
    players_selected_cols = players_feature_engineering(players_phys, debug=debug)[
        ["nflId", "height_inches", "weight_numeric", "position","birthDate"]
    ]

    # Merge receiver
    df = df.merge(
        players_selected_cols.rename(
            columns={
                "nflId": "receiverId",
                "height_inches": "rcv_height",
                "weight_numeric": "rcv_weight",
                "position": "rcv_position"
            }
        ),
        on="receiverId", how="left"
    )


    if debug:
        null_rcv = df['rcv_height'].isna().sum()
        print(f"[DEBUG] after rcv merge: {null_rcv} missing rcv_height")
        print(
            "[DEBUG] sample rows with missing rcv_height:",
            df.loc[df['rcv_height'].isna(), ['gameId','playId','receiverId']]
            .head(5)
            .to_dict('records')
        )

    # --- DEBUG: before cb merge ---
    if debug:
        print(
            f"[DEBUG] before cb merge: {df.shape[0]} rows,"
            f" {df['cbId'].nunique()} distinct cbIds,"
            f" {df['cbId'].isna().sum()} missing cbId entries"
        )
    # ── DROP plays with no defender snapshot ─────────────────────────
    if debug:
        print(
            f"[DEBUG] dropping {df['cbId'].isna().sum()} rows with no defender snapshot"
        )
    df = df[df['cbId'].notna()]

    df = df.merge(
        players_feature_engineering(players_phys, debug=debug)
        .rename(
            columns={
                "nflId": "cbId",
                "height_inches": "cb_height",
                "weight_numeric": "cb_weight"
            }
        ),
        on="cbId", how="left"
    )
    if debug:
        null_cb_h = df['cb_height'].isna().sum()
        null_cb_w = df['cb_weight'].isna().sum()
        print(
            f"[DEBUG] after cb merge: {null_cb_h} missing cb_height,"
            f" {null_cb_w} missing cb_weight"
        )
        print(
            "[DEBUG] sample rows with missing cb_height:",
            df.loc[df['cb_height'].isna(), ['gameId','playId','cbId']]
            .head(5)
            .to_dict('records')
        )

    # placeholder kinematics
    df = add_kinematic_features(df, debug=debug)

    # context, matchup, temporal
    df = add_contextual_features(df, plays_fe, player_pp, debug=debug)
    df = add_matchup_features(df, plays, player_pp)
    df = add_temporal_features(df)

    # … all your existing code up to the final df = add_pass_rush_sep(…) …
    df = add_pass_rush_sep(
        df, plays, players_phys, tracking_weeks, debug=debug
    )

    # ── FINAL DEBUG: before adding VSRG ────────────────────────────────────
    if debug:
        print("🔍 before VSRG merge:", df.shape, "columns:", df.columns.tolist())

    # ── NEW BLOCK: VSRG SUMMARY & OVER-EXPECTATION ─────────────────────────
    # 1. Build the long VSRG table (drops receivers with < min_attempts)
    vsrg_long = make_vsrg_summary(df, min_attempts=5)
    # 2. Collapse to one score per receiver
    vsrg_ovrl = add_vsrg_overall(vsrg_long)

    # 3. Merge overall score back into main DF
    df = df.merge(vsrg_ovrl, on="receiverId", how="left")

    # 4. Debug: count missing vsrg_overall
    if debug:
        missing_overall = df["vsrg_overall"].isna().sum()
        print(f"[DEBUG] vsrg_overall missing after merge: {missing_overall}")

    # 5. Fill missing with league-average (or choose another baseline)
    baseline = df["vsrg_overall"].mean()
    df["vsrg_overall"] = df["vsrg_overall"].fillna(baseline)
    if debug:
        print(f"[DEBUG] filled vsrg_overall NaNs with baseline (mean = {baseline:.2f})")

    # 6. Compute over-expectation: observed success minus expected rate
    df["vsrg_oe"] = np.where(
        df["is_contested"],
        df["contested_success"].astype(float) - df["vsrg_overall"].div(100),
        0.0
    )

    # 7. Debug: ensure no NaNs remain in vsrg_oe
    if debug:
        missing_oe = df["vsrg_oe"].isna().sum()
        print(f"[DEBUG] vsrg_oe missing after computation: {missing_oe}")

    # ── END NEW BLOCK ───────────────────────────────────────────────────────

    # ── SAVE if requested ───────────────────────────────────────────────────
    if save_path:
        # write out exactly this DataFrame
        save_fe_datasets(df, save_path, save_format)
        if debug:
            print(f"[DEBUG] feature_engineering: saved DataFrame to {save_path}")
    if debug:
        print("[DBG][feature_engineering] final columns:", df.columns.tolist())
        assert {"preWP","wp_delta","postWP"}.issubset(df.columns), \
               "One of the win-prob columns is missing!"

    return df






if __name__ == '__main__':
    from src.load_data.load_data import download_dataset, load_base_data
    from src.feature_engineering.column_schema import ColumnSchema 
    schema = ColumnSchema()
    # Single-bucket constants (immutable copies)
    INFO_NON_ML= schema.info_non_ml()
    NOMINAL = schema.nominal_cols()
    ORDINAL = schema.ordinal_cols()
    NUMERICAL = schema.numerical_cols()
    TARGET = schema.target_col()

    print("INFO_NON_ML", INFO_NON_ML)
    print("NOMINAL", NOMINAL)
    print("ORDINAL", ORDINAL)
    print("NUMERICAL", NUMERICAL)
    print("TARGET", TARGET)

    print("schema", schema)
    print('Downloading dataset...')
    download_dataset(force=False)
    plays, players, player_play, games = load_base_data()

    print("=========columns=========")
    print(plays.columns, players.columns, player_play.columns, games.columns)
    print("=========shapes=========")
    print(f" plays={plays.shape}, players={players.shape},"
          f" player_play={player_play.shape}, games={games.shape}")

    # Smoke test: run pipeline without saving
    # ml_df = feature_engineering(plays, players, player_play, games)
    # print(f"[smoke] pipeline output shape: {ml_df.shape}")

    # Smoke test: save features to disk
    save_path = 'data/ml_dataset/ml_features.parquet'
    ml_df_saved = feature_engineering(
        plays, players, player_play, games,
        save_path=save_path,
        save_format='parquet',
        debug=True
    )
    print(f"[smoke] saved features to: {save_path}")

    # Smoke test: load features back and verify
    loaded_df = load_fe_dataset(save_path, file_format='parquet')
    print(f"[smoke] loaded features shape: {loaded_df.shape}")
    # 1) Check column order
    print("Saved columns: ", ml_df_saved.columns.tolist())
    print("Loaded columns:", loaded_df.columns.tolist())

    # 2) Check dtypes
    print("\nSaved dtypes:")
    print(ml_df_saved.dtypes)
    print("\nLoaded dtypes:")
    print(loaded_df.dtypes)

    # 3) Find any columns with mismatched values or dtypes
    mismatches = []
    for col in ml_df_saved.columns:
        same_dtype = ml_df_saved[col].dtype == loaded_df[col].dtype
        same_vals = ml_df_saved[col].equals(loaded_df[col])
        if not (same_dtype and same_vals):
            mismatches.append((col, ml_df_saved[col].dtype, loaded_df[col].dtype,
                               (~(ml_df_saved[col] == loaded_df[col])).sum()
                               if same_dtype else "dtype_mismatch"))
    print("\nColumns with mismatches (col, saved_dtype, loaded_dtype, #diffs):")
    for info in mismatches:
        print(" ", info)

    assert loaded_df.equals(ml_df_saved), "Loaded DataFrame does not match saved DataFrame"
    print("[smoke] save/load consistency check passed!")

    # check sum of nulls
    play_nulls = loaded_df.isnull().sum()
    print('play_nulls (should all be 0):', play_nulls)
    print('total rows:', len(loaded_df))
    print('percentage of nulls:', play_nulls.sum() / len(loaded_df))
    assert play_nulls.sum() == 0
