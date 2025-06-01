"""
Feature engineering utilities that use ONLY the *player_play* table.
"""
import pandas as pd
from src.feature_engineering.utils import _label_height_zone, convert_height_to_inches, _height_to_inches


def player_play_feature_engineering(pp_raw: pd.DataFrame, *, debug: bool=False) -> pd.DataFrame:
    """
    One row per (gameId, playId, receiverId), with:
      ▸ qb_was_hit flag
      ▸ route one-hots
      ▸ motion / shift flags
      ▸ pass-protection help flags
    """
    df = pp_raw.copy()

    # ─── 0. Debug: Show how many raw rows have missing motion/shift  ───
    raw_nulls_imotion = df["inMotionAtBallSnap"].isna().sum()
    raw_nulls_shift   = df["motionSinceLineset"].isna().sum()
    if debug:
        print(f"[DEBUG-raw] Total rows: {len(df)}")
        print(f"[DEBUG-raw] inMotionAtBallSnap missing: {raw_nulls_imotion}")
        print(f"[DEBUG-raw] motionSinceLineset missing: {raw_nulls_shift}")

    # ─── 1️⃣  targeted-receiver rows  ──────────────────────────────────
    targets = (
        df.loc[df["wasTargettedReceiver"] == 1, ["gameId", "playId", "nflId"]]
          .rename(columns={"nflId": "receiverId"})
    )
    if debug:
        print(f"[DEBUG] Number of targeted-receiver rows in raw df: {len(targets)}")
        print(f"[DEBUG] Unique plays in targets: {targets[['gameId','playId']].drop_duplicates().shape[0]}")

    # ─── 2️⃣  QB-hit per play  ──────────────────────────────────────────
    hit = (
        df.groupby(["gameId","playId"])["quarterbackHit"]
          .max()
          .rename("qb_was_hit")
          .reset_index()
    )
    out = targets.merge(hit, on=["gameId","playId"], how="left")
    if debug:
        print(f"[DEBUG] After merging qb_was_hit, rows={len(out)}, null qb_was_hit: {out['qb_was_hit'].isna().sum()}")

    # ─── 3️⃣  route one-hots  ───────────────────────────────────────────
    route_col = "routeRan" if "routeRan" in df.columns else ("route" if "route" in df.columns else None)
    if route_col:
        rts = (
            df[["gameId","playId","nflId",route_col]]
              .rename(columns={route_col: "route", "nflId": "receiverId"})
        )
        rts = pd.get_dummies(rts, columns=["route"], prefix="route")
        out = out.merge(rts, on=["gameId","playId","receiverId"], how="left")
        if debug:
            null_routes = out.filter(like="route_").isna().sum().sum()
            print(f"[DEBUG] After merging routes, total null entries among route dummies: {null_routes}")

    # ─── 4️⃣  motion / shift flags  ─────────────────────────────────────
    # 4a. Compute grouped max (could be NaN if entire play is null)
    snap_motion = (
        df.groupby(["gameId","playId"])["inMotionAtBallSnap"]
          .max()
          .rename("in_motion_at_snap")
    )
    shift_flag = (
        df.groupby(["gameId","playId"])["motionSinceLineset"]
          .max()
          .rename("shift_since_line")
    )

    # 4b. Debug how many plays are “all null” in raw data
    if debug:
        tmp1 = df.copy()
        tmp1["_is_null_imotion"] = tmp1["inMotionAtBallSnap"].isna()
        imotion_by_play = (
            tmp1.groupby(["gameId","playId"])["_is_null_imotion"]
                .agg(all_null=("all"), any_null=("any"))
                .reset_index()
        )
        num_play_all_null_imotion = imotion_by_play["all_null"].sum()
        print(f"[DEBUG] Number of plays where EVERY row’s inMotionAtBallSnap is null: {num_play_all_null_imotion}")

        tmp2 = df.copy()
        tmp2["_is_null_shift"] = tmp2["motionSinceLineset"].isna()
        shift_by_play = (
            tmp2.groupby(["gameId","playId"])["_is_null_shift"]
                .agg(all_null=("all"), any_null=("any"))
                .reset_index()
        )
        num_play_all_null_shift = shift_by_play["all_null"].sum()
        print(f"[DEBUG] Number of plays where EVERY row’s motionSinceLineset is null: {num_play_all_null_shift}")

    # 4c. Merge into “out”
    out = out.merge(snap_motion, on=["gameId","playId"], how="left")
    out = out.merge(shift_flag , on=["gameId","playId"], how="left")

    if debug:
        null_imotion_after_merge = out["in_motion_at_snap"].isna().sum()
        null_shift_after_merge  = out["shift_since_line"].isna().sum()
        print(f"[DEBUG] After merging, plays with in_motion_at_snap = NaN: {null_imotion_after_merge}")
        print(f"[DEBUG] After merging, plays with shift_since_line  = NaN: {null_shift_after_merge}")
        print(f"[DEBUG] Unique plays in out: {out[['gameId','playId']].drop_duplicates().shape[0]}")

    # ─────────— CHOOSE ONE: Filter  OR  Fill  ────────────────────────────

    # === OPTION A: DROP all plays missing either flag ===
    # Uncomment the entire block below if you prefer to filter out those plays.

    if debug:
        print("[DEBUG] Entering drop‐mode for missing flags...")
    missing_motion_mask = out["in_motion_at_snap"].isna()
    missing_shift_mask  = out["shift_since_line"].isna()
    to_drop_mask = missing_motion_mask | missing_shift_mask
    if debug:
        print(f"[DEBUG] Dropping {to_drop_mask.sum()} rows (plays) lacking motion/shift info")
    out = out.loc[~to_drop_mask].reset_index(drop=True)
    if debug:
        print(f"[DEBUG] After dropping, total rows = {len(out)}")
        print(f"[DEBUG] in_motion_at_snap nulls now: {out['in_motion_at_snap'].isna().sum()}")
        print(f"[DEBUG] shift_since_line nulls now:  {out['shift_since_line'].isna().sum()}")



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
    player_play_fe = player_play_feature_engineering(player_play)
    print(player_play_fe.columns)


    # check sum of nulls
    play_nulls = player_play_fe.isnull().sum()
    print('play_nulls:', play_nulls)
    print('total rows:', len(player_play_fe))
    print('percentage of nulls:', play_nulls.sum() / len(player_play_fe))
    assert play_nulls.sum() == 0
