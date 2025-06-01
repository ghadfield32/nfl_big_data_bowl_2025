"""
Feature engineering that *starts* from tracking CSVs but does NOT touch other
raw tables except for an OPTIONAL plays slice to tag offensive vs defensive
clubs.  All calls that actually read tracking files are still delegated to
`load_tracking_data` from `src/load_data/load_data.py`.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict
import numpy as np, pandas as pd
from math import atan2, hypot, cos, sin
from datetime import timedelta
from src.load_data.load_data import load_tracking_data


def tracking_feature_engineering(
    weeks: Iterable[int],
    *, debug: bool = False,
    burst_cap: float = 13.0      # yd / s – generous biomechanical ceiling
) -> pd.DataFrame:
    """Receiver-centric snapshot builder (fast, memory-lean)."""

    one_sec = timedelta(seconds=1)
    rows: List[pd.DataFrame] = []

    if debug:
        print(f"[DEBUG][tracking] Processing weeks: {list(weeks)}")

    for wk in weeks:
        # 1. load tracking data -------------------------------------------------
        tr = load_tracking_data(week=wk)[
            ["gameId","playId","nflId","club","x","y","s","o","event","time"]
        ].copy()
        tr["time"] = pd.to_datetime(tr["time"], errors="coerce")
        # ★ If `time` is unparseable, that row's time becomes NaT.

        # 2. cache per-play slices ----------------------------------------
        play_groups: Dict[Tuple[int,int], pd.DataFrame] = {
            k: g for k, g in tr.groupby(["gameId","playId"])
        }

        # 3. snapshot at pass_arrived -------------------------------------
        snap = (
            tr[tr["event"] == "pass_arrived"]
              .rename(columns={"nflId":"receiverId",
                               "x":"rcv_x","y":"rcv_y",
                               "s":"rcv_s","o":"rcv_o"})
              [["gameId","playId","receiverId","club",
                "rcv_x","rcv_y","rcv_s","rcv_o","time"]]
        )
        if debug:
            print(f"[DEBUG][tracking] Week {wk}: loaded {len(snap)} pass_arrived rows")

        # potential CB candidates from same snapshot
        cb_cand = snap.rename(columns={
            "receiverId":"cbId","club":"cb_club",
            "rcv_x":"cb_x","rcv_y":"cb_y",
            "rcv_s":"cb_s","rcv_o":"cb_o",
        })

        # join every receiver with every candidate then filter
        m = pd.merge(snap, cb_cand, on=["gameId","playId"])
        m = m[m["club"] != m["cb_club"]]
        m["sep_receiver_cb"] = np.hypot(m["rcv_x"]-m["cb_x"], m["rcv_y"]-m["cb_y"])
        if debug:
            print(f"[DEBUG][tracking] Week {wk}: {len(m)} receiver-CB pairs after filter")

        # second-closest defender distance
        second_sep = (
            m.groupby(["gameId","playId","receiverId"])["sep_receiver_cb"]
             .apply(lambda s: s.nsmallest(2).iloc[-1] if len(s) >= 2 else np.nan)
             .rename("min_other_sep")
             .reset_index()
        )

        # nearest defender per receiver
        idx  = m.groupby(["gameId","playId","receiverId"])["sep_receiver_cb"].idxmin()
        snap = m.loc[idx].reset_index(drop=True).merge(
            second_sep, on=["gameId","playId","receiverId"], how="left"
        )
        if debug:
            print(f"[DEBUG][tracking] Week {wk}: {len(snap)} nearest-CB snapshots")

        # 3e. rename time column
        if "time_x" in snap.columns:
            snap.rename(columns={"time_x": "time_rcv"}, inplace=True)
            snap.drop(columns=["time_y"], errors="ignore", inplace=True)
        else:
            snap.rename(columns={"time": "time_rcv"}, inplace=True)
        # ★ At this point, some `time_rcv` may be NaT if original `time` was unparseable.

        assert "time_rcv" in snap.columns, "time_rcv column missing after merges"

        # 4. per-snapshot dynamics ---------------------------------------
        sep_vel, close_spd, burst = [], [], []
        for row in snap.itertuples(index=False):
            gid, pid, rcv, cb, x_r, y_r, x_c, y_c, s_c, o_c, t_end = (
                row.gameId, row.playId, row.receiverId, row.cbId,
                row.rcv_x, row.rcv_y, row.cb_x, row.cb_y,
                row.cb_s, row.cb_o, row.time_rcv
            )

            play_df = play_groups.get((gid, pid))
            if play_df is None:
                # ★ If the play group is missing entirely, we cannot compute dynamics
                sep_vel.append(np.nan); close_spd.append(np.nan); burst.append(np.nan)
                continue

            win  = play_df[play_df["time"] >= t_end - one_sec]
            two  = win[win["nflId"].isin([rcv, cb])]
            r_sl = two[two["nflId"] == rcv]

            # receiver burst speed
            burst.append(r_sl["s"].max() if not r_sl.empty else np.nan)

            if two.empty:
                # ★ If no positions for receiver + CB in that last second window
                sep_vel.append(np.nan); close_spd.append(np.nan)
                continue

            r = r_sl.sort_values("time")
            c = two[two["nflId"] == cb].sort_values("time")
            if len(r) < 2 or c.empty:
                # ★ Not enough data to compute velocity or closing speed
                sep_vel.append(np.nan); close_spd.append(np.nan)
                continue

            # separation velocity
            L   = min(len(r), len(c))
            sep = np.hypot(r["x"].values[:L] - c["x"].values[:L],
                           r["y"].values[:L] - c["y"].values[:L])
            dt  = (r["time"].iloc[-1] - r["time"].iloc[0]).total_seconds()
            sep_vel.append((sep[-1] - sep[0]) / dt if dt else np.nan)

            # closing speed inline
            dx, dy = x_r - x_c, y_r - y_c
            dist   = hypot(dx, dy) or 1e-6
            rad    = np.deg2rad((360 - o_c + 90) % 360)
            close_spd.append(s_c * (cos(rad)*dx/dist + sin(rad)*dy/dist))

        snap["sep_velocity"]   = sep_vel
        snap["closing_speed"]  = close_spd
        snap["s_max_last1s"]   = burst

        # 4b. cap implausible burst ---------------------------------------
        n_cap = snap.loc[snap["s_max_last1s"] > burst_cap, "s_max_last1s"].count()
        snap.loc[snap["s_max_last1s"] > burst_cap, "s_max_last1s"] = np.nan

        # 5. leverage angle ----------------------------------------------
        snap["leverage_angle"] = np.arctan2(
            snap["cb_y"] - snap["rcv_y"],
            snap["cb_x"] - snap["rcv_x"]
        )

        rows.append(snap)

        if debug:
            n_nan = snap["s_max_last1s"].isna().sum()
            print(f"W{wk}: {len(snap):,} snaps | avg_sep_vel {np.nanmean(sep_vel):.2f} | burst>cap {n_cap} | burst_nan {n_nan}")

    # ── Concatenate all weeks ──────────────────────────────────────────────────
    out = pd.concat(rows, ignore_index=True)

    # ── NEW: Identify & drop incomplete rows (where any critical field is NaN) ──
    ### <<<<< NEW >>>>> ###
    if debug:
        print(f"[DEBUG][tracking] Before dropping incompletes: total rows = {len(out)}")
        na_counts = out[[
            "time_rcv","cbId","sep_velocity","closing_speed","s_max_last1s"
        ]].isna().sum()
        print(f"[DEBUG][tracking] NaN counts before drop:\n{na_counts.to_frame(name='n_missing')}\n")

    # Build mask for rows that have *all* required fields present:
    complete_mask = (
        out["time_rcv"].notna() &
        out["cbId"].notna() &
        out["sep_velocity"].notna() &
        out["closing_speed"].notna() &
        out["s_max_last1s"].notna()
    )

    if debug:
        n_incomplete = len(out) - complete_mask.sum()
        print(f"[DEBUG][tracking] Dropping {n_incomplete} incomplete rows "
              f"(lack time_rcv or cbId or dynamics).")
        # Show a tiny sample of which keys are incomplete
        sample_bad = out.loc[~complete_mask, ["gameId","playId","receiverId"]].drop_duplicates().sample(
            min(5, (~complete_mask).sum()), random_state=0
        )
        print(f"[DEBUG][tracking] Example dropped (gameId, playId, receiverId):\n{sample_bad.to_dict('records')}\n")

    # Actually drop those rows:
    out = out.loc[complete_mask].reset_index(drop=True)
    ### <<<<< END NEW >>>>> ###

    # 6. (OPTIONAL) final validation checks ---------------------------
    if debug:
        # 6a. Uniqueness: one row per (gameId,playId,receiverId)
        assert not out.duplicated(['gameId','playId','receiverId']).any(), \
            'Duplicates found in (gameId,playId,receiverId)'

        # 6b. Club mismatch: receiver.club != cb_club always
        assert (out['club'] != out['cb_club']).all(), \
            'Some snapshots have receiver and CB on same club'

        # 6c. Burst capping: no s_max_last1s > burst_cap
        max_burst = out['s_max_last1s'].max()
        assert max_burst <= burst_cap, f'Found uncapped burst: {max_burst:.2f} yd/s'

        # 6d. Second‐closest separation ≥ closest
        bad = out.query('min_other_sep < sep_receiver_cb')
        assert bad.empty, 'min_other_sep < sep_receiver_cb in some rows'

        # 6e. Print summary stats for all new features
        print("--- feature summaries AFTER drop ---")
        print(
            out[[
                'sep_receiver_cb',
                'min_other_sep',
                'sep_velocity',
                'closing_speed',
                's_max_last1s',
                'leverage_angle'
            ]].describe()
        )

        total_rows = len(out)
        total_plays = out[["gameId","playId"]].drop_duplicates().shape[0]
        print(f"[DEBUG][tracking] total tracking rows (after drop): {total_rows}, unique plays tracked: {total_plays}")
        print(f"[DEBUG][tracking] Rows: {len(out):,}, Memory: {out.memory_usage(deep=True).sum()/1e6:.1f} MB\n")

    # Finally, we know there are *no* NaNs left
    assert not out.isnull().any().any(), "Some columns have NaNs after final drop"
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

    fe = tracking_feature_engineering(range(1, 5), debug=True)
    print(fe[["s_max_last1s","leverage_angle"]].describe())

    # check sum of nulls
    play_nulls = fe.isnull().sum()
    print('play_nulls (should all be 0):', play_nulls)
    print('total rows:', len(fe))
    print('percentage of nulls:', play_nulls.sum() / len(fe))
    assert play_nulls.sum() == 0
