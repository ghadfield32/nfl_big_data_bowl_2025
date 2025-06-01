# src/load_data/load_data.py

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


def download_dataset(force: bool = False) -> bool:
    """
    Download & extract NFL Big Data Bowl dataset.
    """
    if os.path.isdir(DATA_DIR) and not force:
        print(f"Data directory {DATA_DIR!r} exists. Skipping download.")
        return True
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{COMPETITION}.zip")
    try:
        a_pi.competition_download_cli(COMPETITION, path=DATA_DIR)
        print(f"Downloaded archive to {zip_path!r}")
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(DATA_DIR)
        print(f"Extracted all files into {DATA_DIR!r}")
        os.remove(zip_path)
        print(f"Removed archive {zip_path!r}")
        return True
    except Exception as e:
        print("Error downloading or extracting dataset:", e)
        return False


def load_base_data():
    plays = pd.read_csv(os.path.join(DATA_DIR, 'plays.csv'))
    players_raw = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))
    player_play = pd.read_csv(os.path.join(DATA_DIR, 'player_play.csv'))
    games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'))
    return plays, players_raw, player_play, games


def load_tracking_data(week: int = 1, nrows: int | None = None) -> pd.DataFrame:
    """
    Read one tracking CSV.

    Parameters
    ----------
    week : int
        Valid weeks are 1-9 inclusive.
    nrows : int | None
        If given, limit rows (for quick smoke tests).
    """
    try:
        week_int = int(week)
    except (TypeError, ValueError):
        raise TypeError(f"`week` must be an int between 1 and 9, got {type(week).__name__}: {week!r}")

    if not 1 <= week_int <= 9:
        raise ValueError("Week must be between 1 and 9")

    path = os.path.join(DATA_DIR, f"tracking_week_{week_int}.csv")
    return pd.read_csv(path, nrows=nrows)







if __name__ == '__main__':
    print('Downloading dataset...')
    download_dataset(force=False)
    plays, players, player_play, games = load_base_data()
    print("=========columns=========")
    print(plays.columns, players.columns, player_play.columns, games.columns)
    print("=========shapes=========")
    print(f" plays={plays.shape}, players={players.shape},"
          f" player_play={player_play.shape}, games={games.shape}")
    ml_df = feature_engineering(plays, players, player_play, games)
    print('ML dataset shape:', ml_df.shape)
    print('ML dataset columns:', ml_df.columns)


    # quick debug
    print(ml_df["is_contested"].mean())          # % of targets truly contested
    print(ml_df["contested_success"].mean())     # contested-catch success-rate
    print(ml_df.query("is_contested").passResult.value_counts())
