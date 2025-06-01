# ─────────────────────────────────────────────────────────────────────────────
"""
Gradient-boosting baseline (XGBoost regressor) module.

This file contains:
  - tune_gbm():   Use Optuna to find best XGBoost hyperparameters by minimizing CV RMSE.
  - fit_gbm():    Train a final XGBoost model (with optional early stopping) and compute test RMSE.
  - save_pipeline(...) & load_pipeline(...): Save/load model + preprocessor together via joblib.
"""
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping     # ① import the new callback
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import optuna
from xgboost.core import XGBoostError
from src.utils.gbm_utils import save_pipeline, load_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
# ─── Detect GPU support using modern API ───────────────────────────────────────
try:
    XGBRegressor(tree_method="hist", device="cuda")
    GPU_SUPPORT = True
except XGBoostError:
    GPU_SUPPORT = False


def _split_xy(df: pd.DataFrame):
    """
    Split a DataFrame into (X, y) pairs. Assumes 'exit_velo' is the target column.
    """
    X = df.drop(columns=["exit_velo"])
    y = df["exit_velo"]
    return X, y


def tune_gbm(X, y, n_trials: int = 50):
    """
    Run an Optuna study to minimize CV RMSE of an XGBRegressor.
    Uses device='cuda' if available, else CPU.
    """
    def objective(trial):
        # 1) Sample hyperparameters for this trial
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 0,
            "n_jobs": -1,
        }
        # Always use histogram tree method
        params.update({"tree_method": "hist"})
        model = XGBRegressor(**params)

        # 2) Evaluate via 3-fold CV on negative RMSE (so we minimize)
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=3, n_jobs=-1
        )
        # 3) Return the mean RMSE (positive number) to minimize
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


def fit_gbm(X_tr, y_tr, X_te, y_te, **gbm_kw):
    """
    Train an XGBRegressor with optional hyperparameters and early stopping.
    Returns (fitted_model, rmse_on_test).

    Parameters
    ----------
    X_tr, y_tr : numpy arrays or DataFrames for training
    X_te, y_te : numpy arrays or DataFrames for validation
    gbm_kw      : optional XGB hyperparameters, e.g., early_stopping_rounds
    """
    # A) Default constructor args → overridden by gbm_kw
    constructor_defaults = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=0,
        n_jobs=-1,
        tree_method="hist",
    )
    constructor_defaults.update(gbm_kw)

    # B) Extract early_stopping_rounds if passed
    early_stopping = constructor_defaults.pop("early_stopping_rounds", None)

    # C) Instantiate the XGBRegressor
    model = XGBRegressor(**constructor_defaults)

    # D) Build fit kwargs: if early stopping, add callbacks + eval_set
    fit_kwargs = {}
    if early_stopping:
        fit_kwargs["callbacks"] = [EarlyStopping(rounds=early_stopping)]
        fit_kwargs["eval_set"] = [(X_te, y_te)]
    fit_kwargs["verbose"] = False

    # E) Fit the model
    model.fit(X_tr, y_tr, **fit_kwargs)

    # F) Evaluate on test set
    preds = model.predict(X_te)
    rmse  = root_mean_squared_error(y_te, preds)

    return model, rmse


# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.load_data.load_data import download_dataset, load_base_data
    from src.feature_engineering.column_schema import ColumnSchema
    from src.feature_engineering.preprocess import fit_preprocessor, transform_preprocessor, inverse_transform_preprocessor
    from src.feature_engineering.feature_engineering import feature_engineering, load_fe_dataset
    # ─── 1) Load raw DataFrames via load_base_data() ─────────────────────────────
    plays, players, pp, games = load_base_data()
    # (Assumes data is already local; if not, call download_dataset() first.)

    # ─── 2) Instantiate ColumnSchema & grab column lists ─────────────────────────
    schema = ColumnSchema()
    INFO_NON_ML = schema.info_non_ml()
    NOMINAL     = schema.nominal_cols()
    ORDINAL     = schema.ordinal_cols()
    NUMERICAL   = schema.numerical_cols()
    TARGET      = schema.target_col()

    # Print them out to verify schema
    print("INFO_NON_ML:", INFO_NON_ML)
    print("NOMINAL   :", NOMINAL)
    print("ORDINAL   :", ORDINAL)
    print("NUMERICAL :", NUMERICAL)
    print("TARGET    :", TARGET)
    print("[smoke] Column schema validation passed ✅\n")
    print(json.dumps(schema.as_dict(), indent=2))

    # ─── 3) Load feature-engineered dataset from disk ─────────────────────────────
    data_path = "data/ml_dataset/ml_features.parquet"
    print(f"\n▶ Loading full ML DataFrame from {data_path} …")
    ml_df = load_fe_dataset(data_path, file_format="parquet")

    # ─── 4) Filter to only contested plays ───────────────────────────────────────
    ml_df = ml_df[ ml_df["is_contested"] == 1 ]
    print(f"   ML DataFrame shape (is_contested == 1): {ml_df.shape}")

    # ─── 5) Subset to exactly [NUMERICAL + NOMINAL + ORDINAL + TARGET] ───────────
    ml_df = ml_df[ NUMERICAL + NOMINAL + ORDINAL + TARGET ]
    print(f"   Subset to features+target → shape: {ml_df.shape}")

    # ─── 6) Train/test split ─────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(ml_df, test_size=0.2, random_state=42)
    print(f"   Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ─── 7) Fit preprocessing on train set (debug=True to see logs) ────────────────
    X_train_np, y_train, tf = fit_preprocessor(train_df, model_type="linear", debug=True)

    # ─── 8) Transform test set ───────────────────────────────────────────────────
    X_test_np, y_test = transform_preprocessor(test_df, tf)
    print("Processed shapes:", X_train_np.shape, X_test_np.shape)

    # ─── 9) Quick check of inverse_transform_preprocessor ─────────────────────────
    print("\n==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train_np, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print(f"Shape: {df_back.shape} → original X_train shape before transform: {X_train_np.shape}\n")

    # ─── 10) Hyperparameter tuning (Optuna) ───────────────────────────────────────
    print("▶ Running hyperparameter tuning with Optuna …")
    best_params = tune_gbm(X_train_np, y_train, n_trials=50)
    print("Tuned params:", best_params)

    # ─── 11) Train & evaluate final GBM ──────────────────────────────────────────
    print("▶ Training final XGBoost model …")
    gbm_model, rmse = fit_gbm(
        X_train_np, y_train, X_test_np, y_test, **best_params
    )
    print(f"Tuned XGBoost RMSE: {rmse:.4f}")

    # ─── 12) Save pipeline (model + preprocessor) ─────────────────────────────────
    save_path = "data/models/saved_models/gbm_pipeline.joblib"
    save_pipeline(gbm_model, tf, path=save_path)

    # ─── 13) Load it back to verify it works ──────────────────────────────────────
    loaded_model, loaded_preprocessor = load_pipeline(save_path)
    print(f"✅ Successfully loaded pipeline from {save_path}.")
