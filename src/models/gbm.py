# ─────────────────────────────────────────────────────────────────────────────
"""
Gradient-boosting baseline (XGBoost classifier) module.

This file contains:
  - tune_gbm():   Use Optuna to find best XGBoost hyperparameters by maximizing CV ROC AUC.
  - fit_gbm():    Train a final XGBoost classifier (with optional early stopping) and compute test AUC.
  - save_pipeline(...) & load_pipeline(...): Save/load model + preprocessor together via joblib.
"""
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import pandas as pd
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.model_selection import cross_val_score
import optuna
from xgboost.core import XGBoostError
from src.utils.gbm_utils import save_pipeline, load_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─── Detect GPU support using modern API ───────────────────────────────────────
try:
    XGBClassifier(tree_method="hist", device="cuda")
    GPU_SUPPORT = True
except XGBoostError:
    GPU_SUPPORT = False


def _split_xy(df: pd.DataFrame, target_col: str):
    """
    Split a DataFrame into (X, y) pairs based on target_col.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def tune_gbm(X, y, n_trials: int = 50):
    """
    Run an Optuna study to maximize CV ROC AUC of an XGBClassifier.
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
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        model = XGBClassifier(**params)

        # 2) Evaluate via 3-fold CV on ROC AUC (so we maximize)
        scores = cross_val_score(
            model, X, y,
            scoring="roc_auc",
            cv=3, n_jobs=-1
        )
        # 3) Return mean AUC to maximize
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


def fit_gbm(X_tr, y_tr, X_te, y_te, **gbm_kw):
    """
    Train an XGBClassifier with optional hyperparameters and early stopping.
    Returns (fitted_model, test_auc).

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
        use_label_encoder=False,
        eval_metric="logloss",
    )
    constructor_defaults.update(gbm_kw)

    # B) Extract early_stopping_rounds if passed
    early_stopping = constructor_defaults.pop("early_stopping_rounds", None)

    # C) Instantiate the XGBClassifier
    model = XGBClassifier(**constructor_defaults)

    # D) Build fit kwargs: if early stopping, add callbacks + eval_set
    fit_kwargs = {}
    if early_stopping:
        fit_kwargs["callbacks"] = [EarlyStopping(rounds=early_stopping)]
        fit_kwargs["eval_set"] = [(X_te, y_te)]
    fit_kwargs["verbose"] = False

    # E) Fit the model
    model.fit(X_tr, y_tr, **fit_kwargs)

    # F) Evaluate on test set: compute AUC
    pred_probs = model.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, pred_probs)

    return model, test_auc


# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.load_data.load_data import download_dataset, load_base_data
    from src.feature_engineering.column_schema import ColumnSchema
    from src.feature_engineering.preprocess import (
        fit_preprocessor,
        transform_preprocessor,
        inverse_transform_preprocessor,
        _check_binary_target
    )
    from src.feature_engineering.feature_engineering import load_fe_dataset
    from src.feature_engineering.visualizations import plot_vsrg_by_zone

    # ─── 1) Load raw DataFrames via load_base_data() ─────────────────────────────
    plays, players, pp, games = load_base_data()
    # (Assumes data is already local; if not, call download_dataset() first.)

    # ─── 2) Instantiate ColumnSchema & grab column lists ─────────────────────────
    schema = ColumnSchema()
    INFO_NON_ML = schema.info_non_ml()
    NOMINAL     = schema.nominal_cols()
    ORDINAL     = schema.ordinal_cols()
    NUMERICAL   = schema.numerical_cols()
    TARGET      = schema.target_col()[0]  # e.g. 'contested_success'

    # Print schema lists for verification
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

    # ─── 5) Confirm binary target and subset columns ─────────────────────────────
    # Ensure 'contested_success' is exactly {0,1}
    _check_binary_target(ml_df, TARGET, debug=True)

    # Now select only the features + target
    ml_df = ml_df[ NUMERICAL + NOMINAL + ORDINAL + [TARGET] ]
    print(f"   Subset to features+target → shape: {ml_df.shape}")

    # ─── 6) Train/test split ─────────────────────────────────────────────────────
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
    # Because our fit_gbm is for classification, we search for hyperparameters that maximize AUC
    def objective(trial):
        # Sample classification hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 0,
            "n_jobs": -1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        # Early stopping is passed separately
        # 3-fold CV on ROC AUC (so we maximize)
        clf = XGBClassifier(**params)
        scores = cross_val_score(
            clf, X_train_np, y_train,
            scoring="roc_auc",
            cv=3, n_jobs=-1
        )
        return scores.mean()  # returning the mean AUC to maximize

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params
    print("Tuned params:", best_params)

    # ─── 11) Train & evaluate final GBM classifier ───────────────────────────────
    print("▶ Training final XGBoost classifier …")
    # Add early stopping (e.g., 10 rounds)
    best_params["early_stopping_rounds"] = 10
    clf_model, test_auc = fit_gbm(X_train_np, y_train, X_test_np, y_test, **best_params)
    print(f"Tuned XGBoost Test AUC: {test_auc:.4f}")

    # ─── 12) Compute predicted probabilities on entire dataset for VSRG ─────────
    # Reconstruct the full contested set's features & target
    X_all_np, y_all, tf_full = fit_preprocessor(ml_df, model_type="linear", debug=False)
    # Now get predicted probabilities:
    pred_probs_all = clf_model.predict_proba(X_all_np)[:, 1]
    ml_df_with_preds = ml_df.copy()
    ml_df_with_preds["pred_prob"] = pred_probs_all

    # ─── 13) Plot VSRG by vertical zone ──────────────────────────────────────────
    print("▶ Plotting VSRG by height_zone …")
    plot_vsrg_by_zone(
        ml_df_with_preds,
        zone_col="height_zone",
        target_col=TARGET,
        pred_prob_col="pred_prob",
        to_grade=True
    )

    # ─── 14) Save pipeline (model + preprocessor) ─────────────────────────────────
    save_path = "data/models/saved_models/gbm_pipeline.joblib"
    save_pipeline(clf_model, tf, path=save_path)

    # ─── 15) Load it back to verify it works ─────────────────────────────────────
    loaded_model, loaded_preprocessor = load_pipeline(save_path)
    print(f"✅ Successfully loaded pipeline from {save_path}.")
