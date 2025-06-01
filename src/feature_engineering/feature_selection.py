
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
from pathlib import Path

# ── Model & importance imports ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# ── (Optional) SHAPIQ ─────────────────────────────────────────────
import shapiq


# ── Imports from the “new” repo ─────────────────────────────────────────────
from src.feature_engineering.column_schema import ColumnSchema
from src.feature_engineering.preprocess import (
    fit_preprocessor,
    transform_preprocessor,
    inverse_transform_preprocessor
)
from src.load_data.load_data import download_dataset, load_base_data
from src.feature_engineering.feature_engineering import load_fe_dataset

# ─────────────────────────────────────────────────────────────────────────────

def ensure_numeric_df(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop any column in X that is either:
      - NOT a numeric dtype (e.g. `object`, `category`)
      - OR is a datetime64 dtype.

    Returns:
      - X_cleaned: A DataFrame with only numeric columns
      - dropped: A list of columns that were removed
    """
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

    dropped = []
    for col in X.columns:
        # Mark non-numeric or datetime columns for dropping
        if (not is_numeric_dtype(X[col])) or is_datetime64_any_dtype(X[col]):
            dropped.append(col)

    if dropped:
        X_cleaned = X.drop(columns=dropped)
    else:
        X_cleaned = X

    return X_cleaned, dropped


def train_baseline_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    Train a simple RandomForestRegressor as our “baseline” model.
    Automatically drops any non-numeric or datetime columns in X.
    Returns the fitted RandomForestRegressor.
    """
    # 1) Drop non-numeric or datetime columns
    X_clean, dropped_cols = ensure_numeric_df(X)
    if dropped_cols:
        print(f"[DEBUG] train_baseline_model: Dropped columns before fitting --> {dropped_cols}")

    # 2) Fit on the cleaned (all-numeric) DataFrame
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_clean, y)
    return model



def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    n_jobs: int = 1,
    max_samples: float | int = None,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute permutation importances for `model` on (X, y).
    Drops any non-numeric or datetime columns from X before proceeding.
    Returns a DataFrame sorted by descending mean importance.
    """
    # 1) Ensure X is purely numeric
    X_clean, dropped_cols = ensure_numeric_df(X)
    if dropped_cols and verbose:
        print(f"[DEBUG] compute_permutation_importance: Dropped columns --> {dropped_cols}")

    # 2) Possibly subsample for speed
    X_sel, y_sel = X_clean, y
    if max_samples is not None:
        nsamp = int(len(X_clean) * max_samples) if isinstance(max_samples, float) else int(max_samples)
        if verbose:
            print(f"   Subsampling to {nsamp} rows for speed")
        X_sel, y_sel = resample(X_clean, y, replace=False, n_samples=nsamp, random_state=random_state)

    # 3) Compute permutation importances (with fallback to n_jobs=1 on OSError)
    try:
        result = permutation_importance(
            model,
            X_sel,
            y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs
        )
    except OSError as e:
        if verbose:
            print(f"⚠️  OSError encountered: {e}. Retrying with n_jobs=1")
        result = permutation_importance(
            model,
            X_sel,
            y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1
        )

    importance_df = pd.DataFrame({
        "feature": X_clean.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    if verbose:
        print("✅ Permutation importances computed.")
    return importance_df


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    nsamples: int = 100
) -> pd.DataFrame:
    """
    Compute mean‐absolute SHAP values for a tree‐based model.
    Drops any non-numeric or datetime columns from X before computing SHAP.
    Returns a DataFrame sorted by descending SHAP importance.
    """
    X_clean, dropped_cols = ensure_numeric_df(X)
    if dropped_cols:
        print(f"[DEBUG] compute_shap_importance: Dropped columns --> {dropped_cols}")

    X_sample = X_clean.sample(n=min(nsamples, len(X_clean)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = pd.DataFrame({
        "feature": X_clean.columns,
        "shap_importance": abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return mean_abs_shap


def filter_permutation_features(importance_df: pd.DataFrame, threshold: float) -> list[str]:
    """
    Given a permutation-importance DataFrame, keep all features
    whose mean importance > threshold.
    """
    kept = importance_df.loc[importance_df["importance_mean"] > threshold, "feature"]
    return kept.tolist()


def filter_shap_features(importance_df: pd.DataFrame, threshold: float) -> list[str]:
    """
    Given a SHAP-importance DataFrame, keep all features
    whose mean‐absolute SHAP value > threshold.
    """
    kept = importance_df.loc[importance_df["shap_importance"] > threshold, "feature"]
    return kept.tolist()


def select_final_features(
    perm_feats: list[str],
    shap_feats: list[str],
    mode: str = "intersection"
) -> list[str]:
    """
    Combine two lists of features (perm_feats, shap_feats). 
    If mode='intersection', return only the overlap; if 'union', return their union.
    """
    set_perm = set(perm_feats)
    set_shap = set(shap_feats)
    if mode == "union":
        final = set_perm | set_shap
    else:
        final = set_perm & set_shap
    return sorted(final)


# ── NEW: helpers to save / load feature lists ───────────────────────────────
def save_feature_list(features: list[str], file_path: str) -> None:
    """
    Save a list of feature names (one per line) to `file_path`.
    """
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        for feat in features:
            fp.write(f"{feat}\n")
    print(f"✔️ Saved feature list ({len(features)} features) to {out_path}")


def load_feature_list(file_path: str) -> list[str]:
    """
    Load a list of feature names from `file_path` (one feature per line, ignoring blanks).
    """
    with open(file_path, "r") as fp:
        return [line.strip() for line in fp if line.strip()]


def generate_and_save_final_features(
    df: pd.DataFrame,
    target_col: str,
    prefix: str = "data/models/features/",
    perm_threshold: float = 0.01,
    shap_threshold: float = 0.01,
    combine_mode: str = "intersection",
    max_samples: float | int = 0.5,
    shap_nsamples: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    file_path: str = "data/models/features/final_features.txt"
) -> tuple[list[str], str]:
    """
    1) Splits df → train/test (80/20).
    2) Drops any non-numeric or datetime columns (except target).
    3) Fits a baseline RandomForest on the train split.
    4) Computes permutation & SHAP importances on train.
    5) Filters features by thresholds and combines them.
    6) Saves the final list to disk under `<prefix>final_features.txt`.

    Returns (final_feature_list, saved_file_path).
    """

    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

    # ─── Drop any non-numeric OR datetime columns ─────────────────────────────
    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        # If NOT numeric OR is a datetime64 column, drop
        if not is_numeric_dtype(df[col]) or is_datetime64_any_dtype(df[col]):
            cols_to_drop.append(col)

    if cols_to_drop:
        if verbose:
            print(f"[DEBUG] Dropping non-numeric or datetime columns before FS: {cols_to_drop}")  # :contentReference[oaicite:14]{index=14}
        df = df.drop(columns=cols_to_drop)
    # ───────────────────────────────────────────────────────────────────────────

    # 1) Split into train/test
    train_df, _ = train_test_split(df, test_size=0.2, random_state=random_state)

    # 2) Build X_train / y_train
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    if verbose:
        print(f"▶ Running baseline feature selection on {len(X_train)} training rows, "
              f"{X_train.shape[1]} initial features (all numeric).")

    # 3) Train baseline model
    model = train_baseline_model(X_train, y_train)  # :contentReference[oaicite:15]{index=15}

    # 4) Compute permutation importances
    perm_imp = compute_permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        n_jobs=2,
        max_samples=max_samples,
        random_state=random_state,
        verbose=verbose
    )

    # 5) Compute SHAP importances
    shap_imp = compute_shap_importance(model, X_train, nsamples=shap_nsamples)
    if verbose:
        print("✅ SHAP importances computed.")

    # 6) Filter by thresholds
    perm_feats = filter_permutation_features(perm_imp, perm_threshold)
    shap_feats = filter_shap_features(shap_imp, shap_threshold)
    final_feats = select_final_features(perm_feats, shap_feats, mode=combine_mode)

    # 7) Save to disk
    save_feature_list(final_feats, file_path)

    if verbose:
        print(f"▶ Final feature count = {len(final_feats)} "
              f"(perm > {perm_threshold}, shap > {shap_threshold}, mode='{combine_mode}')")

    return final_feats, file_path




def filter_to_final_features(
    df: pd.DataFrame,
    file_path: str = "data/models/features/final_features.txt"
) -> pd.DataFrame:
    """
    Given a DataFrame that contains at least {info_non_ml IDs} + final_features + {target},
    returns a copy of df with only those columns, in the order:
      [ all info_non_ml IDs ] + [ all final_features ] + [ target ].

    Raises ValueError if any required column is missing.
    """
    cols        = ColumnSchema()
    final_feats = load_feature_list(file_path)
    id_cols     = cols.info_non_ml()   # e.g. ['gameId','playId','receiverId','cbId','birthDate_x','birthDate_y','has_target']
    target      = cols.target_col()[0] # e.g. 'contested_success'

    keep_cols = id_cols + final_feats + [ target ]
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot filter: missing columns {missing}")
    return df[ keep_cols ].copy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke tests / usage examples:

    1) We generate & save a baseline feature list (using both permutation + SHAP importances),
       then reload that list from disk and show how to slice a fresh ML DataFrame to only 
       [IDs + final_feats + target]. 
    """
    from src.feature_engineering.column_schema import ColumnSchema
    import json

    # 0) Bring in column‐schema constants
    schema      = ColumnSchema()
    INFO_NON_ML  = schema.info_non_ml()
    NOMINAL      = schema.nominal_cols()
    ORDINAL      = schema.ordinal_cols()
    NUMERICAL    = schema.numerical_cols()
    TARGET       = schema.target_col()

    print("INFO_NON_ML:", INFO_NON_ML)
    print("NOMINAL   :", NOMINAL)
    print("ORDINAL   :", ORDINAL)
    print("NUMERICAL :", NUMERICAL)
    print("TARGET    :", TARGET)
    print("[smoke] Column schema validation passed ✅\n")
    print(json.dumps(schema.as_dict(), indent=2))

    # ——————————————————————————————————————————————————————————————————————————
    # 1) Load the full ML dataset (with all info_non_ml columns)
    data_path  = 'data/ml_dataset/ml_features.parquet'
    print(f"\n▶ Loading full ML DataFrame from {data_path} …")
    ml_df_full = load_fe_dataset(data_path, file_format='parquet')
    ml_df_full = ml_df_full[ ml_df_full['is_contested'] == 1 ]
    print(f"   ML DataFrame shape (is_contested == 1): {ml_df_full.shape}")

    # 2) Build a “model-ready” DataFrame that drops only info_non_ml IDs, 
    #    so that when we run generate_and_save_final_features(), 
    #    we pass exactly [features + target].
    feature_cols = NUMERICAL + NOMINAL + ORDINAL
    target_col   = TARGET[0]  # 'contested_success'

    df_for_fs = ml_df_full[ feature_cols + [target_col] ].copy()
    print(f"\n▶ Running baseline feature‐selection on {df_for_fs.shape} …")
    feat_file: str = "data/models/features/final_features.txt"
    # 3) Generate & save final feature list (perm_threshold=0.01, shap_threshold=0.01)
    final_feats = generate_and_save_final_features(
        df     = df_for_fs,
        target_col = target_col,
        prefix     = "data/models/features/",
        perm_threshold = 0.01,
        shap_threshold = 0.01,
        combine_mode   = "intersection",
        max_samples    = 0.5,
        shap_nsamples  = 100,
        random_state   = 42,
        verbose        = True,
        file_path      = feat_file
    )
    train_df, _ = train_test_split(df_for_fs, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    model   = train_baseline_model(X_train, y_train)

    perm_imp = compute_permutation_importance(
        model, X_train, y_train,
        n_repeats=10, n_jobs=2, max_samples=0.5, random_state=42, verbose=False
    )
    shap_imp = compute_shap_importance(model, X_train, nsamples=100)

    # Extract and print top-10 of each:
    top10_perm = perm_imp.head(10).reset_index(drop=True)
    top10_shap = shap_imp.head(10).reset_index(drop=True)

    print("\n🔹 Top 10 Permutation Importances:")
    print(top10_perm[["feature","importance_mean"]].to_string(index=False))

    print("\n🔹 Top 10 SHAP Importances:")
    print(top10_shap[["feature","shap_importance"]].to_string(index=False))
    # ────────────────────────

    # 4) Show the saved feature list
    print(f"\n▶ Saved feature list (count={len(final_feats)}):")
    print(final_feats)
    print("▶ Saved to:", feat_file)

    # 5) Load that list back from disk
    loaded_feats = load_feature_list(feat_file)
    print(f"\n▶ Loaded {len(loaded_feats)} features from {feat_file}:")
    print(loaded_feats[:10], "…")

    # 6) Finally, demonstrate how to filter the full ml_df_full 
    #    down to exactly [ all info_non_ml IDs ] + loaded_feats + [ target ].
    print(f"\n▶ Filtering ml_df_full (shape={ml_df_full.shape}) to only [IDs + final_feats + target] …")
    ml_df_filtered = filter_to_final_features(ml_df_full, file_path=feat_file)
    print(f"   Resulting shape: {ml_df_filtered.shape}")
    print(f"   Columns: {ml_df_filtered.columns.tolist()}")

