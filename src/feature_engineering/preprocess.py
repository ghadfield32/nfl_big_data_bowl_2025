import pandas as pd
import numpy as np
import warnings

from sklearn.experimental import enable_iterative_imputer  # noqa: registers IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from src.feature_engineering.column_schema import ColumnSchema
from scipy import sparse


def _ensure_dense(mat):
    return mat.toarray() if sparse.issparse(mat) else mat


def _check_binary_target(df: pd.DataFrame, target_col: str, debug: bool = False) -> None:
    """
    Verify that `df[target_col]` is binary (only 0 and 1). 
    If float dtype, cast to integer if values are exactly 0.0/1.0.
    Raises ValueError if any other values or >2 unique categories are found.
    """
    # 1) Check dtype
    series = df[target_col]
    if series.dtype.kind in {"i", "u", "b"}:
        # already integer or boolean
        pass
    elif series.dtype.kind == "f":
        # float: check if only 0.0 or 1.0
        unique_vals = series.dropna().unique()
        set_vals = set(unique_vals.tolist())
        if set_vals <= {0.0, 1.0}:
            if debug:
                print(f"[DEBUG][_check_binary_target] Casting float column "
                      f"'{target_col}' to int.")
            df[target_col] = series.astype(int)
        else:
            raise ValueError(
                f"Target column '{target_col}' has float values other than "
                f"0.0/1.0: {set_vals}"
            )
    else:
        raise ValueError(
            f"Target column '{target_col}' must be integer/boolean or float "
            f"of {{0.0, 1.0}}; got dtype={series.dtype}"
        )

    # 2) Check for exactly two unique values: 0 and 1
    unique_vals = df[target_col].dropna().unique()
    set_vals = set(unique_vals.tolist())
    if set_vals != {0, 1}:
        raise ValueError(
            f"Target column '{target_col}' is not binary. "
            f"Found unique values: {set_vals}"
        )

    if debug:
        print(f"[DEBUG][_check_binary_target] '{target_col}' is confirmed "
              f"binary with values {set_vals}.")


def compute_clip_bounds(
    series: pd.Series,
    *,
    method: str = "quantile",
    quantiles: tuple[float, float] = (0.01, 0.99),
    std_multiplier: float = 3.0,
    debug: bool = False
) -> tuple[float, float]:
    """
    Compute (lower, upper) but do not apply them.

    If the series is not numeric, return (None, None) and optionally print a debug message.
    """
    s = series.dropna()

    # 1) If dtype is not numeric, skip computing bounds
    if not is_numeric_dtype(s):
        if debug:
            print(f"[DEBUG] compute_clip_bounds: series dtype is {s.dtype}, not numeric; skipping clip bounds.")
        return (None, None)

    # 2) Proceed based on method
    if method == "quantile":
        arr = s.astype("float64")
        result = arr.quantile(list(quantiles))
        return (float(result.iloc[0]), float(result.iloc[1]))

    if method == "mean_std":
        mu, sigma = s.mean(), s.std()
        return (mu - std_multiplier * sigma, mu + std_multiplier * sigma)

    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    raise ValueError(f"Unknown method {method}")


def filter_and_clip(
    df: pd.DataFrame,
    lower: float = None,
    upper: float = None,
    quantiles: tuple[float, float] = (0.01, 0.99),
    debug: bool = False
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """
    Clean the dataset by:
      1. (Placeholder) Filtering out unwanted rows (e.g., bunts/popups).
      2. Dropping any rows where the target is outside the (lower, upper) quantile bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean (must contain the target column).
    lower : float, optional
        Lower bound for clipping; if None, computed from data using quantiles.
    upper : float, optional
        Upper bound for clipping; if None, computed from data using quantiles.
    quantiles : tuple[float, float], default=(0.01, 0.99)
        Quantiles to use if computing bounds from data. E.g., (0.01, 0.99).
    debug : bool, default=False
        If True, print diagnostic information.

    Returns
    -------
    cleaned_df : pd.DataFrame
        A new DataFrame with rows dropped where target < lower or target > upper.
    (lower, upper) : tuple[float, float]
        The numeric bounds that were used for filtering.
    """
    # 1) Determine the name of the target column via ColumnSchema
    cols = ColumnSchema()
    TARGET = cols.target_col()[0]

    # 2) Compute clip bounds if not provided
    if lower is None or upper is None:
        lower_computed, upper_computed = compute_clip_bounds(
            df[TARGET],
            method="quantile",
            quantiles=quantiles,
            debug=debug
        )
        if lower is None:
            lower = lower_computed
        if upper is None:
            upper = upper_computed

    if debug:
        print(f"[DEBUG][filter_and_clip] computed lower={lower:.4f}, upper={upper:.4f} for '{TARGET}'")
        before_n = len(df)

    # 3) Drop rows whose target is outside [lower, upper]
    mask = df[TARGET].between(lower, upper)
    df_filtered = df.loc[mask].copy()

    if debug:
        after_n = len(df_filtered)
        n_dropped = before_n - after_n
        print(f"[DEBUG][filter_and_clip] Dropped {n_dropped} rows outside [{lower:.4f}, {upper:.4f}]")

    return df_filtered, (lower, upper)


def fit_preprocessor(
    df: pd.DataFrame,
    model_type: str = "linear",
    debug: bool = False,
    quantiles: tuple[float, float] = (0.01, 0.99),
    max_safe_rows: int = 200000
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """
    Fit a ColumnTransformer to the input DataFrame and return (X_matrix, y_series, fitted_transformer).
    We only use columns that actually exist in df, and skip any MissingIndicator if there are
    no missing values in the ordinal columns.

    This version also relies on filter_and_clip to drop extreme target rows.
    """

    # 6.3.1: Early warning if too few features remain
    cols = ColumnSchema()
    all_num_feats = [c for c in cols.numerical_cols() if c in df.columns and c != cols.target_col()[0]]
    all_ord_feats = [c for c in cols.ordinal_cols()   if c in df.columns]
    all_nom_feats = [c for c in cols.nominal_cols()   if c in df.columns]

    total_feats = len(all_num_feats) + len(all_ord_feats) + len(all_nom_feats)
    if total_feats <= 1:
        warnings.warn(
            f"[WARN] Only {total_feats} column(s) detected for preprocessing: "
            f"numeric={all_num_feats}, ordinal={all_ord_feats}, nominal={all_nom_feats}. "
            "Proceeding anyway (will likely lead to shape mismatch later)."
        )  # :contentReference[oaicite:18]{index=18}

    # 0) Domain cleaning & clip extremes.  This actually filters out rows beyond [lower, upper].
    df, (lower, upper) = filter_and_clip(df, quantiles=quantiles, debug=debug)

    # 1) Identify the target column and confirm it's binary
    TARGET = cols.target_col()[0]  # e.g. "contested_success"
    # Ensure the target is binary {0,1}
    _check_binary_target(df, TARGET, debug=debug)

    # 2) Recompute feature lists after filtering
    all_num_feats = [c for c in cols.numerical_cols() if (c in df.columns and c != TARGET)]
    all_ord_feats = [c for c in cols.ordinal_cols() if c in df.columns]
    all_nom_feats = [c for c in cols.nominal_cols() if c in df.columns]

    if debug:
        print(f"[DEBUG] Schema numerical_cols: {cols.numerical_cols()}")
        print(f"[DEBUG] actual numeric_feats → {all_num_feats}")
        print(f"[DEBUG] Schema ordinal_cols:   {cols.ordinal_cols()}")
        print(f"[DEBUG] actual ord_feats     → {all_ord_feats}")
        print(f"[DEBUG] Schema nominal_cols:  {cols.nominal_cols()}")
        print(f"[DEBUG] actual nom_feats     → {all_nom_feats}")

    # 3) Coerce numeric columns to numeric dtype (errors → NaN)
    if all_num_feats:
        df[all_num_feats] = df[all_num_feats].apply(pd.to_numeric, errors="coerce")

    # 4) Build X and y
    X = df[all_num_feats + all_ord_feats + all_nom_feats].copy()
    y = df[TARGET].astype(int)  # ensure integer dtype for classification

    # 5) Prepare ordinal columns for OrdinalEncoder (only if any exist)
    ordinal_pipe = None
    if all_ord_feats:
        X[all_ord_feats] = X[all_ord_feats].astype("string")
        X.loc[:, all_ord_feats] = X.loc[:, all_ord_feats].mask(
            X[all_ord_feats].isna(), np.nan
        )

        ordinal_categories = []
        for c in all_ord_feats:
            cats = list(X[c].dropna().unique())
            cats = [*cats, "MISSING"]
            ordinal_categories.append(cats)

        ordinal_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encode", OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype="int32"
            )),
        ])

    # 6) Prepare numeric pipeline
    numeric_pipe = None
    if all_num_feats:
        if model_type == "linear":
            num_imputer = SimpleImputer(strategy="median", add_indicator=True)
        else:
            num_imputer = IterativeImputer(random_state=0, add_indicator=True)

        numeric_pipe = Pipeline([
            ("impute", num_imputer),
            ("scale", StandardScaler()),
        ])

    # 7) Build ColumnTransformer blocks
    transformers = []

    if numeric_pipe is not None and all_num_feats:
        transformers.append(("num", numeric_pipe, all_num_feats))

    if all_ord_feats and ordinal_pipe is not None:
        if X[all_ord_feats].isna().any(axis=1).any():
            transformers.append((
                "ord_ind",
                MissingIndicator(missing_values=np.nan),
                all_ord_feats
            ))
            transformers.append(("ord", ordinal_pipe, all_ord_feats))
        else:
            transformers.append(("ord", ordinal_pipe, all_ord_feats))


    if all_nom_feats:
        nominal_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            # ▶ dense output straight from the encoder ◀
            ("encode", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False      # ✅ for scikit-learn ≥1.2
                # use `sparse=False` if you are on an older version
            )),
        ])
        transformers.append(("nom", nominal_pipe, all_nom_feats))

    # ----------– existing code until X_mat creation –-----------------------
    ct = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    ct.lower_, ct.upper_ = lower, upper
    X_mat = ct.fit_transform(X, y)

    # NEW: force dense so downstream pandas logic never sees CSR
    X_mat = _ensure_dense(X_mat)

    return X_mat, y, ct


def transform_preprocessor(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> tuple[np.ndarray, pd.Series]:
    """
    Transform new data using a fitted preprocessor.
    This version uses the same logic to only pick columns that exist,
    and to prepare ordinal columns correctly.  It also reuses the
    clipping bounds saved in transformer.lower_ and transformer.upper_.
    """
    # 1) Determine the target name
    cols = ColumnSchema()
    TARGET = cols.target_col()[0]  # e.g. "contested_success"

    # 2) Domain filter & clip using stored bounds
    df, _ = filter_and_clip(
        df,
        lower=transformer.lower_,
        upper=transformer.upper_
    )

    # 3) Rebuild feature lists, only keeping columns that exist
    all_num_feats = [c for c in cols.numerical_cols() if (c in df.columns and c != TARGET)]
    all_ord_feats = [c for c in cols.ordinal_cols() if c in df.columns]
    all_nom_feats = [c for c in cols.nominal_cols() if c in df.columns]

    # 4) Coerce numeric columns to numeric dtype (errors → NaN)
    if all_num_feats:
        df[all_num_feats] = df[all_num_feats].apply(pd.to_numeric, errors="coerce")

    # 5) Build X (features only)
    X = df[all_num_feats + all_ord_feats + all_nom_feats].copy()

    # 6) For ordinal features, replace NaN with "MISSING" so ordinal encoder matches fit time
    if all_ord_feats:
        X[all_ord_feats] = X[all_ord_feats].astype("string")
        X.loc[:, all_ord_feats] = X.loc[:, all_ord_feats].where(
            X.loc[:, all_ord_feats].notna(),
            "MISSING"
        )

    # 7) Extract y
    y = df[TARGET].astype(int)

    # ----------– existing code unchanged up to X_mat –----------------------
    X_mat = transformer.transform(X)

    # NEW: make sure we keep the same dense contract here too
    X_mat = _ensure_dense(X_mat)

    return X_mat, y


def inverse_transform_preprocessor(
    X_trans: np.ndarray,
    transformer: ColumnTransformer
) -> pd.DataFrame:
    """
    Invert each block of a ColumnTransformer back to its original features,
    skipping transformers that lack inverse_transform (e.g., MissingIndicator).

    Fix: if the numeric block is sparse, convert it to dense before calling
    StandardScaler.inverse_transform, because sparse centering is not allowed.
    """
    # 1) Recover original feature order
    orig_features: list[str] = []
    for name, _, cols in transformer.transformers_:
        if cols == 'drop':
            continue
        orig_features.extend(cols)

    parts = []
    start = 0
    n_rows = X_trans.shape[0]

    # 2) Loop through each transformer block
    for name, trans, cols in transformer.transformers_:
        if cols == 'drop':
            continue

        # The fitted sub‐estimator or pipeline for this block:
        fitted = transformer.named_transformers_[name]

        # 2a) Figure out how many output columns this block produced
        dummy = np.zeros((1, len(cols)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                out = fitted.transform(dummy)
            except Exception:
                out = dummy
        n_out = out.shape[1]

        # 2b) Slice that part of X_trans
        block = X_trans[:, start : start + n_out]
        start += n_out

        # 2c) Inverse‐transform depending on which pipeline this was
        if isinstance(fitted, MissingIndicator):
            # Skip MissingIndicator (no inverse_transform available)
            continue

        elif trans == 'passthrough':
            # If we had passed these features through, they come back as-is
            inv = block

        elif name == 'num':
            # —  NEW — : if block is sparse, convert to dense for StandardScaler
            if sparse.issparse(block):
                block_dense = block.toarray()
            else:
                block_dense = block

            scaler = fitted.named_steps['scale']
            inv_full = scaler.inverse_transform(block_dense)
            inv = inv_full[:, :len(cols)]

        else:
            # 1) First convert block to dense if needed:
            if sparse.issparse(block):
                block = block.toarray()

            # 2) Now call inverse_transform on the final step of the pipeline:
            if isinstance(fitted, Pipeline):
                last = list(fitted.named_steps.values())[-1]
                inv = last.inverse_transform(block)
            else:
                inv = fitted.inverse_transform(block)

        # 2d) Collect the inverted block into a DataFrame with the original column names
        parts.append(
            pd.DataFrame(inv, columns=cols, index=range(n_rows))
        )

    # 3) Concatenate all blocks side‐by‐side, then reorder to the original feature list
    df_orig = pd.concat(parts, axis=1)
    return df_orig[orig_features]


def prepare_for_mixed_and_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the rows *and* adds convenience covariates expected by the
    hierarchical and mixed-effects models.
    """
    # 1) Use ColumnSchema instead of _ColumnSchema (legacy)
    cols = ColumnSchema()
    TARGET = cols.target_col()[0]

    # 2) Domain cleaning + clipping
    df_clean, _ = filter_and_clip(df.copy())

    # 3) Category coding for batter
    df_clean["batter_id"] = df_clean["batter_id"].astype("category")

    # 4) Category coding for season
    df_clean["season_cat"] = df_clean["season"].astype("category")
    df_clean["season_idx"] = df_clean["season_cat"].cat.codes

    # 5) Category coding for pitcher
    df_clean["pitcher_cat"] = df_clean["pitcher_id"].astype("category")
    df_clean["pitcher_idx"] = df_clean["pitcher_cat"].cat.codes

    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# 6. Smoke test: only run when module executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.load_data.load_data import download_dataset, load_base_data
    from src.feature_engineering.column_schema import ColumnSchema
    from src.feature_engineering.feature_engineering import load_fe_dataset
    import json

    plays, players, pp, games = load_base_data()
    schema = ColumnSchema()
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

    print("[smoke] Column schema validation passed ✅\n")
    print(json.dumps(schema.as_dict(), indent=2))

    data_path = 'data/ml_dataset/ml_features.parquet'
    ml_df = load_fe_dataset(data_path, file_format='parquet')
    ml_df = ml_df[ml_df['is_contested'] == 1]

    ml_df = ml_df[NUMERICAL + NOMINAL + ORDINAL + TARGET]
    train_df, test_df = train_test_split(ml_df, test_size=0.2, random_state=42)

    # run with debug prints
    X_train_np, y_train, tf = fit_preprocessor(train_df, model_type='linear', debug=True)
    X_test_np,  y_test      = transform_preprocessor(test_df, tf)

    print("Processed shapes:", X_train_np.shape, X_test_np.shape)

    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train_np, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print("Shape:", df_back.shape, "→ original X_train shape before transform:", X_train_np.shape)
