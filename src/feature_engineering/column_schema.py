"""
Column schema helper for VSRG ML dataset.

Separates columns into the **five** coherent buckets we now care about:

* **info_non_ml** – identifiers / metadata not used as model inputs
* **nominal**      – unordered categoricals
* **ordinal**      – ordered categoricals
* **numerical**    – continuous / discrete numeric predictors
* **target**       – label the model tries to predict

All downstream code should reference these accessors instead of hard‑coding
strings.  The helper also provides validation utilities and PyTest checks so a
schema / dataset mismatch is caught early.
"""
from typing import List, Dict, Literal
import pandas as pd
import pytest
import json


class ColumnSchema:
    # ─────────────────────────────────────────────────────────────────
    # 1️⃣ Non-ML informational columns (identifiers, metadata)
    # ─────────────────────────────────────────────────────────────────
    _INFO_NON_ML: List[str] = [
        "gameId",
        "playId",
        "receiverId",
        "cbId",
        "birthDate_x",
        "birthDate_y",
        "has_target",
    ]

    # ------------------------------------------------------------------
    # 2️⃣ Physical measurements (numeric)
    # ------------------------------------------------------------------
    _PHYSICAL_COLS: List[str] = [
        "rcv_height",
        "rcv_weight",
        "cb_height",
        "cb_weight",
    ]

    # ─────────────────────────────────────────────────────────────────
    # OPTIONAL: If you later derive quantile bins for height/weight,
    # uncomment this and add these to `nominal_cols()` or `ordinal_cols()`.
    # ─────────────────────────────────────────────────────────────────
    # _PHYSICAL_BINS: List[str] = [
    #     "rcv_height_bin",
    #     "rcv_weight_bin",
    #     "cb_height_bin",
    #     "cb_weight_bin",
    # ]

    # ------------------------------------------------------------------
    # 3️⃣ Position & context numeric features (unchanged)
    # ------------------------------------------------------------------
    _POSITION_COLS: List[str] = [
        "sep_receiver_cb", "sideline_dist", "pass_rush_sep", "sep_stddev", "hypo_sep",
        "passLength", "targetX", "targetY", "yardsToGo", "yardline_number", "timeToThrow",
        "time_rcv", "sep_velocity", "closing_speed", "min_other_sep", "rcv_s", "rcv_o",
        "cb_s", "cb_o", "leverage_angle", "contest_thresh", "s_max_last1s",
    ]

    # ------------------------------------------------------------------
    # ~~~ We are intentionally leaving _OUTCOME_NUMERICAL defined here
    #     in case you want it for a post‐model pipeline, but it should
    #     NOT be returned by numerical_cols() anymore. ~~~
    # ------------------------------------------------------------------
    _OUTCOME_NUMERICAL: List[str] = [
        "epa_change",
        "preWP",
        "postWP",
        "wp_delta",
        "vsrg_overall",
        "vsrg_oe",
    ]

    # ------------------------------------------------------------------
    # 5️⃣ Cluster-style categoricals (unchanged)
    # ------------------------------------------------------------------
    _CLUSTER_COLS: List[str] = [
        "rcv_club",
        "cb_club",
    ]

    # ------------------------------------------------------------------
    # 6️⃣ Ordinal features (unchanged)
    # ------------------------------------------------------------------
    _ZONE_COLS: List[str] = [
        "height_zone",
        "air_yards_bin",
        "down",
        "down_ctx",
    ]

    # ------------------------------------------------------------------
    # 7️⃣ Binary/nominal flags & categories (we remove these from training)
    #    _OUTCOME_CATEGORICAL is still defined but not used in any accessor.
    # ------------------------------------------------------------------
    _OUTCOME_CATEGORICAL: List[str] = [
        "is_contested",
        "was_tipped",
        "qb_was_hit",
        "pass_dir",
        "yardline_side",
        "is_play_action",
        "is_rpo",
        "passTippedAtLine",
        "caught_flag",
        "passResult",
        "time_imputed",
    ]

    # ------------------------------------------------------------------
    # 8️⃣ Route‐encoding + raw route (unchanged)
    # ------------------------------------------------------------------
    _ROUTE_COLS: List[str] = [
        "route_ANGLE", "route_CORNER", "route_CROSS", "route_FLAT",
        "route_GO", "route_HITCH", "route_IN", "route_OUT",
        "route_POST", "route_SCREEN", "route_SLANT", "route_WHEEL",
        "route",
    ]

    # ------------------------------------------------------------------
    # 9️⃣ Coverage columns (unchanged)
    # ------------------------------------------------------------------
    _COVERAGE_COLS: List[str] = [
        "cov_2-Man",
        "cov_Bracket",
        "cov_Cover 6-Left",
        "cov_Cover-0",
        "cov_Cover-1",
        "cov_Cover-1 Double",
        "cov_Cover-2",
        "cov_Cover-3",
        "cov_Cover-3 Cloud Left",
        "cov_Cover-3 Cloud Right",
        "cov_Cover-3 Double Cloud",
        "cov_Cover-3 Seam",
        "cov_Cover-6 Right",
        "cov_Goal Line",
        "cov_Miscellaneous",
        "cov_Prevent",
        "cov_Quarters",
        "cov_Red Zone",
    ]

    # ------------------------------------------------------------------
    # Target (unchanged)
    # ------------------------------------------------------------------
    _TARGET_COL: List[str] = ["contested_success"]

    # ------------------------------------------------------------------
    # Public accessors (modified)
    # ------------------------------------------------------------------

    def info_non_ml(self) -> List[str]:
        """Identification / metadata columns (never fed to the model)."""
        return self._INFO_NON_ML.copy()

    def target_col(self) -> List[str]:
        return self._TARGET_COL.copy()

    def nominal_cols(self) -> List[str]:
        """
        Unordered categorical features. We have removed _OUTCOME_CATEGORICAL
        to prevent leakage. If you uncomment _PHYSICAL_BINS above, add it here.
        """
        return (
            self._CLUSTER_COLS
            # + self._PHYSICAL_BINS      # ← Uncomment if you have quantile bins
            + self._ROUTE_COLS
            + self._COVERAGE_COLS
        ).copy()

    def ordinal_cols(self) -> List[str]:
        """
        Ordered categories. Examples: height_zone, air_yards_bin, down, down_ctx.
        """
        return self._ZONE_COLS.copy()

    def numerical_cols(self) -> List[str]:
        """
        Continuous / discrete numeric predictors. We removed _OUTCOME_NUMERICAL
        so that no post‐play outcome metrics leak into training.
        """
        return (self._PHYSICAL_COLS + self._POSITION_COLS).copy()

    def all_features(self) -> List[str]:
        """Nominal + ordinal + numerical (no target)."""
        return self.nominal_cols() + self.ordinal_cols() + self.numerical_cols()

    def modelling_columns(self) -> List[str]:
        """Feature set + target."""
        return self.all_features() + self.target_col()

    def all_columns(self) -> List[str]:
        """Every column expected in the ML parquet (info + features + target)."""
        return self.info_non_ml() + self.modelling_columns()

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "info_non_ml": self.info_non_ml(),
            "nominal": self.nominal_cols(),
            "ordinal": self.ordinal_cols(),
            "numerical": self.numerical_cols(),
            "target": self.target_col(),
        }



# ---------------------------------------------------------------------------
# Validation helpers --------------------------------------------------------
# ---------------------------------------------------------------------------

schema = ColumnSchema()


def validate_schema(s: ColumnSchema, df_cols: List[str] | None = None) -> None:
    """Assert bucket invariants and (optionally) match against a dataset."""

    GROUP_FUNCS = [
        s.info_non_ml,
        s.nominal_cols,
        s.ordinal_cols,
        s.numerical_cols,
        s.target_col,
    ]

    buckets = {f.__name__: f() for f in GROUP_FUNCS}

    # 1️⃣ No duplicates within any bucket
    for name, cols in buckets.items():
        assert len(cols) == len(set(cols)), f"Duplicates in bucket '{name}'"

    # 2️⃣ Buckets are disjoint (ignore target in overlap check)
    seen: set[str] = set()
    for name, cols in buckets.items():
        if name == "target_col":
            continue
        dup = seen.intersection(cols)
        assert not dup, f"Overlap across buckets: {dup}"
        seen.update(cols)

    # 3️⃣ Leakage‐keyword check (fail fast if a disallowed column appears)
    LEAKY_KEYWORDS = ["epa_", "preWP", "postWP", "vsrg_overall", "wp_delta"]
    for col in s.numerical_cols():
        for key in LEAKY_KEYWORDS:
            assert key not in col, f"Leaky column detected in numerical_cols(): {col}"

    # 4️⃣ Optionally compare to dataset headers
    if df_cols is not None:
        missing = set(s.all_columns()) - set(df_cols)
        extra = set(df_cols) - set(s.all_columns())
        assert not missing, f"Dataset missing columns: {missing}"
        assert not extra, f"Dataset has unexpected columns: {extra}"



# ---------------------------------------------------------------------------
# PyTest checks -------------------------------------------------------------
# ---------------------------------------------------------------------------

EXPECTED_BUCKET = {
    **{c: "info_non_ml" for c in schema.info_non_ml()},
    **{c: "nominal"    for c in schema.nominal_cols()},
    **{c: "ordinal"    for c in schema.ordinal_cols()},
    **{c: "numerical"  for c in schema.numerical_cols()},
    **{c: "target"     for c in schema.target_col()},
}


BUCKET_FUNCS = {
    "info_non_ml": schema.info_non_ml,
    "nominal": schema.nominal_cols,
    "ordinal": schema.ordinal_cols,
    "numerical": schema.numerical_cols,
    "target": schema.target_col,
}


@pytest.mark.parametrize("col,expected_bucket", EXPECTED_BUCKET.items())
def test_column_in_expected_bucket(col, expected_bucket):
    cols = BUCKET_FUNCS[expected_bucket]()
    assert col in cols, f"{col} missing from {expected_bucket} bucket"

    for bucket, func in BUCKET_FUNCS.items():
        if bucket != expected_bucket:
            assert col not in func(), f"{col} incorrectly present in {bucket} bucket"




# ---------------------------------------------------------------------------
# Smoke script --------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

