import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vsrg_by_zone(
    df: pd.DataFrame,
    zone_col: str = "height_zone",
    target_col: str = "contested_success",
    pred_prob_col: str = "pred_prob",
    to_grade: bool = True,
    figsize: tuple[int, int] = (8, 5),
    title: str = "Vertical Success Rate Grade: Actual vs. Predicted by Zone"
) -> None:
    """
    Plot actual vs. predicted success rates (VSRG) by vertical zone.
    Expects:
      - df must contain columns: zone_col, target_col, pred_prob_col
      - target_col is 0/1 indicator of actual success
      - pred_prob_col is model's predicted probability of success

    If to_grade=True, multiplies rates by 100 to display on 0-100 scale.
    """
    # 1) Aggregate actual success rates by zone
    vsrg_actual = (
        df
        .groupby(zone_col)[target_col]
        .mean()
        .reset_index(name="success_rate_actual")
    )

    # 2) Aggregate predicted probabilities by zone
    vsrg_pred = (
        df
        .groupby(zone_col)[pred_prob_col]
        .mean()
        .reset_index(name="success_rate_pred")
    )

    # 3) Merge into a single DataFrame
    vsrg_combined = pd.merge(vsrg_actual, vsrg_pred, on=zone_col)

    if to_grade:
        vsrg_combined["grade_actual"] = (
            vsrg_combined["success_rate_actual"] * 100
        )
        vsrg_combined["grade_pred"] = vsrg_combined["success_rate_pred"] * 100
        plot_vars = ["grade_actual", "grade_pred"]
        y_label = "Success Rate (0–100)"
    else:
        plot_vars = ["success_rate_actual", "success_rate_pred"]
        y_label = "Success Rate (0–1)"

    # 4) Melt for seaborn
    plot_df = vsrg_combined.melt(
        id_vars=zone_col,
        value_vars=plot_vars,
        var_name="metric",
        value_name="value"
    )

    # 5) Plot side-by-side bar chart
    plt.figure(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x=zone_col,
        y="value",
        hue="metric"
    )
    plt.title(title)
    plt.xlabel("Height Zone")
    plt.ylabel(y_label)
    # Adjust legend labels
    if to_grade:
        plt.legend(
            title="Metric",
            labels=["Actual Success Rate", "Predicted Success Rate"]
        )
    else:
        plt.legend(
            title="Metric",
            labels=["Actual Success Rate (0–1)", "Predicted Success Rate (0–1)"]
        )
    plt.tight_layout()
    plt.show() 
