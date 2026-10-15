"""
data_prep.py
Load, clean, and visualize the historical demand dataset.

Usage:
    python src/data_prep.py --input data/supply_chain_demand.csv --output outputs/cleaned_demand.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    """Load the raw demand CSV and parse dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw demand series:
      1. Time-aware interpolation for missing demand values.
      2. IQR-based outlier detection on non-promo/non-holiday days
         (days with promotions or holidays are expected to have elevated
         demand, so they're excluded from the outlier check).
      3. Outliers are replaced with a centered 7-day rolling median rather
         than dropped, to preserve a continuous daily series for forecasting.
    """
    df = df.set_index("date")
    df["demand_units"] = df["demand_units"].interpolate(method="time")

    normal_days = df[(df["promotion_flag"] == 0) & (df["holiday_flag"] == 0)]["demand_units"]
    q1, q3 = normal_days.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    is_outlier = (
        ((df["promotion_flag"] == 0) & (df["holiday_flag"] == 0)) &
        ((df["demand_units"] < lower) | (df["demand_units"] > upper))
    )

    rolling_median = df["demand_units"].rolling(7, center=True, min_periods=1).median()
    df.loc[is_outlier, "demand_units"] = rolling_median[is_outlier]

    df = df.reset_index()
    df["demand_units"] = df["demand_units"].round().astype(int)
    return df


def plot_trends(df: pd.DataFrame, out_dir: str = "outputs") -> None:
    """Generate and save trend / weekly / monthly seasonality plots."""
    # Overall trend with rolling mean
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["date"], df["demand_units"], linewidth=0.7, color="steelblue")
    ax.plot(df["date"], df["demand_units"].rolling(30).mean(),
            color="darkorange", linewidth=2, label="30-day rolling mean")
    ax.set_title("Daily Demand — Full History")
    ax.set_ylabel("Units")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/trend_overview.png", dpi=120)
    plt.close(fig)

    # Weekly seasonality
    df["dow"] = df["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_avg = df.groupby("dow")["demand_units"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(8, 4))
    weekly_avg.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Average Demand by Day of Week")
    ax.set_ylabel("Avg Units")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/weekly_seasonality.png", dpi=120)
    plt.close(fig)

    # Monthly seasonality
    df["month"] = df["date"].dt.month
    monthly_avg = df.groupby("month")["demand_units"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_avg.plot(kind="bar", ax=ax, color="indianred")
    ax.set_title("Average Demand by Month")
    ax.set_ylabel("Avg Units")
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/monthly_seasonality.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Load, clean, and visualize demand data.")
    parser.add_argument("--input", default="data/supply_chain_demand.csv")
    parser.add_argument("--output", default="outputs/cleaned_demand.csv")
    parser.add_argument("--plots_dir", default="outputs")
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df)} rows | missing values: {df['demand_units'].isna().sum()}")

    df_clean = clean_data(df)
    df_clean.to_csv(args.output, index=False)
    print(f"Cleaned data saved to {args.output}")

    plot_trends(df_clean, out_dir=args.plots_dir)
    print(f"Trend/seasonality plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
