"""
forecasting.py
Build a Prophet demand forecast with external regressors and evaluate it.

Usage:
    python src/forecasting.py --input outputs/cleaned_demand.csv --horizon 45 --output outputs/forecast.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

REGRESSORS = ["promotion_flag", "holiday_flag", "temperature_c"]


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to Prophet's required ds/y schema."""
    prophet_df = df[["date", "demand_units"] + REGRESSORS].copy()
    prophet_df.columns = ["ds", "y"] + REGRESSORS
    return prophet_df


def train_test_split(prophet_df: pd.DataFrame, horizon: int):
    """Chronological split — last `horizon` days held out for evaluation."""
    train = prophet_df.iloc[:-horizon].copy()
    test = prophet_df.iloc[-horizon:].copy()
    return train, test


def fit_model(train: pd.DataFrame, interval_width: float = 0.95) -> Prophet:
    """Fit a Prophet model with weekly/yearly seasonality and external regressors."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=interval_width,
    )
    for reg in REGRESSORS:
        model.add_regressor(reg)
    model.fit(train)
    return model


def forecast(model: Prophet, prophet_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Generate forecast for `horizon` days beyond training data.
    Future regressor values are pulled from prophet_df (known/assumed calendar
    of promotions, holidays, and a weather estimate) — swap this merge for your
    real promo/holiday calendar and weather forecast in production.
    """
    future = model.make_future_dataframe(periods=horizon, freq="D")
    future = future.merge(prophet_df[["ds"] + REGRESSORS], on="ds", how="left")
    return model.predict(future)


def evaluate(test: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """Compute MAPE and RMSE on the held-out test set."""
    fc_indexed = fc.set_index("ds").loc[test["ds"]]
    y_true = test.set_index("ds")["y"]
    y_pred = fc_indexed["yhat"]

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    residuals = y_true - y_pred

    return {
        "mape": mape,
        "rmse": rmse,
        "residual_std": residuals.std(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_forecast(model: Prophet, fc: pd.DataFrame, split_date, out_dir: str = "outputs") -> None:
    fig1 = model.plot(fc)
    plt.axvline(split_date, color="red", linestyle="--", label="train/test split")
    plt.legend()
    plt.title("Prophet Forecast with 95% Prediction Interval")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/forecast_plot.png", dpi=120)
    plt.close(fig1)

    fig2 = model.plot_components(fc)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/forecast_components.png", dpi=120)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Forecast demand with Prophet.")
    parser.add_argument("--input", default="outputs/cleaned_demand.csv")
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--output", default="outputs/forecast.csv")
    parser.add_argument("--plots_dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    prophet_df = prepare_prophet_df(df)

    train, test = train_test_split(prophet_df, args.horizon)
    print(f"Train: {train['ds'].min().date()} to {train['ds'].max().date()} ({len(train)} days)")
    print(f"Test:  {test['ds'].min().date()} to {test['ds'].max().date()} ({len(test)} days)")

    model = fit_model(train)
    fc = forecast(model, prophet_df, args.horizon)

    metrics = evaluate(test, fc)
    print(f"Test MAPE: {metrics['mape']:.2f}%")
    print(f"Test RMSE: {metrics['rmse']:.1f} units")
    print(f"Residual std (used for safety stock): {metrics['residual_std']:.1f} units")

    fc.to_csv(args.output, index=False)
    print(f"Forecast saved to {args.output}")

    plot_forecast(model, fc, train["ds"].max(), out_dir=args.plots_dir)
    print(f"Forecast plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
