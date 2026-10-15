"""
inventory_optimization.py
Compute safety stock / reorder point and simulate stockout rates
with vs. without safety stock.

Usage:
    python src/inventory_optimization.py \
        --demand_csv outputs/cleaned_demand.csv \
        --forecast_csv outputs/forecast.csv \
        --lead_time 7 --service_level 0.95
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def compute_safety_stock(residual_std: float, lead_time_days: int, service_level: float) -> dict:
    """
    Safety Stock  = z * sigma_d * sqrt(L)
    Reorder Point = (avg_daily_demand * L) + Safety Stock

    residual_std   : std dev of forecast error (demand-during-lead-time uncertainty proxy)
    lead_time_days : replenishment lead time in days
    service_level  : target cycle service level (e.g. 0.95 for 95%)
    """
    z = norm.ppf(service_level)
    safety_stock = z * residual_std * np.sqrt(lead_time_days)
    return {"z": z, "safety_stock": safety_stock}


def simulate_policy(historical_demand: np.ndarray, threshold: float, order_up_to: float,
                     lead_time: int, n_days: int = 500, n_reps: int = 200, seed: int = 42) -> tuple:
    """
    Monte Carlo simulation of a reorder-point inventory policy.

    - Demand each day is bootstrapped from historical daily demand.
    - When on-hand inventory <= threshold and no order is pending, a replenishment
      order for `order_up_to` units is placed and arrives after `lead_time` days.
    - A stockout day = a day where demand exceeds available on-hand inventory.

    Returns (mean_stockout_rate, std_stockout_rate) across n_reps simulations.
    """
    rng = np.random.default_rng(seed)
    stockout_rates = []

    for _ in range(n_reps):
        on_hand = order_up_to
        pending_orders = []
        stockout_days = 0

        for day in range(n_days):
            arriving = [q for (d, q) in pending_orders if d == day]
            on_hand += sum(arriving)
            pending_orders = [(d, q) for (d, q) in pending_orders if d != day]

            demand_today = rng.choice(historical_demand)
            if demand_today > on_hand:
                stockout_days += 1
                on_hand = 0
            else:
                on_hand -= demand_today

            if on_hand <= threshold and not pending_orders:
                pending_orders.append((day + lead_time, order_up_to))

        stockout_rates.append(stockout_days / n_days)

    return float(np.mean(stockout_rates)), float(np.std(stockout_rates))


def plot_stockout_comparison(rate_no_ss: float, rate_with_ss: float, target_rate: float,
                              out_dir: str = "outputs") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Without Safety Stock", "With Safety Stock"],
           [rate_no_ss, rate_with_ss], color=["indianred", "seagreen"])
    ax.axhline(target_rate, color="black", linestyle="--", label=f"{target_rate:.0%} target stockout rate")
    ax.set_ylabel("Simulated stockout rate (fraction of days)")
    ax.set_title("Impact of Safety Stock on Stockout Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/stockout_comparison.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute safety stock / ROP and simulate stockouts.")
    parser.add_argument("--demand_csv", default="outputs/cleaned_demand.csv")
    parser.add_argument("--forecast_csv", default="outputs/forecast.csv")
    parser.add_argument("--lead_time", type=int, default=7)
    parser.add_argument("--service_level", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=45,
                         help="Must match the horizon used in forecasting.py")
    parser.add_argument("--plots_dir", default="outputs")
    args = parser.parse_args()

    demand_df = pd.read_csv(args.demand_csv, parse_dates=["date"])
    forecast_df = pd.read_csv(args.forecast_csv, parse_dates=["ds"])

    # Use residuals over the same held-out window used in forecasting.py
    test_dates = demand_df["date"].iloc[-args.horizon:]
    y_true = demand_df.set_index("date").loc[test_dates, "demand_units"]
    y_pred = forecast_df.set_index("ds").loc[test_dates, "yhat"]
    residual_std = (y_true - y_pred).std()

    mean_forecast_demand = forecast_df.set_index("ds").loc[test_dates, "yhat"].mean()

    ss = compute_safety_stock(residual_std, args.lead_time, args.service_level)
    reorder_point = mean_forecast_demand * args.lead_time + ss["safety_stock"]

    print(f"z-score for {args.service_level:.0%} service level: {ss['z']:.3f}")
    print(f"Residual std: {residual_std:.1f} units")
    print(f"Avg forecasted daily demand: {mean_forecast_demand:.1f} units")
    print(f"Safety Stock: {ss['safety_stock']:.0f} units")
    print(f"Reorder Point: {reorder_point:.0f} units")

    historical_demand = demand_df["demand_units"].values
    no_ss_threshold = mean_forecast_demand * args.lead_time
    with_ss_threshold = reorder_point

    rate_no_ss, _ = simulate_policy(historical_demand, no_ss_threshold, no_ss_threshold, args.lead_time)
    rate_with_ss, _ = simulate_policy(historical_demand, with_ss_threshold, with_ss_threshold, args.lead_time)

    print(f"\nStockout rate WITHOUT safety stock: {rate_no_ss:.1%}")
    print(f"Stockout rate WITH safety stock:    {rate_with_ss:.1%}")
    print(f"Target stockout rate (1 - service level): {1 - args.service_level:.0%}")

    plot_stockout_comparison(rate_no_ss, rate_with_ss, 1 - args.service_level, out_dir=args.plots_dir)
    print(f"\nStockout comparison plot saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
