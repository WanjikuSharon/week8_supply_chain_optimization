"""
run_pipeline.py
End-to-end runner: data prep -> forecasting -> inventory optimization -> distribution LP.

Usage:
    python src/run_pipeline.py
"""

import argparse
import os
import pandas as pd

import data_prep
import forecasting
import inventory_optimization
import distribution_lp


def main():
    parser = argparse.ArgumentParser(description="Run the full supply chain pipeline end-to-end.")
    parser.add_argument("--input", default="data/supply_chain_demand.csv")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--lead_time", type=int, default=7)
    parser.add_argument("--service_level", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1/4: Data preparation")
    print("=" * 70)
    df = data_prep.load_data(args.input)
    df_clean = data_prep.clean_data(df)
    cleaned_path = f"{args.outputs_dir}/cleaned_demand.csv"
    df_clean.to_csv(cleaned_path, index=False)
    data_prep.plot_trends(df_clean, out_dir=args.outputs_dir)
    print(f"Cleaned data + plots saved to {args.outputs_dir}/\n")

    print("=" * 70)
    print("STEP 2/4: Forecasting")
    print("=" * 70)
    prophet_df = forecasting.prepare_prophet_df(df_clean)
    train, test = forecasting.train_test_split(prophet_df, args.horizon)
    model = forecasting.fit_model(train)
    fc = forecasting.forecast(model, prophet_df, args.horizon)
    metrics = forecasting.evaluate(test, fc)
    print(f"Test MAPE: {metrics['mape']:.2f}% | Test RMSE: {metrics['rmse']:.1f} units")
    forecast_path = f"{args.outputs_dir}/forecast.csv"
    fc.to_csv(forecast_path, index=False)
    forecasting.plot_forecast(model, fc, train["ds"].max(), out_dir=args.outputs_dir)
    print(f"Forecast + plots saved to {args.outputs_dir}/\n")

    print("=" * 70)
    print("STEP 3/4: Inventory optimization")
    print("=" * 70)
    test_dates = df_clean["date"].iloc[-args.horizon:]
    y_true = df_clean.set_index("date").loc[test_dates, "demand_units"]
    y_pred = fc.set_index("ds").loc[test_dates, "yhat"]
    residual_std = (y_true - y_pred).std()
    mean_forecast_demand = y_pred.mean()

    ss = inventory_optimization.compute_safety_stock(residual_std, args.lead_time, args.service_level)
    reorder_point = mean_forecast_demand * args.lead_time + ss["safety_stock"]
    print(f"Safety Stock: {ss['safety_stock']:.0f} units | Reorder Point: {reorder_point:.0f} units")

    historical_demand = df_clean["demand_units"].values
    no_ss_threshold = mean_forecast_demand * args.lead_time
    rate_no_ss, _ = inventory_optimization.simulate_policy(
        historical_demand, no_ss_threshold, no_ss_threshold, args.lead_time)
    rate_with_ss, _ = inventory_optimization.simulate_policy(
        historical_demand, reorder_point, reorder_point, args.lead_time)
    print(f"Stockout rate without SS: {rate_no_ss:.1%} | with SS: {rate_with_ss:.1%}")

    inventory_optimization.plot_stockout_comparison(
        rate_no_ss, rate_with_ss, 1 - args.service_level, out_dir=args.outputs_dir)
    print(f"Inventory plots saved to {args.outputs_dir}/\n")

    print("=" * 70)
    print("STEP 4/4: Distribution LP")
    print("=" * 70)
    demand = distribution_lp.build_demand(mean_forecast_demand)
    status, total_cost, results_df = distribution_lp.solve_distribution(demand)
    print(f"Status: {status} | Total minimum weekly shipping cost: ${total_cost:,.2f}")
    results_df.to_csv(f"{args.outputs_dir}/shipping_plan.csv", index=False)
    distribution_lp.plot_shipping_plan(results_df, out_dir=args.outputs_dir)
    print(f"Shipping plan + chart saved to {args.outputs_dir}/\n")

    print("Pipeline complete. All outputs in:", args.outputs_dir)


if __name__ == "__main__":
    main()
