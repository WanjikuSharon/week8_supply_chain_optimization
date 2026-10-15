"""
distribution_lp.py
Solve a plant -> distribution-center transportation problem with PuLP:
minimize total shipping cost subject to plant capacity and DC demand constraints.

Usage:
    python src/distribution_lp.py --forecast_csv outputs/forecast.csv --horizon 45
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# Illustrative network — edit to match your real plants/DCs/costs
PLANTS = ["Plant_A", "Plant_B", "Plant_C"]
DCS = ["DC_Our_Region", "DC_North", "DC_South", "DC_West"]

CAPACITY = {"Plant_A": 9000, "Plant_B": 7000, "Plant_C": 6000}

# Illustrative demand for the other three DCs; DC_Our_Region is populated
# from the Prophet forecast at runtime.
OTHER_DC_DEMAND = {"DC_North": 5200, "DC_South": 4300, "DC_West": 3800}

COST = {
    ("Plant_A", "DC_Our_Region"): 2.10, ("Plant_A", "DC_North"): 3.40,
    ("Plant_A", "DC_South"): 4.50,      ("Plant_A", "DC_West"): 5.80,
    ("Plant_B", "DC_Our_Region"): 3.90, ("Plant_B", "DC_North"): 1.80,
    ("Plant_B", "DC_South"): 4.10,      ("Plant_B", "DC_West"): 4.70,
    ("Plant_C", "DC_Our_Region"): 5.20, ("Plant_C", "DC_North"): 4.60,
    ("Plant_C", "DC_South"): 2.00,      ("Plant_C", "DC_West"): 3.30,
}


def build_demand(forecast_avg_daily_demand: float) -> dict:
    demand = {"DC_Our_Region": round(forecast_avg_daily_demand * 7)}
    demand.update(OTHER_DC_DEMAND)
    return demand


def solve_distribution(demand: dict, capacity: dict = CAPACITY, cost: dict = COST):
    """Build and solve the transportation LP. Returns (status, total_cost, results_df)."""
    prob = LpProblem("Distribution_Cost_Minimization", LpMinimize)
    ship = {(p, d): LpVariable(f"ship_{p}_{d}", lowBound=0) for p in PLANTS for d in DCS}

    prob += lpSum(ship[p, d] * cost[p, d] for p in PLANTS for d in DCS)

    for p in PLANTS:
        prob += lpSum(ship[p, d] for d in DCS) <= capacity[p], f"Capacity_{p}"
    for d in DCS:
        prob += lpSum(ship[p, d] for p in PLANTS) >= demand[d], f"Demand_{d}"

    prob.solve()

    results = []
    for p in PLANTS:
        for d in DCS:
            qty = ship[p, d].value()
            if qty and qty > 0.5:
                results.append({"Plant": p, "DC": d, "Units": round(qty), "Cost": round(qty * cost[p, d], 2)})

    results_df = pd.DataFrame(results).sort_values(["Plant", "DC"])
    return LpStatus[prob.status], value(prob.objective), results_df


def plot_shipping_plan(results_df: pd.DataFrame, out_dir: str = "outputs") -> None:
    pivot = results_df.pivot(index="Plant", columns="DC", values="Units").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("Units shipped (weekly)")
    ax.set_title("Optimal Plant -> DC Shipping Plan")
    ax.legend(title="DC", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shipping_plan.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Solve plant->DC distribution LP.")
    parser.add_argument("--forecast_csv", default="outputs/forecast.csv")
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--output", default="outputs/shipping_plan.csv")
    parser.add_argument("--plots_dir", default="outputs")
    args = parser.parse_args()

    forecast_df = pd.read_csv(args.forecast_csv, parse_dates=["ds"])
    avg_daily_demand = forecast_df["yhat"].iloc[-args.horizon:].mean()

    demand = build_demand(avg_daily_demand)
    print("Weekly demand by DC:", demand)
    print("Total demand:", sum(demand.values()), "| Total capacity:", sum(CAPACITY.values()))

    status, total_cost, results_df = solve_distribution(demand)
    print(f"\nStatus: {status}")
    print(f"Total minimum weekly shipping cost: ${total_cost:,.2f}")

    results_df.to_csv(args.output, index=False)
    print(f"Shipping plan saved to {args.output}")

    plot_shipping_plan(results_df, out_dir=args.plots_dir)
    print(f"Shipping plan chart saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
