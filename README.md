# Week 8 — Supply Chain Demand Forecasting & Inventory Optimization

End-to-end pipeline that forecasts SKU demand, converts forecast uncertainty into an
inventory policy (safety stock / reorder point), and solves a plant → distribution-center
shipping problem with linear programming.

## Repo structure

```
.
├── data/
│   └── supply_chain_demand.csv        # 3 years of synthetic-but-realistic daily demand
├── notebooks/
│   └── week8_supply_chain_optimization.ipynb   # full walkthrough, pre-executed
├── src/
│   ├── data_prep.py                   # load, clean, visualize trend/seasonality
│   ├── forecasting.py                 # Prophet model + MAPE/RMSE evaluation
│   ├── inventory_optimization.py      # safety stock, reorder point, stockout simulation
│   ├── distribution_lp.py             # PuLP transportation problem
│   └── run_pipeline.py                # runs all four steps end-to-end
├── outputs/                           # generated CSVs + PNG charts (created on run)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Option A — Jupyter notebook** (recommended for reviewing the full narrative + charts inline):
```bash
jupyter notebook notebooks/week8_supply_chain_optimization.ipynb
```

**Option B — modular scripts**, run individually:
```bash
python src/data_prep.py --input data/supply_chain_demand.csv --output outputs/cleaned_demand.csv
python src/forecasting.py --input outputs/cleaned_demand.csv --horizon 45 --output outputs/forecast.csv
python src/inventory_optimization.py --demand_csv outputs/cleaned_demand.csv --forecast_csv outputs/forecast.csv --lead_time 7 --service_level 0.95
python src/distribution_lp.py --forecast_csv outputs/forecast.csv --horizon 45
```

**Option C — full pipeline in one command:**
```bash
python src/run_pipeline.py
```

## Dataset

`data/supply_chain_demand.csv` is a synthetically generated but realistically structured
dataset (3 years, daily granularity) built because no dataset was supplied — it includes
trend, weekly seasonality, yearly seasonality (holiday + back-to-school peaks), promotion
lifts, holiday spikes, a weather effect, noise, and injected missing values / outliers so
the cleaning step is genuinely exercised. Columns:

| column | description |
|---|---|
| `date` | calendar date |
| `demand_units` | daily demand (target variable) |
| `promotion_flag` | 1 if the SKU was on promotion that day |
| `holiday_flag` | 1 if the date is a major retail holiday |
| `temperature_c` | daily average temperature (external/weather regressor) |

Swap in your own CSV with the same columns and everything downstream works unchanged.

## Key results (from the pre-executed notebook)

| Metric | Value |
|---|---|
| Forecast horizon | 45 days |
| Test MAPE | 5.79% |
| Test RMSE | 22.4 units |
| Lead time | 7 days |
| Target service level | 95% (z = 1.645) |
| Safety stock | 98 units |
| Reorder point | 2,199 units |
| Stockout rate — no safety stock | 1.5% |
| Stockout rate — with safety stock | 0.5% |
| Optimal weekly distribution cost (LP) | $38,180.00 |

## Assumptions

- **Lead time:** 7 days from order placement to stock arrival.
- **Service level:** 95% cycle service level (probability of not stocking out during a
  lead-time window).
- **Safety stock basis:** standard deviation of *Prophet forecast residuals* (not raw
  historical demand variance) scaled by √lead time — this ties the inventory buffer to
  how good the forecast actually is.
- **External regressors:** `promotion_flag` and `holiday_flag` are assumed known in
  advance (a standard retail planning calendar); `temperature_c` represents a weather
  driver and would be sourced from a forecast/climatology API in production.
- **LP distribution problem:** 3 plants, 4 DCs, fixed weekly capacity, feasible problem
  (total capacity ≥ total demand). One DC's demand is taken directly from the Prophet
  forecast; the other three are illustrative for the purposes of the LP formulation.

See the notebook's Markdown cells for the full assumption-by-assumption writeup.
