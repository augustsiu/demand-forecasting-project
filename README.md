# Demand Forecasting Project

This project predicts daily product **demand** for a retail business. It looks at product, pricing, promotion and inventory data. It covers the full workflow:

1. **Exploratory data analysis and feature engineering** (`1_analysis.ipynb`)
2. **Model training and tuning** with XGBoost (`3_machine_learning.ipynb`)
3. **An interactive web app** that serves predictions with Streamlit (`app.py`)

## Project Structure

| File | Description |
|---|---|
| `demand_forecasting.csv` | Raw dataset (76,000 rows, 16 columns) |
| `1_analysis.ipynb` | Data cleaning checks, feature engineering and visual EDA |
| `2_preprocessed_demand_forecasting.csv` | Output of the analysis notebook: the raw data plus engineered features |
| `3_machine_learning.ipynb` | Trains, tunes and evaluates an XGBoost regressor, then saves the model |
| `xgboost_demand_model.pkl` | Trained XGBoost model (pickled) |
| `label_encoder.pkl` | Dictionary of fitted `LabelEncoder`s (`{"Category": LabelEncoder}`) |
| `app.py` | Streamlit app for making demand predictions |
| `requirements.txt` | Pinned dependencies for running the app (used by Streamlit Cloud) |
| `requirements-dev.txt` | Extra dependencies for running the notebooks |

## Dataset

- **Rows:** 76,000. Each row is one product in one store on one day.
- **Date range:** 2022-01-01 to 2024-01-30 (760 days)
- **Stores:** 5 (`S001`–`S005`); **Products:** 20 (`P0001`–`P0020`)
- **Missing values:** none; **Duplicate rows:** none

| Column | Type | Description |
|---|---|---|
| Date | date | Day of the observation |
| Store ID / Product ID | text | Store and product identifiers |
| Category | text | Clothing, Electronics, Furniture, Groceries, Toys |
| Region | text | North, South, East, West |
| Inventory Level | int | Units in stock |
| Units Sold | int | Units sold that day |
| Units Ordered | int | Units ordered for restocking |
| Price | float | Product price |
| Discount | int | Discount % (0, 5, 10, 15, 20, 25) |
| Weather Condition | text | Sunny, Cloudy, Rainy, Snowy |
| Promotion | 0/1 | Whether a promotion was running |
| Competitor Pricing | float | Competitor's price for the product |
| Seasonality | text | Spring, Summer, Autumn, Winter |
| Epidemic | 0/1 | Whether an epidemic period was in effect |
| **Demand** | int | **Target variable** |

## 1. Analysis (`1_analysis.ipynb`)

### Engineered features
These are saved to `2_preprocessed_demand_forecasting.csv`:
- `Year`, `Month`, `Day`, `Weekday`, taken from `Date`
- `Discounted Price = Price × (1 − Discount / 100)`
- `Sell Through Rate = Units Sold / Inventory Level`. This is left blank (NaN) for the 406 rows with zero inventory.

### Key findings
| Factor | Average demand |
|---|---|
| **Promotion** | 123.3 with a promotion vs 95.0 without (about +30%) |
| **Epidemic** | 70.2 during an epidemic vs 112.9 otherwise (about −38%) |
| **Category** | Groceries 121.0 > Clothing 112.6 > Electronics 97.5 > Toys 92.6 > Furniture 73.6 |
| **Weather** | Sunny 115.2 > Cloudy 105.4 > Rainy 95.1 > Snowy 94.0 |
| **Season** | Summer 112.9 > Autumn ≈ Winter 103.4 > Spring 97.7 |

- Groceries have by far the highest **total** demand (about 3.68M units), mostly because they make up 40% of all rows.
- The North region has the highest total demand because it has twice as many rows as each other region.
- Monthly averages peak in June and August and are lowest in May.

The notebook also includes plots of: demand distribution, inventory vs units sold, demand by category and weather, average demand by month, total daily demand over time, promotion impact, discounted price vs demand, seasonality and epidemic impact.

## 2. Machine Learning (`3_machine_learning.ipynb`)

- **Features:** `Price`, `Discount`, `Inventory Level`, `Promotion`, `Competitor Pricing`, `Category`
- **Target:** `Demand`
- **Encoding:** `Category` is label-encoded (Clothing=0, Electronics=1, Furniture=2, Groceries=3, Toys=4)
- **Scaling:** none. XGBoost is tree-based, so it doesn't need feature scaling.
- **Split:** 80% train / 20% test (`random_state=42`)
- **Model:** `XGBRegressor(objective="reg:squarederror")`
- **Tuning:** `RandomizedSearchCV` tries 25 combinations with 3-fold cross-validation, minimising MSE, over these settings:
  - `n_estimators` [200, 300, 500], `max_depth` [3, 4, 6, 8], `learning_rate` [0.01, 0.05, 0.1]
  - `subsample` [0.7, 0.8, 1.0], `colsample_bytree` [0.7, 0.8, 1.0], `min_child_weight` [1, 3, 5]

### Results (saved model `xgboost_demand_model.pkl`)
Best parameters: `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0, min_child_weight=1`

| Metric | Test set |
|---|---|
| MAE | 27.15 |
| RMSE | 35.53 |
| R² | 0.428 |

**Feature importance:** Promotion (0.65) > Category (0.20) > Price (0.08) > Competitor Pricing (0.03) > Discount (0.02) > Inventory Level (0.02)

The model explains about 43% of the variation in demand, and its predictions are off by about 27 units on average. The mean demand is about 104 units, so this is a fair baseline rather than a highly accurate model.

> The search now uses `random_state=42` so results can be reproduced. If you re-run the notebook, you may get slightly different best parameters than the saved model. One test run gave MAE 26.93, RMSE 35.46 and R² 0.431. Re-running also overwrites both `.pkl` files.

## 3. Web App (`app.py`)

**Live demo:** _add your Streamlit Cloud link here_

The Streamlit app loads the saved model and encoder. You enter Price, Discount, Inventory Level, Promotion, Competitor Pricing and Category, and it shows the predicted demand in units.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # or requirements.txt for the app only

# Optional: re-run the notebooks in order
jupyter notebook                   # run 1_analysis.ipynb, then 3_machine_learning.ipynb

# Launch the app
streamlit run app.py
```

## Limitations & Possible Improvements

- **Random train/test split on time-series data.** Future dates can end up in training, which can make the scores look better than they would be in real use. A date-based split (for example, train on 2022–2023 and test on Jan 2024) would be more realistic.
- **Limited feature set.** The model ignores Region, Weather, Seasonality, Epidemic, Store/Product IDs and the date features from the analysis notebook. Epidemic, Weather and Seasonality all clearly affect demand, so adding them would probably improve accuracy.
- **Label encoding** gives categories an artificial order. One-hot encoding, or XGBoost's native categorical support, avoids this.
- `Units Sold` is highly correlated with Demand (r ≈ 0.83). It was correctly **left out** of the features, because it wouldn't be known before the day it's forecasting.
