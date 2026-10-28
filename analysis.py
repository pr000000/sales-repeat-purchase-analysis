# Programming Supplement
# Author: Peiran (Cynthia) Jing | GitHub: https://github.com/pr000000
# Multi-channel sales analysis + repeat-purchase prediction
# (mirrors my data-analysis internship; data here is synthetic)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ingest data from CSV ---
df = pd.read_csv("sales.csv")

# --- 2. Manage different data types ---
df["order_date"] = pd.to_datetime(df["order_date"])
df["category"]   = df["category"].astype("category")
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
print(df.dtypes.to_string(), "\n")

# --- 3. Wrangle data ---
# Raw channel labels are inconsistent ("tmall  ", "JD", "jd"); map to canonical
# names explicitly (str.title() would wrongly turn "JD" into "Jd").
CHANNELS = {"tmall": "Tmall", "jd": "JD", "pinduoduo": "Pinduoduo", "offline": "Offline"}
df["channel"] = df["channel"].str.strip().str.lower().map(CHANNELS).astype("category")
# Impute missing prices with the per-category median (robust to outliers)
n_missing = df["unit_price"].isna().sum()
df["unit_price"] = df["unit_price"].fillna(
    df.groupby("category", observed=True)["unit_price"].transform("median"))
df["revenue"] = df["units"] * df["unit_price"]
df["month"]   = df["order_date"].dt.to_period("M")
print(f"Imputed {n_missing} missing prices; rows = {len(df)}\n")

# --- 4. Write your own function (analysis) ---
def channel_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per-channel KPIs: revenue, orders, avg order value, repeat rate, share."""
    out = data.groupby("channel", observed=True).agg(
        revenue=("revenue", "sum"), orders=("order_id", "count"),
        avg_order_value=("revenue", "mean"), repeat_rate=("is_repeat", "mean"))
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    return out.sort_values("revenue", ascending=False).round(2)

summary = channel_summary(df)
print("Per-channel KPI summary"); print(summary.to_string())
monthly = df.pivot_table(index="month", columns="channel", values="revenue",
                         aggfunc="sum", observed=True).round(0)
print("\nMonthly revenue by channel"); print(monthly.to_string())

# --- 5. Logistic regression from scratch (NumPy, gradient descent) ---
# Features: units and unit_price only. revenue = units * unit_price, so adding
# it would make the coefficients collinear and hard to interpret.
def fit_logistic(X, y, lr=0.1, epochs=500):
    """Train logistic regression with batch gradient descent; return weights."""
    Xb = np.c_[np.ones(len(X)), X]                 # add intercept column
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-Xb @ w))              # sigmoid
        w -= lr * Xb.T @ (p - y) / len(y)          # gradient step
    return w

feat_cols = ["units", "unit_price"]
X = df[feat_cols].to_numpy(float); y = df["is_repeat"].to_numpy(float)
rng = np.random.default_rng(42)
train = rng.random(len(df)) < 0.75                 # 75/25 train-test split
mu, sd = X[train].mean(0), X[train].std(0)         # fit scaler on TRAIN only
Xs = (X - mu) / sd                                 # -> no test-set leakage
w = fit_logistic(Xs[train], y[train])
prob = 1 / (1 + np.exp(-np.c_[np.ones((~train).sum()), Xs[~train]] @ w))
acc = ((prob > 0.5) == y[~train]).mean()
base = max(y[~train].mean(), 1 - y[~train].mean()) # always predict majority
print(f"\nTest accuracy = {acc:.2%} | majority-class baseline = {base:.2%}"
      f" | lift = {acc - base:+.2%}")
print("Weights (intercept, units, unit_price):", np.round(w, 3))

# --- 6. Visualize data ---
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
monthly.plot(ax=ax[0], marker="o")
ax[0].set_title("Monthly Revenue by Channel"); ax[0].set_ylabel("Revenue (RMB)")
ax[0].set_xlabel(""); ax[0].legend(fontsize=7)
ax[1].bar(feat_cols, w[1:], color="#E0A458", edgecolor="black")
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_title("Repeat-Purchase Model Weights (standardized)")
plt.tight_layout(); plt.savefig("chart.png", dpi=150, bbox_inches="tight")
