"""
Demand model for the retail pricing RL environment.

Pipeline:
  load_data() -> preprocess() -> fit_demand_model()

The fitted sklearn Pipeline is the sole dependency of pricing_env.py.
"""

from __future__ import annotations

import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00352/Online%20Retail.xlsx"
)

FEATURES_NUM = ["log_price", "qty_lag1", "qty_lag7", "n_txn"]
FEATURES_CAT = ["dow", "month"]


def load_data(url: str = UCI_URL, verify_ssl: bool = False) -> pd.DataFrame:
    """Download the UCI Online Retail dataset and return the raw DataFrame."""
    print("Downloading UCI Online Retail dataset...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resp = requests.get(url, verify=verify_ssl)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw retail data and aggregate to daily product-level table.

    Returns columns: StockCode, date, qty, price, n_txn, dow, month,
                     qty_lag1, qty_lag7
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "StockCode", "Quantity", "UnitPrice"])

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[df["StockCode"].astype(str).str.len() > 0]

    df["date"] = df["InvoiceDate"].dt.date
    g = df.groupby(["StockCode", "date"], as_index=False).agg(
        qty=("Quantity", "sum"),
        price=("UnitPrice", "mean"),
        n_txn=("InvoiceNo", "nunique"),
    )

    g["date"] = pd.to_datetime(g["date"])
    g["dow"] = g["date"].dt.dayofweek
    g["month"] = g["date"].dt.month

    g = g.sort_values(["StockCode", "date"])
    g["qty_lag1"] = g.groupby("StockCode")["qty"].shift(1)
    g["qty_lag7"] = g.groupby("StockCode")["qty"].shift(7)
    g[["qty_lag1", "qty_lag7"]] = g[["qty_lag1", "qty_lag7"]].fillna(0)

    return g.reset_index(drop=True)


def fit_demand_model(data: pd.DataFrame) -> Pipeline:
    """
    Fit log-linear demand model:
      log(1 + qty) = f(log(price), qty_lag1, qty_lag7, n_txn, dow, month)

    Returns a fitted sklearn Pipeline.
    Price elasticity is interpretable as the coefficient on log_price.
    """
    d = data.copy()
    d["log_qty"] = np.log1p(d["qty"])
    d["log_price"] = np.log(d["price"])

    X = d[FEATURES_NUM + FEATURES_CAT]
    y = d["log_qty"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )
    model = Pipeline(steps=[("pre", pre), ("reg", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"  Demand model R² (log_qty, hold-out): {r2:.3f}")
    return model


def price_elasticity(model: Pipeline) -> float:
    """Return the estimated price elasticity from the fitted model."""
    return float(model.named_steps["reg"].coef_[0])
