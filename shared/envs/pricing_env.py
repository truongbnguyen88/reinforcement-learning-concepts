"""
Simulated retail pricing environment for RL experiments.

Provides:
  - Discrete action space (price multipliers)
  - Tabular state representation  (qty_bin)
  - Continuous/vector state representation (state_to_vec, for DQN/PG)
  - Episode table construction from historical data
  - Single-step transition: env_step()

Both the tabular and deep RL notebooks import from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ACTION_MULTS: np.ndarray = np.array(
    [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25], dtype=np.float32
)
N_ACTIONS: int = len(ACTION_MULTS)
BASELINE_ACTION_IDX: int = int(np.argmin(np.abs(ACTION_MULTS - 1.0)))  # index of 1.00×

# ---------------------------------------------------------------------------
# Episode table helpers
# ---------------------------------------------------------------------------

def make_top_products(data: pd.DataFrame, top_k: int = 200) -> list[str]:
    """Return stock codes of the top-k products by total quantity sold."""
    counts = data.groupby("StockCode")["qty"].sum().sort_values(ascending=False)
    return counts.head(top_k).index.astype(str).tolist()


def build_episode_table(data: pd.DataFrame, top_products: list[str]) -> pd.DataFrame:
    """Subset and sort data to the top products — used as the episode pool."""
    d = data[data["StockCode"].astype(str).isin(top_products)].copy()
    d["StockCode"] = d["StockCode"].astype(str)
    return d.sort_values(["StockCode", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabular state: qty_bin discretization
# ---------------------------------------------------------------------------

# Bin boundaries for lagged demand quantity.
# Chosen to give reasonable coverage over the UCI retail qty distribution.
_QTY_THRESHOLDS = [1, 10, 20, 30, 40, 50, 100, 200]
N_QTY_BINS: int = len(_QTY_THRESHOLDS) + 1  # 9 bins (0–8)


def qty_bin(q: float) -> int:
    """Map a quantity value to a discrete bin index (0–8)."""
    for i, t in enumerate(_QTY_THRESHOLDS):
        if q < t:
            return i
    return len(_QTY_THRESHOLDS)


# ---------------------------------------------------------------------------
# Continuous state: feature vector for DQN / policy gradient agents
# ---------------------------------------------------------------------------

STATE_DIM: int = 23  # 4 numerical + 7 dow-onehot + 12 month-onehot


def state_to_vec(row_dict: dict) -> np.ndarray:
    """
    Convert a state dict to a float32 vector of shape (STATE_DIM,).

    Numerical features are log/clipped and scaled to roughly [-5, 5].
    Categorical features (dow, month) are one-hot encoded.
    """
    price   = float(row_dict["price"])
    lag1    = float(row_dict["qty_lag1"])
    lag7    = float(row_dict["qty_lag7"])
    n_txn   = float(row_dict["n_txn"])

    numerical = np.array([
        np.log(np.clip(price, 1e-6, 1e6)) / 10.0,
        np.clip(lag1,  0.0, 1e6) / 200.0,
        np.clip(lag7,  0.0, 1e6) / 200.0,
        np.clip(n_txn, 0.0, 1e6) / 20.0,
    ], dtype=np.float32)
    numerical = np.clip(numerical, -5.0, 5.0)

    dow = int(row_dict["dow"]) % 7
    dow_oh = np.zeros(7, dtype=np.float32)
    dow_oh[dow] = 1.0

    month = max(1, min(12, int(row_dict["month"])))
    month_oh = np.zeros(12, dtype=np.float32)
    month_oh[month - 1] = 1.0

    vec = np.concatenate([numerical, dow_oh, month_oh])
    assert vec.shape[0] == STATE_DIM
    return vec


# ---------------------------------------------------------------------------
# Demand prediction
# ---------------------------------------------------------------------------

def predict_qty(model: Pipeline, row_dict: dict, price: float) -> float:
    """
    Predict demand quantity using the fitted demand model.

    Applies safe clamping on inputs and clips the log-space output to
    prevent extreme predictions from destabilizing training.
    """
    price  = max(1e-6, float(price))
    lag1   = max(0.0, float(row_dict["qty_lag1"]))
    lag7   = max(0.0, float(row_dict["qty_lag7"]))
    n_txn  = max(1.0, float(row_dict["n_txn"]))

    x = pd.DataFrame([{
        "log_price": np.log(price),
        "qty_lag1":  lag1,
        "qty_lag7":  lag7,
        "n_txn":     n_txn,
        "dow":       int(row_dict["dow"]),
        "month":     int(row_dict["month"]),
    }])

    try:
        log_qty = float(model.predict(x)[0])
        log_qty = np.clip(log_qty, -10.0, 10.0)
        return max(0.0, float(np.expm1(log_qty)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Environment step
# ---------------------------------------------------------------------------

def env_step(
    model: Pipeline,
    row_dict: dict,
    action_mult: float,
) -> tuple[dict, float, dict]:
    """
    Apply one pricing action and return (next_state, reward, next_state).

    Transition logic:
      - chosen_price = base_price × action_mult
      - predicted_qty = demand_model(chosen_price, context)
      - reward = chosen_price × predicted_qty  (revenue)
      - next state: lag features updated with predicted_qty; dow advances by 1

    Args:
        model:       Fitted demand model Pipeline.
        row_dict:    Current state as a feature dict.
        action_mult: Price multiplier from ACTION_MULTS.

    Returns:
        (next_state_dict, reward, next_state_dict)
        The third element is identical to the first for API symmetry.
    """
    base_price = float(row_dict["price"])
    chosen_price = base_price * float(action_mult)

    qty_hat = predict_qty(model, row_dict, chosen_price)
    reward = chosen_price * qty_hat

    next_state = {
        "price":    chosen_price,
        "qty_lag1": qty_hat,
        "qty_lag7": float(row_dict["qty_lag1"]),  # yesterday's lag1 becomes lag7 proxy
        "n_txn":    float(row_dict["n_txn"]),
        "dow":      (int(row_dict["dow"]) + 1) % 7,
        "month":    int(row_dict["month"]),
    }
    return next_state, reward, next_state
