"""
Policy evaluation utilities for the retail pricing RL environment.

Provides:
  - evaluate_policy()    paired statistical evaluation of one policy vs baseline
  - compare_policies()   multi-policy fair rollout comparison
  - bootstrap_ci()       bootstrap confidence interval for mean lift
  - cohens_d()           effect size
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline

from shared.envs.pricing_env import (
    ACTION_MULTS,
    BASELINE_ACTION_IDX,
    env_step,
    predict_qty,
    qty_bin,
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    diff: np.ndarray,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of diff."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size for paired comparison (x - y)."""
    diff = np.asarray(x) - np.asarray(y)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


# ---------------------------------------------------------------------------
# Single-policy evaluation (tabular interface)
# ---------------------------------------------------------------------------

def evaluate_policy(
    Q: np.ndarray,
    actions: np.ndarray,
    episode_table: pd.DataFrame,
    demand_model: Pipeline,
    n_eval: int = 5_000,
    seed: int = 420,
    do_bootstrap: bool = True,
    n_boot: int = 2_000,
) -> dict:
    """
    Paired evaluation of a tabular Q-policy vs the baseline (1.0× multiplier).

    Statistical tests:
      - Paired t-test (ttest_rel)
      - Wilcoxon signed-rank test (robust to heavy tails)
      - Sign/binomial test for win-rate
      - Bootstrap CI for mean revenue lift (optional)

    Args:
        Q:             Q-table, shape (n_states, n_actions).
        actions:       Action multiplier array aligned with Q columns.
        episode_table: DataFrame of (product, date) rows used as initial states.
        demand_model:  Fitted demand model Pipeline.
        n_eval:        Number of paired episodes to sample.
        seed:          RNG seed for reproducibility.
        do_bootstrap:  Whether to compute bootstrap CI (slower).
        n_boot:        Bootstrap resamples.

    Returns:
        Dictionary of metrics, revenues, and statistical test results.
    """
    rng = np.random.default_rng(seed)

    rl_revenues, baseline_revenues = [], []
    rl_qtys, baseline_qtys = [], []
    rl_prices, baseline_prices = [], []

    for _ in range(n_eval):
        idx = int(rng.integers(0, len(episode_table)))
        row = episode_table.iloc[idx].to_dict()

        state = qty_bin(float(row["qty_lag1"]))
        base_price = float(row["price"])

        # RL policy
        best_a = int(np.argmax(Q[state]))
        rl_mult = float(actions[best_a])
        rl_price = base_price * rl_mult
        rl_qty = predict_qty(demand_model, row, rl_price)
        rl_rev = rl_price * rl_qty

        # Baseline (1.0× multiplier)
        bl_qty = predict_qty(demand_model, row, base_price)
        bl_rev = base_price * bl_qty

        rl_revenues.append(rl_rev);    rl_qtys.append(rl_qty);    rl_prices.append(rl_price)
        baseline_revenues.append(bl_rev); baseline_qtys.append(bl_qty); baseline_prices.append(base_price)

    rl_rev = np.asarray(rl_revenues, float)
    bl_rev = np.asarray(baseline_revenues, float)
    diff   = rl_rev - bl_rev

    rl_mean, bl_mean = float(rl_rev.mean()), float(bl_rev.mean())
    lift_pct = 100.0 * (rl_mean - bl_mean) / (bl_mean + 1e-12)

    t_stat, t_p = stats.ttest_rel(rl_rev, bl_rev)
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    wins   = int((diff > 0).sum())
    ties   = int((diff == 0).sum())
    n_eff  = n_eval - ties
    binom  = stats.binomtest(wins, n_eff, p=0.5) if n_eff > 0 else None

    ci = bootstrap_ci(diff, n_boot=n_boot) if do_bootstrap else (np.nan, np.nan)

    return {
        "n_eval": n_eval,
        "rl_avg_revenue":       rl_mean,
        "baseline_avg_revenue": bl_mean,
        "revenue_lift_pct":     lift_pct,
        "rl_avg_quantity":      float(np.mean(rl_qtys)),
        "baseline_avg_quantity":float(np.mean(baseline_qtys)),
        "rl_avg_price":         float(np.mean(rl_prices)),
        "baseline_avg_price":   float(np.mean(baseline_prices)),
        "rl_revenue_std":       float(rl_rev.std(ddof=1)),
        "baseline_revenue_std": float(bl_rev.std(ddof=1)),
        "rl_revenues":          rl_rev,
        "baseline_revenues":    bl_rev,
        "stats": {
            "paired_ttest":     {"t": float(t_stat), "p": float(t_p)},
            "wilcoxon":         {"stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
                                 "p": float(w_p) if not np.isnan(w_p) else np.nan},
            "sign_test_binom":  {"wins": wins, "ties": ties, "n_eff": n_eff,
                                 "p": float(binom.pvalue) if binom else np.nan},
            "bootstrap_ci_95":  ci,
            "cohens_d":         cohens_d(rl_rev, bl_rev),
        },
    }


# ---------------------------------------------------------------------------
# Multi-policy fair rollout comparison
# ---------------------------------------------------------------------------

def compare_policies(
    policies: dict[str, callable],
    episode_table: pd.DataFrame,
    demand_model: Pipeline,
    actions: np.ndarray = ACTION_MULTS,
    n_eval: int = 1_000,
    horizon: int = 20,
    seed: int = 42,
) -> dict:
    """
    Evaluate multiple policies on the same initial states (fair paired comparison).

    Args:
        policies:      Dict of {name: policy_fn} where policy_fn(state_dict) -> action_idx.
        episode_table: Pool of initial states.
        demand_model:  Fitted demand model Pipeline.
        actions:       Action multiplier array.
        n_eval:        Number of episodes per policy.
        horizon:       Steps per episode.
        seed:          RNG seed.

    Returns:
        Dict of {name: {"returns": [...], "mean": float, "median": float,
                        "std": float, "min": float, "max": float,
                        "action_distribution": {mult: fraction}}}
    """
    rng = np.random.default_rng(seed)
    actions = np.asarray(actions, dtype=np.float32)

    initial_indices = [int(rng.integers(0, len(episode_table))) for _ in range(n_eval)]

    results = {name: {"returns": [], "actions_flat": []} for name in policies}

    for idx in initial_indices:
        init_row = episode_table.iloc[idx].to_dict()

        for name, policy_fn in policies.items():
            s = dict(init_row)
            ep_return = 0.0
            ep_actions = []

            for _ in range(horizon):
                a_idx = int(policy_fn(s))
                a_mult = float(actions[a_idx])
                s, r, s = env_step(demand_model, s, a_mult)
                ep_return += r
                ep_actions.append(a_idx)

            results[name]["returns"].append(ep_return)
            results[name]["actions_flat"].extend(ep_actions)

    # Summarise
    summary = {}
    for name, data in results.items():
        ret = np.asarray(data["returns"], float)
        acts = np.asarray(data["actions_flat"])
        action_counts = {float(actions[i]): float((acts == i).mean()) for i in range(len(actions))}
        summary[name] = {
            "returns":            ret,
            "mean":               float(ret.mean()),
            "median":             float(np.median(ret)),
            "std":                float(ret.std(ddof=1)),
            "min":                float(ret.min()),
            "max":                float(ret.max()),
            "action_distribution": action_counts,
        }

    # Win-rate matrix
    names = list(summary.keys())
    win_rates = {}
    for i, n1 in enumerate(names):
        for n2 in names:
            if n1 == n2:
                continue
            r1 = summary[n1]["returns"]
            r2 = summary[n2]["returns"]
            win_rates[f"{n1}_over_{n2}"] = float((r1 > r2).mean())

    for v in summary.values():
        v["win_rates"] = win_rates

    return summary
