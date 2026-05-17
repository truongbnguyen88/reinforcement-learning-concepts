---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 22px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 1.8em; color: #1a1a2e; }
  h2 { font-size: 1.4em; color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }
  h3 { font-size: 1.1em; color: #0f3460; }
  code { background: #f4f4f8; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #f4f4f8; border-left: 4px solid #0f3460; }
  .warning { background: #fff3cd; border-left: 4px solid #e6a817; padding: 8px 12px; }
  .callout { background: #e8f4fd; border-left: 4px solid #1a73e8; padding: 8px 12px; }
  .callback { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 6px 10px; font-size: 0.88em; }
---

# Module 3 — Applied Tabular RL
## Pricing a Retail Product End-to-End

**Course: Reinforcement Learning Concepts**
*UCI Online Retail Dataset → Demand Model → Simulated Env → Tabular Policies → Statistical Evaluation*

---

## Where We Are

```
Module 1: RL Foundations          ← MDP, Bellman, value functions
Module 2: Tabular RL              ← Q-learning, Double Q, Expected SARSA, n-step
Module 3: Applied Tabular RL      ← *** YOU ARE HERE ***
Module 4: Deep RL (DQN)
Module 5: Policy Gradients
Module 6: RL in Production
```

<div class="callback">
<b>M1 callback:</b> We defined the MDP tuple (S, A, P, R, γ) abstractly.
<b>M2 callback:</b> We derived the Q-learning update rule and ran it on FrozenLake.
This module instantiates those abstractions on a real business problem.
</div>

---

## Learning Objectives

By the end of this module you should be able to:

**Understand conceptually**
- Why offline RL uses a learned simulator rather than live interaction
- How a log-linear demand model acts as the environment's transition function
- Why reward design choices matter more than algorithm choice in applied RL
- What statistical tests are appropriate for paired policy evaluation

**Implement**
- Build a simulated pricing MDP from historical transaction data
- Train tabular Q-learning and Expected SARSA on the pricing env
- Evaluate with paired t-test, Wilcoxon signed-rank, and bootstrap CI

**Recognize mistakes**
- Leakage through future demand signals in state features
- Unpaired evaluation inflating significance
- Reward hacking against a proxy objective
- Ignoring heavy-tailed revenue distributions in statistical tests

---

## The Business Problem

**Setting:** Online retailer with ~4,000 products selling in the UK (UCI Online Retail dataset, 2010–2011)

**Decision:** Each day, for each product, choose a **price multiplier** relative to historical price.

| | |
|---|---|
| **Action space** | 8 discrete multipliers: {0.90×, 0.95×, 1.00×, 1.05×, 1.10×, 1.15×, 1.20×, 1.25×} |
| **Objective** | Maximize daily revenue: $r = \text{price} \times \text{quantity}$ |
| **Challenge** | Price affects demand — setting price too high destroys volume |
| **Constraint** | We only have historical data; no live pricing A/B test |

This last constraint is the key: we cannot learn by *interacting* with real customers.

---

# Part 1: Offline RL Framing

---

## Online vs. Offline RL

**Online RL:** Agent interacts directly with the real environment to collect experience.

$$s_t \xrightarrow{\pi} a_t \xrightarrow{\text{real env}} (r_t, s_{t+1}) \xrightarrow{\text{update}} \pi$$

*Examples: game simulators, robotic arms, A/B test with live traffic*

---

**Offline RL (also called batch RL):** Agent learns entirely from a fixed dataset of logged transitions — no new interaction.

$$\mathcal{D} = \{(s_i, a_i, r_i, s_{i+1})\}_{i=1}^{N} \xrightarrow{\text{training}} \pi$$

*Real interaction is too expensive, risky, or slow (pricing, medical, supply chain)*

---

**Our approach — model-based offline RL via a learned simulator:**

$$\text{Historical data} \xrightarrow{\text{fit demand model}} \hat{P}(s' | s, a) \xrightarrow{\text{synthetic rollouts}} \text{Q-learning}$$

We replace the real environment with a *simulator built from data*. This is sometimes called **model-based offline RL** or simply a **learned world model**.

---

## The Simulator Hypothesis

A simulator is only useful if it approximates reality.

**What we assume:**
1. The demand model generalizes well from train to test periods
2. Demand is a deterministic function of price and lagged context (up to noise)
3. The lag update rule in `env_step()` is a reasonable proxy for temporal dynamics

**What we explicitly do not assume:**
- That the model is correct at price ranges far from historical data (extrapolation)
- That the pricing dynamics are stationary over months or years
- That competitor pricing, promotions, or stockouts are irrelevant

<div class="warning">
<b>Risk:</b> Policy trained on a misspecified simulator may exploit model errors rather than real demand structure. This is reward hacking against a proxy.
</div>

---

# Part 2: From Data to Simulator

---

## UCI Online Retail: Dataset Overview

| Field | Description |
|-------|-------------|
| `InvoiceNo` | Transaction ID (prefix "C" = cancellation) |
| `StockCode` | Product code |
| `Quantity` | Units sold (positive for sales) |
| `UnitPrice` | Price per unit (GBP) |
| `InvoiceDate` | Timestamp |
| `CustomerID` | (Sparse — not used) |

**Raw → cleaned:**
- Remove cancellations (`InvoiceNo` starts with "C")
- Remove returns (`Quantity ≤ 0`) and zero/negative prices
- Aggregate to **(StockCode, date)** level: daily qty, mean price, transaction count

**Result:** ~540k rows → ~100k product-day rows across top-200 products

---

## Feature Engineering Pipeline

From the daily product-date table, we derive:

| Feature | Definition | Role |
|---------|------------|------|
| `price` | Mean unit price that day | Base for action scaling |
| `qty_lag1` | Sales yesterday | Demand momentum signal |
| `qty_lag7` | Sales 7 days ago | Weekly seasonality proxy |
| `n_txn` | Distinct invoices that day | Intensity / demand volume |
| `dow` | Day-of-week (0=Mon) | Weekly cycle |
| `month` | Month (1–12) | Seasonal cycle |

**Tabular state:** `qty_bin(qty_lag1)` — 9 bins defined by thresholds [1, 10, 20, 30, 40, 50, 100, 200]

**Vector state (for M4+):** 23-dimensional float vector (4 numerical + 7 DOW one-hot + 12 month one-hot)

---

## State Space Design: Discretization

Why discretize demand into bins rather than use raw quantity?

**Q-table requirement:** State space must be finite and small enough to visit each state-action pair many times during training.

```
qty_lag1 ∈ [0, ∞) → 9 bins:
  bin 0: qty < 1       (near-zero demand)
  bin 1: 1 ≤ qty < 10
  bin 2: 10 ≤ qty < 20
  ...
  bin 8: qty ≥ 200     (high-volume product-days)
```

**Q-table shape:** `(9 bins × 8 actions)` = 72 parameters. Tiny. Every cell is updated thousands of times.

**Tradeoff:** Bins discard within-bin variation. Products with qty=11 and qty=19 look identical to the agent. This information loss is acceptable — and unavoidable — in tabular RL.

---

# Part 3: Demand Model as Simulator

---

## The Log-Linear Demand Model

$$\log(1 + \hat{q}) = \beta_0 + \underbrace{\beta_1}_{\text{elasticity}} \log(p) + \beta_2 \cdot q^{(1)} + \beta_3 \cdot q^{(7)} + \beta_4 \cdot n_{\text{txn}} + \text{DOW} + \text{Month}$$

where $q^{(1)}$ = `qty_lag1`, $q^{(7)}$ = `qty_lag7`, $n_{\text{txn}}$ = transaction count.

**Why log-log for price?** It yields a constant elasticity model:

$$\frac{\partial \log \hat{q}}{\partial \log p} = \beta_1 \quad \Rightarrow \quad \% \Delta q \approx \beta_1 \cdot \% \Delta p$$

A 10% price increase leads to approximately $10 \beta_1$% change in quantity.
With elastic demand, $\beta_1 < -1$.

**Fit:** `sklearn.Pipeline` with `ColumnTransformer` + `LinearRegression`, 80/20 train-test split by rows.

---

## Interpreting the Demand Model

After fitting on the UCI dataset, typical results:

| Metric | Value |
|--------|-------|
| $R^2$ (log-qty, hold-out) | ~0.55–0.65 |
| $\hat{\beta}_1$ (log-price elasticity) | typically −1.0 to −2.0 |

**What $R^2 \approx 0.60$ means:** The model explains ~60% of variance in log(1+qty). The remaining 40% is noise from stockouts, promotions, competitor prices, and random demand shocks.

**Implication for the simulator:**
- `env_step()` uses the point prediction of `predict_qty()` — no noise injection
- This makes the environment *deterministic* given a feature row and action
- Real demand is stochastic; our simulator is a smoothed approximation

---

## env_step(): The Transition Function

```python
def env_step(model, row_dict, action_mult) -> (next_state, reward, next_state):
    chosen_price  = row_dict["price"] * action_mult          # scale base price
    qty_hat       = predict_qty(model, row_dict, chosen_price) # demand prediction
    reward        = chosen_price * qty_hat                    # revenue

    next_state = {
        "price":    chosen_price,
        "qty_lag1": qty_hat,                    # predicted qty becomes new lag1
        "qty_lag7": row_dict["qty_lag1"],       # old lag1 becomes lag7 proxy
        "n_txn":    row_dict["n_txn"],          # carried forward (simplified)
        "dow":      (row_dict["dow"] + 1) % 7,  # advance day
        "month":    row_dict["month"],
    }
    return next_state, reward, next_state
```

**Simplifications baked in:**
- `n_txn` is carried forward unchanged (real n_txn varies with demand)
- `qty_lag7` is approximated from `qty_lag1` — not a true 7-day history
- Price compounding: after k steps, base price has shifted to the policy's chosen price

---

## Simulator Limitations: Know What You're Trusting

| Assumption | Reality | Risk |
|------------|---------|------|
| Demand is linear in log-price | Likely nonlinear for extreme prices | Policy may exploit the linear region |
| Context features are exogenous | Stockouts, promotions affect price | Confounding in the data |
| `n_txn` is stationary | Higher prices → fewer transactions | State drift underestimated |
| Lag update is correct | Simulated qty ≠ real qty | Compounding error over horizon |
| No competitor dynamics | Competitor pricing shifts demand | Model misses cross-price effects |

<div class="callout">
<b>The simulator is not reality. It is a structured approximation. </b>
Evaluation on held-out historical rows (not simulator rollouts) is the sanity check.
</div>

---

# Part 4: Reward Design

---

## Revenue as Reward

The natural reward in pricing is revenue:

$$r_t = p_t \cdot \hat{q}_t = (\text{base\_price} \times a_t) \cdot \text{predict\_qty}(\cdot)$$

**Why revenue is reasonable:**
- Directly measurable from transactions
- Aligns with business KPI at a single product level
- Decomposes cleanly into price × quantity — intuitive for debugging

**Why revenue is not enough:**

| Problem | Description |
|---------|-------------|
| No cost term | Revenue ≠ profit; high-price high-revenue may have high returns |
| Short-horizon proxy | Maximize today's revenue may sacrifice long-term customer retention |
| Proxy mismatch | Our $\hat{q}$ is a model; extreme prices may yield unrealistic predictions |
| No clipping | Unconstrained revenue can blow up with model extrapolation errors |

---

## Reward Hacking Against the Demand Model

Because our reward is computed through a learned model, the agent can find actions that exploit model errors rather than exploit real demand structure.

**Example:** The log-linear model predicts `qty > 0` for any price because `predict_qty` is clipped to [0, ∞). At price=1.25×, if the model underestimates elasticity (i.e., predicts demand doesn't drop much), the agent will always choose 1.25×.

$$\text{Agent learns: } a^* = \arg\max_a \hat{p}(a) \cdot \hat{q}(a)$$
$$\text{But: } \hat{q} \text{ may be optimistic at high prices}$$

<div class="warning">
<b>Reward hacking signal to watch:</b> If the trained policy always selects the highest multiplier (1.25×) for all states, it has likely hacked the demand model, not learned real elasticity.
</div>

**Mitigation options:**
- Add soft regularization: penalize large deviations from 1.00× (status quo)
- Clip predicted quantity: cap `qty_hat` at a percentile of historical distribution
- Use a held-out demand model for evaluation (not the training simulator)

---

## Reward Design Principles

**General principles (applicable beyond pricing):**

1. **Reward should measure what you actually care about**, not a convenient proxy
2. **Proxy objectives are fine if the proxy is well-calibrated and bounded**
3. **Never reward the agent on its own prediction** — always use an independent signal when possible
4. **Decompose reward to detect failure modes:** track price separately from quantity separately from revenue
5. **Clipping and scaling matter:** unscaled reward differences dominate learning signal

**In the pricing context:**

$$r_t = \text{clip}\left(p_t \cdot \hat{q}_t,\ 0,\ R_{\text{max}}\right)$$

Setting $R_{\text{max}}$ at the 99th percentile of historical daily revenue per product prevents outlier products from distorting the Q-table.

---

# Part 5: Q-Learning in the Pricing Environment

---

## Q-Table Setup (Callback to Module 2)

<div class="callback">
<b>M2 recap:</b> Q(s, a) estimates the expected discounted return from taking action a in state s and following policy π thereafter. The Q-learning update (off-policy, bootstrapped):
</div>

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big( r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big)$$

**In the pricing environment:**
- $s_t$ = `qty_bin(qty_lag1)` ∈ {0, 1, ..., 8}
- $a_t$ ∈ {0, ..., 7} mapping to multipliers {0.90, ..., 1.25}
- $r_t$ = `chosen_price × predict_qty(...)`
- $s_{t+1}$ = updated feature dict → `qty_bin(qty_hat)`

**Q-table shape:** `Q[9, 8]` — 72 floats. Every cell is visited thousands of times.

---

## Training Loop

```
For each training episode:
  1. Sample a random row from episode_table (initial state)
  2. Discretize: s = qty_bin(row["qty_lag1"])
  3. For T steps:
       a. ε-greedy: with prob ε pick random a; else a = argmax Q[s]
       b. mult = ACTION_MULTS[a]
       c. next_row, reward, _ = env_step(demand_model, row, mult)
       d. s' = qty_bin(next_row["qty_lag1"])
       e. Q[s, a] += α * (reward + γ * max(Q[s']) - Q[s, a])
       f. row = next_row; s = s'
  4. Decay ε
```

**Key hyperparameters:**
| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| α (learning rate) | 0.1 | Controls how fast Q updates |
| γ (discount) | 0.95 | Weight on future revenue |
| ε start / end | 1.0 → 0.05 | Exploration decay |
| T (horizon) | 20 steps | Multi-step episode length |

---

## Expected SARSA: A Variance-Reduced Alternative

<div class="callback">
<b>M2 recall:</b> Q-learning uses max over next actions (optimistic); Expected SARSA uses the expectation under current policy (less variance, more stable).
</div>

**Expected SARSA update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big( r_t + \gamma \sum_{a'} \pi(a'|s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \Big)$$

where $\pi(a|s) = (1-\varepsilon) \cdot \mathbf{1}[a = \arg\max Q(s,\cdot)] + \frac{\varepsilon}{|A|}$

**In the pricing context:**
- Revenue has high variance across products (small-qty items vs. large-qty items)
- Expected SARSA is less prone to overshooting on high-revenue outliers
- Typically yields smoother learning curves and more stable convergence

**Rule of thumb:** If Q-learning oscillates or diverges early, switch to Expected SARSA first before tuning α.

---

## Double Q-Learning: Reducing Overestimation Bias

<div class="callback">
<b>M2 recall:</b> Q-learning overestimates because: $E[\max_a Q(s,a)] \geq \max_a E[Q(s,a)]$ (max of noisy estimates > max of means).
</div>

**Double Q-learning:** Maintain two independent Q-tables $Q_A$ and $Q_B$. Use one to *select* the action, the other to *evaluate* it:

$$Q_A(s_t, a_t) \leftarrow Q_A(s_t, a_t) + \alpha \Big( r_t + \gamma Q_B\!\left(s_{t+1}, \arg\max_{a'} Q_A(s_{t+1}, a')\right) - Q_A(s_t, a_t) \Big)$$

Alternate updates between $Q_A$ and $Q_B$.

**In the pricing context:**
- Revenue from high-multiplier actions may be inflated by demand model errors
- Double Q dampens the tendency to overcommit to high-price actions
- Particularly relevant when the demand model has high uncertainty at extremes

---

# Part 6: Evaluation Methodology

---

## Why Evaluation Design Is Non-Trivial

After training a Q-policy, we want to answer: **"Does this policy actually outperform not changing prices?"**

The naive approach — "compare mean revenues from RL rollouts vs. baseline rollouts" — has multiple failure modes:

1. **Unpaired comparison:** RL and baseline may be evaluated on different product-days with different intrinsic demand levels. Apparent lift may be selection bias.
2. **High-variance estimate:** Revenue is heavy-tailed (a few product-days with huge volume). A single-trajectory comparison is unreliable.
3. **Wrong null hypothesis:** t-test assumes normality; revenue distribution is right-skewed.

**Correct design:**
- Paired comparison: same row, same initial state, only action differs
- Many independent samples (n ≥ 2000)
- Multiple statistical tests with different assumptions

---

## The Paired Comparison Protocol

For each evaluation episode i:
1. Sample row $i$ from `episode_table`
2. Compute RL revenue: $r^{\text{RL}}_i = \text{RL policy price}_i \times \hat{q}^{\text{RL}}_i$
3. Compute baseline revenue: $r^{\text{base}}_i = \text{base price}_i \times \hat{q}^{\text{base}}_i$ (1.00× multiplier)
4. Record difference: $d_i = r^{\text{RL}}_i - r^{\text{base}}_i$

The random variable of interest is $d_i$, not $r^{\text{RL}}_i$ in isolation.

**Why pairing eliminates confounding:**
- Products with naturally high demand get high revenue *regardless* of policy
- $d_i$ cancels out product-level demand level: only the *differential effect of action* remains
- Variance of $d_i$ is typically much lower than variance of $r^{\text{RL}}_i$ alone

---

## Statistical Tests

**1. Paired t-test** — $H_0$: $E[d_i] = 0$

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} \quad \sim t_{n-1}$$

Assumption: $d_i \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$. Sensitive to heavy tails in revenue.

---

**2. Wilcoxon signed-rank test** — $H_0$: distribution of $d_i$ is symmetric around 0

Ranks the absolute differences $|d_i|$; tests whether positive differences tend to be larger in magnitude than negative ones. **Robust to heavy tails and outliers.** Preferred when revenue distribution is clearly non-normal.

---

**3. Sign test / Binomial test** — $H_0$: $P(d_i > 0) = 0.5$

$$\text{Wins} = \sum_i \mathbf{1}[d_i > 0], \quad \text{Wins} \sim \text{Binomial}(n_{\text{eff}},\ 0.5)$$

Answers "does RL win more often than chance?" Ignores magnitude — very robust but low power.

---

**4. Bootstrap CI** — no distributional assumption

$$\bar{d}^{(b)} = \frac{1}{n} \sum_i d^{(b)}_i, \quad b = 1, \ldots, B$$

Report 95% CI: $[\hat{q}_{0.025}(\bar{d}^{(b)}),\ \hat{q}_{0.975}(\bar{d}^{(b)})]$. If CI excludes 0, lift is statistically significant.

---

## Revenue Lift and Effect Size

**Primary metric:** Revenue lift (%)

$$\text{Lift\%} = 100 \times \frac{\bar{r}^{\text{RL}} - \bar{r}^{\text{base}}}{\bar{r}^{\text{base}}}$$

**Effect size (Cohen's d for paired data):**

$$d = \frac{\bar{d}}{s_d}$$

| $d$ | Interpretation |
|-----|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

Cohen's d is scale-invariant — useful for comparing across experiments with different revenue scales.

**Interpretation rule:** Statistical significance (p < 0.05) + meaningful effect size (d > 0.2) + consistent CI are all needed. A statistically significant result with d = 0.01 is practically irrelevant.

---

## Why Revenue Is Heavy-Tailed

$$r = p \times q$$

Price $p$ has moderate variance. Quantity $q$ is heavy-tailed (some products sell thousands of units on a single day; most sell <20). Their product inherits the tail behavior.

**Consequence for test choice:**
- Paired t-test will underestimate variance → overstate significance
- Wilcoxon + bootstrap CI give more reliable inference
- Always report both t-test and Wilcoxon; if they disagree, Wilcoxon is more trustworthy

**Practical check:** Plot the distribution of $d_i = r^{\text{RL}}_i - r^{\text{base}}_i$. If it has extreme outliers (> 10× median), report the median lift alongside the mean lift.

---

## Evaluation Summary: What to Report

For a complete evaluation result, always report:

```
RL avg revenue:       $X.XX   Baseline avg revenue: $Y.YY
Revenue lift:         +Z.Z%

Paired t-test:        t = ..., p = ...
Wilcoxon signed-rank: W = ..., p = ...
Sign test:            wins=..., n_eff=..., p = ...
Bootstrap 95% CI:     [lo%, hi%]
Cohen's d:            ...

RL avg price:         X.XX×   Baseline avg price: 1.00×
RL avg quantity:      X.XX    Baseline avg quantity: Y.YY
```

**Action distribution sanity check:** If the policy always picks 1.25× for every state, it has likely hacked the demand model — not learned real elasticity. Compare action distributions across demand states.

---

# Part 7: Common Pitfalls

---

## Implementation Mistakes

**1. Leakage via future demand signals**
Using `qty` of the current day (not lagged) as a state feature means the agent "knows" how well the day went before choosing a price. In production, this information doesn't exist at decision time.

✗ State includes `qty` (current day)
✓ State includes only `qty_lag1`, `qty_lag7` (past observations)

**2. Evaluating on the training simulator without held-out data**
If you evaluate your Q-policy by rolling it out in the same demand model you trained on, you are testing in-sample. The policy may exploit model idiosyncrasies.

✓ Evaluate by applying the Q-policy to historical rows (direct predict_qty call), not multi-step rollouts from the training environment.

**3. Forgetting to seed RNG before training**
Q-learning with random initialization is sensitive to initialization noise. Always seed numpy RNG at the top of the training section.

---

## Conceptual Misunderstandings

**4. Treating the simulator as the real environment**
The demand model is a statistical approximation. A policy with +5% lift on the simulator may have 0% lift on real customers if demand model errors are systematic.

**5. Confusing stationary policy with convergence**
Q-values stabilizing during training (loss plateaus) does not mean the policy has converged to optimal. The Q-table may have stabilized at a suboptimal fixed point, especially with ε-greedy exploration that has decayed too early.

**6. Ignoring the discount factor's role**
With γ = 0.95 and horizon T = 20, future rewards are discounted by $0.95^{20} \approx 0.36$. The agent effectively looks ~7 steps ahead. Setting γ too low makes the agent myopically maximize immediate revenue at the expense of demand health.

**7. Assuming tabular Q-learning generalizes across products**
A Q-table trained on Product A has no information about Product B. Each product has its own demand curve. The shared tabular policy is only sensible because we discretize by demand bin — products in the same demand bin are treated as similar.

---

## Failure Modes in the Pricing Context

**8. Always-high-price policy**
Symptom: `argmax Q[s]` = 7 (1.25× multiplier) for all states.
Cause: Demand model underestimates elasticity; revenue reward never penalizes demand destruction enough.
Fix: Check `price_elasticity(model)`. If |β₁| < 0.5, the model is insensitive to price — the agent will always prefer high price.

**9. Evaluation confidence without variance**
Symptom: Reporting "RL achieved +8% lift" from a single 100-episode rollout.
Fix: Use n_eval ≥ 2000 paired episodes. Report 95% CI. Single-trajectory estimates have high variance for heavy-tailed revenue.

**10. Reward scaling mismatch across products**
Products with high unit price × high volume generate 100× more reward than low-price low-volume products. If you train a single Q-table across all products, updates are dominated by high-revenue products.
Fix: Normalize reward within product by its historical mean revenue, or train per-product Q-tables.

---

# Part 8: From Notebook to System

---

## What's Production-Ready (From This Module)

| Component | Status | Notes |
|-----------|--------|-------|
| Demand model (sklearn Pipeline) | ✅ Production-viable | Needs retraining cadence, drift monitoring |
| Evaluation framework (paired t-test, bootstrap CI) | ✅ Production-viable | Already handles heavy tails correctly |
| Action space (price multipliers) | ✅ Production-viable | May need business rule constraints |
| Episode table construction | ✅ Production-viable | Needs live data feed to replace batch pull |

## What's Experimental

| Component | Status | Limitation |
|-----------|--------|------------|
| Q-table policy | ⚠️ Experimental | Doesn't generalize to new products or states |
| Demand model as transition function | ⚠️ Experimental | Systematic errors compound over rollout horizon |
| Single-step tabular state | ⚠️ Experimental | Loses temporal and cross-product information |

---

## Pipeline: Notebook → Decision System

```
notebooks/pricing_tabular.ipynb
         ↓
  shared/demand_model/demand_model.py    (retrainable pipeline)
  shared/envs/pricing_env.py             (environment + action space)
  shared/utils/evaluation.py             (statistical evaluation)
         ↓
  Batch inference: for each product-day, query Q-policy
         ↓
  A/B test: route X% of traffic to RL-suggested prices
         ↓
  Online monitoring: track revenue, price elasticity, return rates
         ↓
  Periodic retraining: demand model drift, policy update
```

**What would need to change for deployment:**
1. Live data pipeline replaces `load_data()` batch pull
2. Demand model served as a microservice with versioning
3. Q-policy replaced by a generalizable policy (Module 4: DQN)
4. Evaluation uses real revenue, not simulated revenue

---

# Part 9: Bridge to Module 4

---

## The Limits of Tabular RL

| Limitation | Consequence |
|------------|-------------|
| State space is manually discretized | Information loss; binning is arbitrary |
| Q-table doesn't generalize across bins | Near-identical demand levels in adjacent bins treated independently |
| Can't incorporate continuous features | DOW, month, n_txn all dropped in tabular state |
| Q-table size grows as `n_states × n_actions` | In pricing, adding product ID explodes the table |

**The fundamental problem:** Hand-designed discretization is a lossy bottleneck.

Instead of:
$$Q: \{0,\ldots,8\} \times \{0,\ldots,7\} \to \mathbb{R}$$

We want:
$$Q_\theta: \mathbb{R}^{23} \times \{0,\ldots,7\} \to \mathbb{R}$$

A neural network $Q_\theta$ can take the full 23-dimensional state vector and approximate the action-value function without explicit discretization.

---

## Module 4 Preview: Deep Q-Network (DQN)

**Same environment, same action space, same evaluation framework.**

What changes:
- Q-table → neural network $Q_\theta(s)$ outputting a Q-value per action
- State → 23-dim float vector (`state_to_vec()` already in `shared/envs/`)
- Update rule: backpropagation instead of tabular assignment
- Two new components needed for stable training:
  - **Experience replay buffer:** breaks temporal correlations in training samples
  - **Target network:** stabilizes the bootstrap target $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$

<div class="callback">
<b>M2 callback:</b> Double Q-learning (tabular, M2) is the conceptual predecessor of Double DQN (M4). The decoupled selection/evaluation trick applies identically — just with neural networks instead of tables.
</div>

---

## Summary: Module 3 Key Takeaways

1. **Offline RL via a learned simulator** lets us train policies without live experimentation, at the cost of simulator fidelity assumptions.

2. **Demand model as environment:** log-linear, log-log price-quantity relationship, interpretable elasticity coefficient $\hat{\beta}_1$.

3. **State discretization** makes the Q-table tractable but discards within-bin information. This is the primary bottleneck motivating DQN.

4. **Reward design is the hardest part.** Revenue is a reasonable proxy; reward hacking against the demand model is a real failure mode to monitor.

5. **Evaluation requires paired testing, multiple tests, and variance accounting.** Revenue is heavy-tailed: Wilcoxon + bootstrap CI are non-negotiable.

6. **The shared/ modules** (demand model, pricing env, evaluation utils) form the stable interface. Module 4–6 reuse them without modification.

---

## Module 3 Checklist

Before moving to Module 4, confirm:

- [ ] Can explain why we use a demand model simulator instead of live interaction
- [ ] Can interpret the price elasticity coefficient from the fitted model
- [ ] Understand the state discretization tradeoff: fewer states → faster learning but less information
- [ ] Can identify the 3 main reward design failure modes in this environment
- [ ] Can set up and interpret all 4 statistical tests in `evaluate_policy()`
- [ ] Can distinguish between *statistical significance* and *practical effect size*
- [ ] Can identify in the Q-table when reward hacking has occurred (always-max-price behavior)
- [ ] Know which components of this pipeline would change for production deployment

---

*Module 3 — Applied Tabular RL*
*Next: Module 4 — Deep RL (DQN): same environment, neural Q-function*
