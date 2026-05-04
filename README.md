# RL Pricing

This repository explores **reinforcement learning for dynamic pricing** using the **UCI Online Retail** dataset. The project starts from historical transactional data, fits a simple offline demand model, builds a simulated pricing environment, and then compares:

- a baseline policy that keeps the observed price,
- a **tabular Q-learning / Double Q-learning** policy,
- a **Deep Q-Network (DQN)** pricing policy.

The work is organized primarily as two Jupyter notebooks:

- [tabuluar_Q_learning.ipynb]
- [deep_Q_learning.ipynb]

Note: the filename `tabuluar_Q_learning.ipynb` contains a typo in the repository and is preserved as-is.

## Repository Contents

```text
RL-pricing/
├── deep_Q_learning.ipynb
├── tabuluar_Q_learning.ipynb
├── figures/
│   └── baseline_vs_doubleQ_vs_DQN.png
└── RL_concepts/
    ├── Slide1_RL_Foundations.pdf
    └── Slide2_TabularRL.pdf
```

## Learning Materials

The `RL_concepts/` directory contains supporting lecture material:

- [RL_concepts/Slide1_RL_Foundations.pdf](RL_concepts/Slide1_RL_Foundations.pdf)
- [RL_concepts/Slide2_TabularRL.pdf](RL_concepts/Slide2_TabularRL.pdf)

These slides complement the notebooks by covering RL foundations and tabular RL concepts used in the implementation.

A simple Tabular Q-learning implementation for FrozenLake problem is given in [frozenlake_q_learning.py]. See the file for details.

## What The Project Does

The notebooks implement the following pipeline:

1. **Download and clean retail data**
   - Pulls the UCI Online Retail dataset directly from the UCI repository.
   - Removes cancellations and non-positive quantity/price rows.
   - Aggregates transactions to a **daily product-level table**.

2. **Engineer pricing and demand features**
   - Builds features such as:
     - `price`
     - `qty`
     - `n_txn`
     - `dow` (day of week)
     - `month`
     - `qty_lag1`
     - `qty_lag7`

3. **Fit an offline demand model**
   - Uses a regression model for:
     - `log(1 + qty) = f(log(price), qty_lag1, qty_lag7, n_txn, dow, month)`
   - Implemented with a `scikit-learn` pipeline using:
     - `ColumnTransformer`
     - `OneHotEncoder`
     - `LinearRegression`

4. **Construct an RL environment**
   - The action is a **price multiplier** applied to the observed base price.
   - Available actions are:
     - `0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25`
   - Reward is simulated **revenue**:
     - `reward = chosen_price * predicted_quantity`
   - State transitions are generated from the demand model by updating lagged-demand features.

5. **Train pricing policies**
   - Tabular notebook:
     - standard Q-learning variants
     - Double Q-learning
     - Expected SARSA
     - n-step updates
   - Deep notebook:
     - DQN
     - Double DQN target logic
     - dueling network architecture
     - replay buffer
     - target network updates

6. **Evaluate against a baseline**
   - Baseline policy keeps the original observed price.
   - Evaluation compares RL revenue against baseline revenue on sampled episodes.
   - The tabular notebook uses paired statistical tests and bootstrap confidence intervals.

## Data and State Design

### Dataset

The notebooks download the **Online Retail** dataset from the UCI repository. A saved notebook output shows the raw dataset shape as:

- `541,909` rows
- `8` columns

The original columns are:

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

## RL Assumptions And Limitations

This project is best understood as a **model-based offline RL prototype**, not a production pricing engine.

Important assumptions:

- the environment is simulated from a supervised demand model rather than from live interaction data
- rewards depend on predicted demand, so policy quality is tied directly to demand model quality
- the state transition is simplified by updating lag features heuristically
- the baseline is a simple no-change pricing policy
- the notebooks focus on top products and short-horizon episodes rather than full operational constraints

Practical implication:

- if the demand model is biased, the RL policy can inherit and amplify that bias

## Requirements

The notebooks import or install the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `requests`
- `openpyxl`
- `scikit-learn`
- `scipy`
- `torch`
- `jupyter`
- `ucimlrepo` (installed in the tabular notebook setup cell)

## How To Run

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib requests openpyxl scikit-learn scipy torch jupyter ucimlrepo
```

Start Jupyter:

```bash
jupyter notebook
```

Suggested execution order:

1. Run `tabuluar_Q_learning.ipynb` first to understand the full preprocessing, demand modeling, and tabular RL setup.
2. Run `deep_Q_learning.ipynb` next to train and compare the DQN agent.
3. Inspect `figures/` and the generated notebook plots for comparisons across methods.
