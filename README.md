# Reinforcement Learning Concepts

A structured, self-contained course in reinforcement learning — from mathematical foundations through deep RL and production deployment. Every concept is taught with formal derivations, and every algorithm is implemented in a minimal toy environment that can be run without any external data or downloads.

---

## Course Philosophy

**Slides** — Each module has a slide deck covering the theory in full generality: formal definitions, derivations, assumptions, failure modes. No domain-specific examples; the concepts stand alone.

**Notebooks** — Each module has one or two Jupyter notebooks that implement the concepts on a small synthetic environment. The environment is simple enough that the true optimal policy can be computed analytically, so the notebook can verify that the algorithm learned the right thing.

This separation is intentional: understanding *why* an algorithm works should not require understanding a particular dataset. The toy environments are designed to make the algorithm's behavior transparent, not to simulate a realistic application.

---

## Prerequisites

| Area | What is assumed |
|------|----------------|
| Python | Comfortable with numpy, matplotlib, sklearn |
| Math | Linear algebra, probability, basic calculus |
| ML | Supervised learning, regression, train/test split |
| RL | None — the course starts from scratch |

---

## Module Map

| Module | Topic | Slides | Notebooks | Status |
|--------|-------|--------|-----------|--------|
| 01 | RL Foundations | `Slide1_RL_Foundations.pdf` | gridworld exercises | slides done |
| 02 | Tabular RL | `Slide2_TabularRL.pdf` | FrozenLake Q-learning | slides done |
| 03 | Applied Tabular RL | `Slide3_AppliedTabularRL.md` | offline RL simulator, reward design & evaluation | **complete** |
| 04 | Deep Q-Networks | — | — | planned |
| 05 | Policy Gradients | — | — | planned |
| 06 | RL in Production | — | — | planned |

---

## Repository Structure

```
modules/
├── 01_foundations/
│   ├── slides/
│   │   └── Slide1_RL_Foundations.pdf       ← MDP, Bellman equations, value functions
│   ├── notebooks/
│   └── README.md
│
├── 02_tabular_rl/
│   ├── slides/
│   │   └── Slide2_TabularRL.pdf            ← Q-learning, Double Q, Expected SARSA, n-step
│   ├── notebooks/
│   │   └── frozenlake_q_learning.py
│   └── README.md
│
├── 03_applied_tabular/
│   ├── slides/
│   │   ├── Slide3_AppliedTabularRL.md      ← offline RL, simulators, reward design, evaluation
|   |   └── Slide3_concepts.pdf             ← pdf presentation of Slide 3 materials (by NotebookLM)
│   ├── notebooks/
│   │   ├── 01_offline_rl_simulator.ipynb   ← dataset → demand model → simulator → Q-learning
│   │   └── 02_reward_design_evaluation.ipynb  ← proxy rewards, reward hacking, paired tests
│   └── README.md
│
├── 04_deep_rl/                             ← planned: replay buffer, target network, DQN
├── 05_policy_gradients/                    ← planned: REINFORCE, advantage, A2C, PPO
└── 06_rl_in_production/                    ← planned: OPE, reward hacking taxonomy, A/B testing

archive/                                    ← original notebooks, preserved as-is
├── tabuluar_Q_learning.ipynb
├── deep_Q_learning.ipynb
├── figures/
└── RL_concepts/

shared/                                     ← utilities extracted from the original project
├── demand_model/demand_model.py
├── envs/pricing_env.py
└── utils/evaluation.py

requirements.txt
```

---

## Module Descriptions

### Module 1 — RL Foundations
**Slide deck:** MDP formalism (states, actions, transitions, rewards, discount), Bellman expectation and optimality equations, policy evaluation, value iteration, policy iteration.

**Key questions answered:**
- What is the mathematical definition of a sequential decision problem?
- What does "optimal" mean and how do we compute it?
- Why does the Bellman equation make dynamic programming tractable?

---

### Module 2 — Tabular RL
**Slide deck:** Model-free prediction and control, Q-learning derivation, overestimation bias and Double Q-learning, Expected SARSA, n-step returns, eligibility traces.

**Notebook:** Q-learning, Double Q-learning, and Expected SARSA on FrozenLake. Learning curves and policy comparison.

**Key questions answered:**
- How does Q-learning converge without a model of the environment?
- Why does `max` introduce bias, and how does Double Q fix it?
- When does Expected SARSA outperform Q-learning?

---

### Module 3 — Applied Tabular RL
**Slide deck:** Offline RL framing, model-based offline RL via a learned simulator, log-linear demand model and price elasticity, reward design and proxy objectives, reward hacking, potential-based reward shaping, paired policy evaluation, statistical tests for heavy-tailed returns.

**Notebooks:**

`01_offline_rl_simulator.ipynb`
- Defines a 5-state toy market with state-dependent price elasticity (true parameters hidden)
- Generates a synthetic offline dataset from a random behavior policy
- Fits a per-state log-linear demand model from the logged data
- Builds a simulator from the fitted model and trains Q-learning on it
- Measures compounding rollout error via total variation distance between true and simulator state distributions

`02_reward_design_evaluation.ipynb`
- Trains two Q-policies on the same dynamics with different reward functions (revenue vs. profit) and shows the learned policies diverge
- Demonstrates reward hacking: a miscalibrated demand model causes the agent to always pick the highest price, which underperforms baseline on the true environment
- Implements the full four-test evaluation protocol: paired t-test, Wilcoxon signed-rank, sign/binomial test, bootstrap CI, and Cohen's d

**Key questions answered:**
- What does it mean to do RL "offline" and what are the risks?
- How does model error compound over multi-step rollouts?
- Why does choosing the wrong reward function matter more than choosing the wrong algorithm?
- When is a policy improvement statistically credible?

---

### Module 4 — Deep Q-Networks *(planned)*
**Slide deck:** Neural Q-function approximation, the instability problem in naive DQN, experience replay buffer (breaks temporal correlation), target network (stabilizes bootstrap target), Double DQN (decouples selection and evaluation), dueling network architecture.

**Notebook:** DQN on a simple continuous-state environment. Replay buffer and target network implemented from scratch. Comparison of DQN vs. tabular Q-learning from Module 3 on a shared benchmark.

---

### Module 5 — Policy Gradients *(planned)*
**Slide deck:** Policy gradient theorem derivation, REINFORCE, high variance of Monte Carlo policy gradients, variance reduction via baselines, advantage estimation, actor-critic (A2C), PPO clipped surrogate objective.

**Notebook:** REINFORCE and A2C on a simple continuous-action environment. Policy network and value baseline implemented from scratch.

---

### Module 6 — RL in Production *(planned)*
**Slide deck:** Offline policy evaluation (importance sampling, doubly-robust estimator), reward hacking taxonomy, contextual bandits as a limiting case of RL, A/B testing and causal inference for policy comparison, distribution shift detection and monitoring.

**Notebook:** IS and doubly-robust OPE on a synthetic logged dataset. Comparison of OPE estimates against true policy value.

---

## Archive

The `archive/` directory contains the original notebooks that this course grew out of:

- `tabuluar_Q_learning.ipynb` — Q-learning, Double Q, Expected SARSA, n-step returns applied to a UCI retail pricing problem
- `deep_Q_learning.ipynb` — DQN, Double DQN, dueling network, experience replay on the same pricing environment

These are preserved as-is. The `shared/` directory contains utilities extracted from those notebooks (demand model, pricing environment, evaluation functions) for reference.

---

## Getting Started

```bash
git clone <repo>
cd reinforcement-learning-concepts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open the modules in order, starting with `modules/01_foundations/`. Each notebook is self-contained and runs without downloading any data.

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
torch
jupyter
```

See `requirements.txt` for pinned versions.
