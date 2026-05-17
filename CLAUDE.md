# CLAUDE.md — RL Course Project

## Me
RL practitioner / MSCS student at U of L. Mathematically advanced.
Building a structured RL course grounded in applied dynamic pricing (UCI Online Retail).

---

## Repo Structure

```
archive/          ← original notebooks and slides, preserved
modules/01–06/    ← one directory per course module
  slides/         ← PDF slide decks
  notebooks/      ← Jupyter notebooks
  exercises/      ← (M1, M2 only)
  README.md       ← module overview + bridge text
shared/
  demand_model/   ← load_data, preprocess, fit_demand_model
  envs/           ← pricing_env: ACTION_MULTS, qty_bin, state_to_vec, env_step
  utils/          ← evaluation: evaluate_policy, compare_policies, bootstrap_ci
requirements.txt
```

## Terms

| Term | Meaning |
|------|---------|
| Tabular notebook | archive/tabuluar_Q_learning.ipynb — Q-learning, Double Q, Expected SARSA, n-step |
| Deep notebook | archive/deep_Q_learning.ipynb — DQN, Double DQN, dueling network, replay buffer |
| Slide1 | modules/01_foundations/slides/Slide1_RL_Foundations.pdf |
| Slide2 | modules/02_tabular_rl/slides/Slide2_TabularRL.pdf |
| FrozenLake script | modules/02_tabular_rl/notebooks/frozenlake_q_learning.py |
| pricing env | shared/envs/pricing_env.py — the single shared RL environment |
| shared modules | shared/ — imported by all module notebooks; never duplicate this logic |

## Course Module Map

| Module | Topic | Key new content needed |
|--------|-------|----------------------|
| 01 | RL Foundations | gridworld exercise notebook |
| 02 | Tabular RL | Double Q / Expected SARSA slide extension |
| 03 | Applied Tabular RL | Slide3 + refactored pricing_tabular.ipynb |
| 04 | Deep RL (DQN) | Slide4 + refactored pricing_dqn.ipynb |
| 05 | Policy Gradients | Slide5 + pricing_pg.ipynb (new) |
| 06 | RL in Production | Slide6 + offline_evaluation.ipynb (new) |

---

## Content Generation Rules

### Audience
- Primary: working data scientists (familiar with sklearn, pandas, ML pipelines)
- Secondary: STEM grad/undergrad students (linear algebra, probability, some ML)
- Assume NO prior RL knowledge; assume strong Python and stats background

### Per-Module Content Pattern
Every module must follow this sequence:
1. **Theory** — formal definitions, key equations, intuition
2. **Toy example** — small clean environment (FrozenLake, gridworld, 2-state MDP)
3. **Pricing application** — same concept applied to the UCI retail pricing env

Never skip the toy example. It is the bridge between theory and the applied domain.

### Explanation Depth
- Define all mathematical notation on first use
- Derive key update rules step by step (don't skip steps for the target audience)
- State assumptions explicitly; flag when they are violated in the pricing context
- Connect each new concept back to a prior module's concept (explicit callbacks)

### Cross-Module Continuity
- Maintain a single running example: pricing a retail product
- Explicitly state which module introduced each concept when referencing it
- Double Q-learning in M4 (DQN) must callback to M2 (tabular Double Q)
- Evaluation methodology introduced in M3 carries forward to M4, M5, M6

---

## Code Generation Rules

### Always import from shared/
All notebooks must import environment, demand model, and evaluation utilities
from `shared/` — never redefine these functions inline:

```python
import sys; sys.path.insert(0, str(Path(__file__).parents[2]))
from shared.demand_model import load_data, preprocess, fit_demand_model
from shared.envs import ACTION_MULTS, qty_bin, state_to_vec, env_step
from shared.utils import evaluate_policy, compare_policies
```

### Notebook structure
Each notebook must have these sections in order:
1. Setup (imports, sys.path, device)
2. Data loading (call shared functions — no inline data code)
3. Concept introduction (markdown with equations)
4. Implementation (clean, typed, commented only where non-obvious)
5. Training / experiment
6. Evaluation (use shared evaluate_policy / compare_policies)
7. Visualization
8. Summary + bridge to next module

### Code style
- Python 3.11+, explicit type hints on function signatures
- No inline redefinition of anything already in `shared/`
- No giant monolithic cells — one logical unit per cell
- Seed all RNG (numpy and torch) at the top of the training section
- Use `pathlib.Path` for all file paths

---

## Cost Efficiency vs. Quality Rules

### When to be concise (save tokens)
- Boilerplate imports, setup cells → minimal comments, no explanation
- Utility functions already documented in `shared/` → import + one-line usage comment only
- Visualizations → standard matplotlib; no elaborate customization unless it aids understanding
- Repeated patterns (e.g., the training loop structure is the same across M3–M5) → reuse, don't rewrite

### When to invest depth (spend tokens)
- First introduction of a concept (e.g., Bellman equation in M1, replay buffer in M4) → full derivation
- Reward design and its failure modes → always thorough; this is the hardest concept for the audience
- Evaluation methodology → full explanation once (M3), then reference back
- Any place where the pricing context diverges from the toy example → explicit callout

### Default behavior
- For **new notebooks**: plan the section structure first, confirm before writing full cells
- For **slide content**: outline bullet points first, confirm depth before expanding
- For **incremental edits**: make targeted changes only; do not rewrite surrounding cells
- For **shared/ modules**: propose interface changes before implementing; downstream notebooks depend on them

---

## Preferences
- Concise, technically deep responses
- Plan before implementing (especially for new notebooks or slides)
- Modular, reproducible code
- No motivational language or filler
- Flag reward hacking risks and leakage risks explicitly when relevant

## Learning Objectives

Each module must explicitly define:

- what the learner should understand conceptually
- what they should be able to implement
- what mistakes they should be able to recognize

Example:
- Understand Bellman expectation equation
- Implement tabular Q-learning from scratch
- Identify overestimation bias in Q-learning

## Common Pitfalls

Each module must explicitly highlight:

- typical implementation mistakes
- conceptual misunderstandings
- failure modes in pricing context

Examples:
- leakage via future demand signals
- unstable Q updates due to large rewards
- misleading convergence in stochastic environments

## Evaluation Philosophy

Evaluation must:

- compare against baseline policies (no-change, heuristic, greedy)
- report variance across runs (not single trajectory)
- use confidence intervals where possible
- separate training vs evaluation environments

In pricing context:
- explicitly compare against historical pricing
- quantify tradeoff: revenue vs churn proxy

## Shared Module Stability

- Treat shared/ as stable interfaces
- Do not modify without justification
- Propose changes before implementing
- Consider downstream impact on all modules

Prefer extending over breaking changes

## From Notebook to System

Whenever possible, highlight:

- what parts are production-ready
- what parts are experimental
- how this would scale to real systems
- what would need to change for deployment

Explicitly bridge:
notebook → pipeline → decision system