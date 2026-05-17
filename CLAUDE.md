# CLAUDE.md — RL Course Project

## Me
RL practitioner / MSCS student at U of L. Mathematically advanced.
Building a structured RL course. Slides teach concepts generally. Notebooks use self-contained toy examples — no external datasets.

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
| shared modules | shared/ — preserved from original repo; not used in new notebooks |

## Course Module Map

| Module | Topic | Content |
|--------|-------|---------|
| 01 | RL Foundations | Slide1 (done); gridworld exercise notebook |
| 02 | Tabular RL | Slide2 (done); FrozenLake notebook |
| 03 | Applied Tabular RL | Slide3 (done); 2 toy notebooks (done) |
| 04 | Deep RL (DQN) | Slide4; toy DQN notebook |
| 05 | Policy Gradients | Slide5; toy policy gradient notebook |
| 06 | RL in Production | Slide6; toy OPE notebook |

---

## Content Generation Rules

### Audience
- Primary: working data scientists (familiar with sklearn, pandas, ML pipelines)
- Secondary: STEM grad/undergrad students (linear algebra, probability, some ML)
- Assume NO prior RL knowledge; assume strong Python and stats background

### Per-Module Content Pattern

**Slides (general — no domain-specific examples):**
1. **Theory** — formal definitions, key equations, intuition
2. **Algorithm or framework** — derivation, pseudocode, assumptions
3. **Failure modes** — what breaks and why
4. **Bridge** — connection to next module

**Notebooks (self-contained toy examples — no external datasets):**
1. One or two notebooks per module using a minimal synthetic environment
2. All data generated inline; no downloads, no `shared/` imports required
3. Environment should be small enough that the true optimum is analytically known

Slides must not reference any specific dataset or domain.
Notebooks must be fully self-contained and runnable without any external data.

### Explanation Depth
- Define all mathematical notation on first use
- Derive key update rules step by step (don't skip steps for the target audience)
- State assumptions explicitly; flag when they can be violated in applied contexts
- Connect each new concept back to a prior module's concept (explicit callbacks)

### Cross-Module Continuity
- Slides: conceptual callbacks only (e.g., "M2 introduced Q-learning; we now extend it")
- Notebooks: each module uses a fresh toy environment suited to the concept being taught
- Explicitly state which module introduced each concept when referencing it
- Double Q-learning in M4 (DQN) must callback to M2 (tabular Double Q)
- Evaluation methodology introduced in M3 carries forward to M4, M5, M6

---

## Code Generation Rules

### Notebook structure
Each notebook must have these sections in order:
1. Setup (stdlib + numpy/torch/sklearn imports only — no shared/ or data downloads)
2. Environment definition (all constants and transition functions defined inline)
3. Concept introduction (markdown with equations)
4. Implementation (clean, typed, no comments except for non-obvious invariants)
5. Training / experiment
6. Evaluation (inline; paired comparison where applicable)
7. Visualization
8. Summary + bridge to next module

### Code style
- Python 3.11+, explicit type hints on function signatures
- No external data dependencies — all environments are synthetic
- No giant monolithic cells — one logical unit per cell
- Seed all RNG (numpy and torch) explicitly; pass seeds as function parameters
- Each notebook must be runnable top-to-bottom with `jupyter nbconvert --to notebook --execute`

---

## Cost Efficiency vs. Quality Rules

### When to be concise (save tokens)
- Boilerplate imports and environment setup cells → minimal comments, no explanation
- Visualizations → standard matplotlib; no elaborate customization unless it aids understanding
- Repeated patterns (e.g., training loop structure is the same across M3–M5) → reuse, don't rewrite

### When to invest depth (spend tokens)
- First introduction of a concept (e.g., Bellman equation in M1, replay buffer in M4) → full derivation
- Reward design and its failure modes → always thorough; this is the hardest concept for the audience
- Evaluation methodology → full explanation once (M3), then reference back
- Any place where an assumption breaks or a subtlety is non-obvious → explicit callout

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
- general failure modes (slides); pricing-specific failure modes (notebooks)

Examples:
- leakage via future information in state
- unstable Q updates due to unscaled rewards
- misleading convergence in stochastic environments

## Evaluation Philosophy

Evaluation must:

- compare against baseline policies (no-change, heuristic, greedy)
- report variance across runs (not single trajectory)
- use confidence intervals where possible
- separate training vs evaluation environments

In notebooks (pricing context):
- explicitly compare against historical pricing
- quantify tradeoff: revenue vs proxy metrics

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