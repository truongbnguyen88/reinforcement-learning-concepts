# Tasks

## Active

## Waiting On

## Next Up (proposed order)

- [ ] **Slide4** — DQN theory deck (replay buffer, target network, Double DQN, dueling); general
- [ ] **M4 notebooks** — toy DQN notebook (CartPole or simple continuous-state env; self-contained)
- [ ] **Slide5** — Policy gradient theory deck (REINFORCE derivation, actor-critic, PPO); general
- [ ] **M5 notebooks** — toy policy gradient notebook (REINFORCE/A2C; self-contained)

## Someday

- [ ] **Slide6** — RL in production deck (OPE, reward hacking, A/B testing, drift); general
- [ ] **M6 notebooks** — toy OPE notebook (IS, doubly-robust; self-contained)
- [ ] **Top-level README rewrite** — course overview with module map

## Done

- [x] ~~Design RL course curriculum~~ (2026-05-17) — 6-module plan with running pricing example
- [x] ~~Track A — Repo restructure~~ (2026-05-17)
  - archive/ created with all original files preserved
  - modules/01–06 directory skeleton with READMEs and bridge text
  - shared/demand_model/demand_model.py extracted
  - shared/envs/pricing_env.py extracted (tabular + DQN state interfaces)
  - shared/utils/evaluation.py extracted (paired tests, bootstrap CI, multi-policy comparison)
  - requirements.txt added
  - All imports verified with smoke tests
- [x] ~~Track B — Slide3 (Module 3 content)~~ (2026-05-17)
  - `modules/03_applied_tabular/slides/Slide3_AppliedTabularRL.md` — general theory (no UCI); covers offline RL framing, demand model as simulator, reward design, evaluation methodology
  - Rewrote from UCI-tied version to domain-agnostic after design change
  - Updated CLAUDE.md: slides = general theory, notebooks = UCI applied
  - Updated M4–M6 READMEs to reflect slides/notebooks split
- [x] ~~Module 3 learning notebooks~~ (2026-05-17)
  - `modules/03_applied_tabular/notebooks/01_offline_rl_simulator.ipynb` — toy market MDP; offline dataset → demand model → simulator → Q-learning → compounding error demo; smoke-tested
  - `modules/03_applied_tabular/notebooks/02_reward_design_evaluation.ipynb` — revenue vs profit reward comparison, reward hacking demo (miscalibrated elasticity), full paired evaluation (t-test, Wilcoxon, sign test, bootstrap CI, Cohen's d); smoke-tested
