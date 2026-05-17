# Tasks

## Active

- [x] ~~Track B — Slide3 (Module 3 content)~~ (2026-05-17) — offline RL framing, reward design, demand model as simulator, evaluation methodology
  - `modules/03_applied_tabular/slides/Slide3_AppliedTabularRL.md` (Marp markdown → PDF)
- [ ] **Track C — pricing_pg.ipynb (Module 5)** - REINFORCE/A2C with continuous price action space; depends on shared/envs

## Waiting On

## Someday

- [ ] **Slide4** - DQN theory deck (replay buffer, target network, Double DQN, dueling)
- [ ] **Slide5** - Policy gradient theory deck (REINFORCE derivation, actor-critic, PPO)
- [ ] **Slide6** - RL in production deck (OPE, reward hacking, A/B testing, drift)
- [ ] **offline_evaluation.ipynb (Module 6)** - IS/doubly-robust OPE on pricing policies
- [ ] **pricing_tabular.ipynb (Module 3)** - refactored notebook using shared/ imports
- [ ] **pricing_dqn.ipynb (Module 4)** - refactored notebook using shared/ imports
- [ ] **Top-level README rewrite** - course overview with module map

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
