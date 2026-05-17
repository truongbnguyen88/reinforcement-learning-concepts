# Module 3 — Additional Notes

## Why Reward Design and Demand Model Are Problematic

### The Structural Problem

The reward is a product of the action and a model prediction:

$$\hat{r}(s, a) = p \cdot \hat{q}(p, s;\, \theta)$$

The agent optimizes against $\hat{q}$, not true demand $q$. Model errors become exploitable gradients.

### Specific Failure: Elasticity Underestimation

If $|\hat{\beta}_1| \ll |\beta_1^\text{true}|$ (price elasticity underestimated due to low price variation in training data):

- Model predicts demand barely responds to price increases
- Agent concludes: raise price for every state → $\hat{r}$ always increases
- Reality: high prices destroy demand — but the simulator hides this

**Diagnostic:** trained policy selects the same extreme action across all states regardless of context → reward hacking, not learned structure.

---

## Three Core Implications

**1. Optimization pressure finds model errors**

Better model fit does not imply safer policy training. The RL agent actively searches for where $\hat{r}$ is maximized — exactly where extrapolation errors are most exploitable. The agent finds model blind spots faster than any diagnostic you would run.

**2. Cannot evaluate a policy on the same simulator used to train it**

In-distribution simulator evaluation measures how well the policy exploits the simulator, not real-world performance. These can be completely uncorrelated. This forces a real evaluation pipeline: holdout real outcomes, A/B testing, or offline evaluation methods (importance sampling, doubly robust estimators) that account for distributional shift.

**3. Behavior policy constrains what can be safely learned**

The simulator is only reliable inside the support of $\mathcal{D}$. The behavior policy $\pi_b$ defines that support. Any learned policy deviating significantly from $\pi_b$'s action distribution operates in the model's extrapolation region. Aggressive policy improvement is self-defeating: the more the learned policy differs from $\pi_b$, the less trustworthy its simulator-based evaluation becomes.

---

## Should We Just Use Model-Free RL (e.g., DQN)?

**No — the choice is not model-based vs. model-free. It is online vs. offline.**

DQN is an online algorithm requiring live environment interaction. In the offline setting (fixed historical dataset), DQN has the same distributional shift problem: training on $\pi_b$-generated transitions while the learned policy deviates from $\pi_b$. Naive offline DQN can be worse than model-based offline RL because there is no explicit model to inspect or constrain.

### The Correct Dichotomy

| Setting | Viable Approaches |
|---------|-------------------|
| Online (can interact) | DQN, PPO, SAC — model-free works well |
| Offline (fixed dataset) | Model-based offline RL, CQL, TD3+BC, IQL |
| Offline → online | Pre-train offline, fine-tune with limited real interactions |

### What Offline Model-Free Methods Actually Do

Algorithms like CQL and IQL add a **pessimism penalty**: explicitly penalize Q-values for out-of-distribution actions, preventing exploitation of regions not covered by $\mathcal{D}$. This is the model-free analogue of constraining rollouts to the simulator's reliable region. The underlying principle is identical — do not trust value estimates for actions the behavior policy never took.

### Bottom Line

> The simulator is a hypothesis about the world. The policy optimizes against that hypothesis. Deployment tests whether the hypothesis was correct.

The M3 pipeline (demand model + tabular Q) illustrates the structural risks — those risks exist in any offline RL method, model-free or not.
