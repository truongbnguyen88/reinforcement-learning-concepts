"""Tabular Q-learning for the classic FrozenLake example.

This version is intentionally dependency-light: it only needs NumPy, so it can
run even when Gymnasium is not installed. The environment follows the standard
4x4 FrozenLake map:

    S F F F
    F H F H
    F F F H
    H F F G

Actions are 0=left, 1=down, 2=right, 3=up. In slippery mode, the agent may
slide to the action to the left or right of the intended action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Human-readable names for the four discrete actions.
# The action index is what the Q-table stores; the letter is only for printing.
ACTION_NAMES = np.array(["L", "D", "R", "U"])

# Movement effect for each action: (change in row, change in column).
# Example: action 2 = right, so the row stays the same and column increases by 1.
ACTION_TO_DELTA = {
    0: (0, -1),   # left
    1: (1, 0),    # down
    2: (0, 1),    # right
    3: (-1, 0),   # up
}


@dataclass
class StepResult:
    """Container for the result of one environment transition."""

    # The next state s_{t+1}, represented as one integer from 0 to 15.
    state: int

    # Immediate reward r_t. FrozenLake gives reward 1 only when reaching G.
    reward: float

    # True if the episode ended by reaching a hole H or the goal G.
    terminated: bool


class FrozenLake:
    """Small tabular Markov Decision Process (MDP) for FrozenLake."""

    def __init__(
        self,
        desc: tuple[str, ...] = ("SFFF", "FHFH", "FFFH", "HFFG"),
        is_slippery: bool = True,
        seed: int = 0,
    ) -> None:
        # Convert strings like "SFFF" into a 2D array of characters.
        # This makes it easy to look up what type of cell the agent moved into.
        self.desc = np.asarray([list(row) for row in desc])

        # Grid size. For the default map this is 4 rows x 4 columns.
        self.n_rows, self.n_cols = self.desc.shape

        # Tabular RL needs finite states and actions.
        # Here each grid square is one state, so 4 x 4 = 16 states.
        self.n_states = self.n_rows * self.n_cols

        # Four actions: left, down, right, up.
        self.n_actions = 4

        # If True, the environment is stochastic: intended actions can slide.
        # If False, each action moves exactly as requested.
        self.is_slippery = is_slippery

        # Random generator used for slippery movement.
        self.rng = np.random.default_rng(seed)

        # Find the starting cell S. State indexing is row-major:
        # state = row * number_of_columns + column.
        self.start_state = int(np.argwhere(self.desc == "S")[0][0] * self.n_cols)

        # Current state of the agent. reset() will put this back to start_state.
        self.state = self.start_state

    def reset(self) -> int:
        """Start a new episode and return the initial state."""
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> StepResult:
        """Apply one action and return (next_state, reward, done)."""

        if self.is_slippery:
            # Match Gym's FrozenLake idea: intended direction plus the two
            # perpendicular directions are equally likely.
            # Example: if the agent chooses down, it may go left, down, or right.
            action = int(self.rng.choice([(action - 1) % 4, action, (action + 1) % 4]))

        # Convert the current integer state back into a grid location.
        # Example on a 4-column grid: state 6 -> row 1, col 2.
        row, col = divmod(self.state, self.n_cols)

        # Translate the action into a proposed movement on the grid.
        d_row, d_col = ACTION_TO_DELTA[action]

        # Move to the proposed cell, but clip at the map boundary.
        # If the agent tries to move left from column 0, it stays in column 0.
        new_row = min(max(row + d_row, 0), self.n_rows - 1)
        new_col = min(max(col + d_col, 0), self.n_cols - 1)

        # Convert the grid location back to a single integer state.
        new_state = new_row * self.n_cols + new_col

        # Reward/termination logic:
        # - F and S: safe cells, reward 0, episode continues.
        # - H: hole, reward 0, episode ends.
        # - G: goal, reward 1, episode ends.
        cell = self.desc[new_row, new_col]
        reward = 1.0 if cell == "G" else 0.0
        terminated = cell in {"H", "G"}

        # Update the environment's internal state before returning.
        self.state = new_state
        return StepResult(new_state, reward, terminated)

    def render(self, state: int | None = None) -> str:
        """Return a printable grid with A marking the agent's current location."""
        state = self.state if state is None else state
        row, col = divmod(state, self.n_cols)
        grid = self.desc.copy()

        # Do not overwrite terminal symbols; this keeps holes and goal visible.
        if grid[row, col] not in {"H", "G"}:
            grid[row, col] = "A"
        return "\n".join(" ".join(r) for r in grid)


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """Choose an action using epsilon-greedy exploration.

    With probability epsilon, explore a random action.
    With probability 1 - epsilon, exploit the currently best Q-value.
    """

    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def train_q_learning(
    env: FrozenLake,
    episodes: int = 20_000,
    max_steps: int = 100,
    alpha: float = 0.10,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    seed: int = 1,
) -> tuple[np.ndarray, list[float]]:
    """Learn Q(s, a) with the standard tabular Q-learning update."""

    # Separate RNG for exploration decisions, so environment randomness and
    # action-selection randomness are easy to control independently.
    rng = np.random.default_rng(seed)

    # Q-table shape is number_of_states x number_of_actions.
    # Q[s, a] estimates the expected discounted return from taking action a
    # in state s, then following the learned policy afterward.
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

    # Store one return per episode. In FrozenLake, this is 1 for success and
    # 0 for failure because only the goal produces reward.
    episode_returns: list[float] = []

    # Start highly exploratory, then decay toward mostly-greedy behavior.
    epsilon = epsilon_start

    for _ in range(episodes):
        # Each episode starts from S.
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            # Choose action a_t from the current state s_t.
            action = epsilon_greedy_action(q_table, state, epsilon, rng)

            # Environment transition: (s_t, a_t) -> (r_t, s_{t+1}).
            result = env.step(action)

            # Q-learning is off-policy: it uses the greedy best next action
            # in the target, even if the behavior policy is epsilon-greedy.
            # If the next state is terminal, there is no future value.
            best_next_q = 0.0 if result.terminated else np.max(q_table[result.state])

            # Temporal-difference target:
            # target = immediate reward + discounted estimate of future reward.
            td_target = result.reward + gamma * best_next_q

            # TD error measures how surprising the transition was compared
            # with the current Q estimate.
            td_error = td_target - q_table[state, action]

            # Standard Q-learning update:
            # Q_new(s,a) = Q_old(s,a) + alpha * TD_error.
            q_table[state, action] += alpha * td_error

            # Move forward in time.
            state = result.state
            total_reward += result.reward

            # Stop early if the agent fell into a hole or reached the goal.
            if result.terminated:
                break

        episode_returns.append(total_reward)

        # Slowly reduce exploration. epsilon_min prevents exploration from
        # disappearing completely.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_returns


def evaluate_policy(
    env: FrozenLake,
    q_table: np.ndarray,
    episodes: int = 2_000,
    max_steps: int = 100,
) -> float:
    """Evaluate the greedy policy implied by the learned Q-table."""
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            # Evaluation uses no epsilon exploration: always choose argmax_a Q(s,a).
            action = int(np.argmax(q_table[state]))
            result = env.step(action)
            state = result.state
            if result.terminated:
                # A terminal reward of 1 means goal reached; 0 means hole.
                wins += int(result.reward > 0)
                break
    return wins / episodes


def policy_grid(env: FrozenLake, q_table: np.ndarray) -> str:
    """Show the learned greedy action for each non-terminal grid cell."""
    symbols = env.desc.copy()
    for state in range(env.n_states):
        row, col = divmod(state, env.n_cols)
        if symbols[row, col] in {"S", "F"}:
            # For each safe cell, print the action with highest Q-value.
            symbols[row, col] = ACTION_NAMES[int(np.argmax(q_table[state]))]
    return "\n".join(" ".join(row) for row in symbols)


def value_grid(env: FrozenLake, q_table: np.ndarray) -> str:
    """Show V(s) = max_a Q(s,a), the value of each state under the greedy policy."""
    values = np.max(q_table, axis=1).reshape(env.n_rows, env.n_cols)
    return "\n".join(" ".join(f"{v:5.2f}" for v in row) for row in values)


def run_one_episode(env: FrozenLake, q_table: np.ndarray, max_steps: int = 30) -> None:
    """Print one greedy rollout so we can see the policy in action."""
    state = env.reset()
    print("Sample greedy episode:")
    print(env.render(state))
    print()

    for step in range(1, max_steps + 1):
        # Follow the learned greedy policy.
        action = int(np.argmax(q_table[state]))
        result = env.step(action)
        state = result.state
        print(f"step={step:02d}, action={ACTION_NAMES[action]}, reward={result.reward:.0f}")
        print(env.render(state))
        print()
        if result.terminated:
            outcome = "goal reached" if result.reward > 0 else "fell into a hole"
            print(f"Episode ended: {outcome}")
            return

    print("Episode ended: max steps reached")


def main() -> None:
    # Use non-slippery mode for a clear classroom walkthrough. Change both
    # flags to True to study the harder stochastic FrozenLake setting.
    # We use separate train/eval environments so evaluation has its own random
    # seed when slippery mode is enabled.
    train_env = FrozenLake(is_slippery=False, seed=0)
    eval_env = FrozenLake(is_slippery=False, seed=123)

    # Learn the Q-table, then evaluate the greedy policy that it defines.
    q_table, returns = train_q_learning(train_env)
    success_rate = evaluate_policy(eval_env, q_table)

    print("FrozenLake Q-learning")
    print("=" * 60)
    print(f"Training success rate over last 1000 episodes: {np.mean(returns[-1000:]):.3f}")
    print(f"Greedy policy evaluation success rate:        {success_rate:.3f}")
    print()
    print("Learned greedy policy:")
    print(policy_grid(train_env, q_table))
    print()
    print("Learned state values max_a Q(s,a):")
    print(value_grid(train_env, q_table))
    print()

    # Print a deterministic demo episode. This makes the learned path easy to
    # follow, especially for students seeing FrozenLake for the first time.
    demo_env = FrozenLake(is_slippery=False, seed=999)
    run_one_episode(demo_env, q_table)


if __name__ == "__main__":
    main()
