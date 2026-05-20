from __future__ import annotations

import numpy as np


class QLearning:
    """Tabular Q-learning for vessel route optimisation.

    Learns a state–action value table Q(s, a) by interacting with an
    environment that provides (next_state, reward, done) transitions.

    Parameters
    ----------
    n_states : int
        Size of the discrete state space.
    n_actions : int
        Number of available actions.
    alpha : float
        Learning rate. Default ``0.1``.
    gamma : float
        Discount factor. Default ``0.99``.
    epsilon : float
        Initial ε for ε-greedy exploration. Default ``1.0``.
    epsilon_decay : float
        Multiplicative decay applied to ε after each episode. Default ``0.995``.
    epsilon_min : float
        Minimum ε. Default ``0.01``.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states    = n_states
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table     = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def train(self, env, episodes: int = 500, max_steps: int = 200) -> list[float]:
        """Train against ``env`` which must expose ``reset()`` and ``step(action)``."""
        returns = []
        for ep in range(episodes):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state  = next_state
                total += reward
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            returns.append(total)
            if (ep + 1) % 100 == 0:
                print(f"episode {ep + 1}/{episodes}  avg_return={np.mean(returns[-100:]):.2f}")
        return returns

    def policy(self) -> np.ndarray:
        """Return the greedy policy: best action per state."""
        return np.argmax(self.q_table, axis=1)


class SimpleRouteEnv:
    """Minimal grid-world proxy for a vessel choosing the shortest route.

    States are waypoints 0 … n_waypoints-1.
    Actions: 0 = stay, 1 = advance.
    Goal: reach state n_waypoints-1.
    """

    def __init__(self, n_waypoints: int = 10):
        self.n = n_waypoints
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action == 1:
            self.state = min(self.state + 1, self.n - 1)
        done   = self.state == self.n - 1
        reward = 1.0 if done else -0.1
        return self.state, reward, done


if __name__ == "__main__":
    np.random.seed(42)
    N_WAYPOINTS = 10
    env   = SimpleRouteEnv(n_waypoints=N_WAYPOINTS)
    agent = QLearning(n_states=N_WAYPOINTS, n_actions=2, alpha=0.1, gamma=0.99)
    agent.train(env, episodes=500)
    print("greedy policy:", agent.policy())
