from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Stochastic policy network: maps states to action probabilities."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) agent for route optimisation.

    Collects full episode trajectories, then updates the policy using the
    discounted return as the advantage signal.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the continuous state vector.
    n_actions : int
        Number of discrete actions.
    lr : float
        Learning rate. Default ``1e-3``.
    gamma : float
        Discount factor. Default ``0.99``.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.gamma  = gamma
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        s    = torch.FloatTensor(state).unsqueeze(0)
        dist = Categorical(self.policy(s))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def _returns(self, rewards: list[float]) -> torch.Tensor:
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update(self, log_probs: list[torch.Tensor], rewards: list[float]):
        returns   = self._returns(rewards)
        loss      = -torch.stack(log_probs) * returns
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

    def train(self, env, episodes: int = 500, max_steps: int = 200) -> list[float]:
        returns_log = []
        for ep in range(episodes):
            state    = env.reset()
            log_probs, rewards = [], []
            for _ in range(max_steps):
                action, lp = self.select_action(state)
                next_state, reward, done = env.step(action)
                log_probs.append(lp)
                rewards.append(reward)
                state = next_state
                if done:
                    break
            self.update(log_probs, rewards)
            returns_log.append(sum(rewards))
            if (ep + 1) % 100 == 0:
                print(f"episode {ep + 1}/{episodes}  avg_return={np.mean(returns_log[-100:]):.2f}")
        return returns_log


class ContinuousRouteEnv:
    """Simple continuous-state environment simulating vessel speed control.

    State : [current_speed, distance_to_destination]
    Actions: 0 = decelerate, 1 = maintain, 2 = accelerate
    """

    def __init__(self):
        self.state_dim = 2
        self.n_actions = 3

    def reset(self) -> np.ndarray:
        self.speed    = np.random.uniform(5.0, 15.0)
        self.distance = np.random.uniform(50.0, 200.0)
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        delta = {0: -1.0, 1: 0.0, 2: 1.0}[action]
        self.speed    = np.clip(self.speed + delta, 0.0, 25.0)
        self.distance = max(0.0, self.distance - self.speed)
        done   = self.distance <= 0
        reward = 1.0 if done else -0.01 * abs(self.speed - 12.0)
        return self._obs(), reward, done

    def _obs(self) -> np.ndarray:
        return np.array([self.speed, self.distance], dtype=np.float32)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    env   = ContinuousRouteEnv()
    agent = REINFORCEAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent.train(env, episodes=500)
