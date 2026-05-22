from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Feed-forward Q-network mapping state vectors to Q-values."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network agent for vessel route / speed optimisation.

    Uses experience replay and a target network for stable training.

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
    epsilon : float
        Initial ε for ε-greedy exploration. Default ``1.0``.
    epsilon_decay : float
        Multiplicative ε decay per step. Default ``0.995``.
    epsilon_min : float
        Minimum ε. Default ``0.01``.
    buffer_size : int
        Replay buffer capacity. Default ``10_000``.
    batch_size : int
        Mini-batch size for each gradient update. Default ``64``.
    target_update_freq : int
        Steps between target-network weight copies. Default ``100``.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size  = batch_size
        self.target_update_freq = target_update_freq
        self._steps      = 0

        self.policy_net = DQNNetwork(state_dim, n_actions)
        self.target_net = DQNNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.policy_net(s).argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.FloatTensor(np.array(states))
        a  = torch.LongTensor(actions).unsqueeze(1)
        r  = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(np.array(next_states))
        d  = torch.FloatTensor(dones)

        q_vals   = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1).values
            targets = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, episodes: int = 300, max_steps: int = 200) -> list[float]:
        returns = []
        for ep in range(episodes):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.store(state, action, reward, next_state, done)
                self.learn()
                state  = next_state
                total += reward
                if done:
                    break
            returns.append(total)
            if (ep + 1) % 50 == 0:
                print(f"episode {ep + 1}/{episodes}  avg_return={np.mean(returns[-50:]):.2f}  ε={self.epsilon:.3f}")
        return returns


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
        reward = 1.0 if done else -0.01 * abs(self.speed - 12.0)  # penalise deviation from eco speed
        return self._obs(), reward, done

    def _obs(self) -> np.ndarray:
        return np.array([self.speed, self.distance], dtype=np.float32)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    env   = ContinuousRouteEnv()
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent.train(env, episodes=300)
