# Deep Q-Network (DQN) — Reinforcement Learning

## How It Works
DQN replaces the Q-table with a neural network that approximates Q(s, a) for continuous or high-dimensional state spaces. Two key stabilisation techniques make training work: (1) **Experience Replay** — transitions (s, a, r, s') are stored in a replay buffer and sampled randomly to break temporal correlations; (2) **Target Network** — a periodically frozen copy of the Q-network is used to compute target Q-values, preventing oscillating updates. The network is trained by minimising the MSE between predicted Q-values and targets computed from the Bellman equation. DQN was the first algorithm to play Atari games at human level and scales to complex real-world control.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Continuous-state route optimisation** | Learn optimal heading and speed from a real-valued state (position, speed, heading, wind, current) across open-ocean passages |
| **Dynamic obstacle avoidance** | React to other vessels' positions and velocities in a continuous environment, learning COLREGS-compliant manoeuvres |
| **Fuel-optimal speed control** | Optimise engine throttle continuously against a reward signal combining fuel cost and schedule adherence |
| **Port approach sequencing** | Learn when to slow down, request pilot, and proceed to berth given real-valued congestion and tidal state |

## State / Action / Reward Design
- **State**: continuous vector (vessel position, SOG, COG, distance to destination, time, nearby-vessel features)
- **Action**: discrete set of (heading change, speed change) combinations
- **Reward**: −fuel per step, −schedule-delay penalty, large negative for collision/grounding, large positive at safe arrival

## Notes
- Requires thousands of simulated episodes; use a maritime simulator or synthesised environment.
- For continuous *action* spaces, use DDPG or TD3 instead of DQN.

## Run
```bash
python main.py
```
