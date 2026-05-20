# Tabular Q-Learning — Reinforcement Learning

## How It Works
Q-Learning is a model-free, off-policy RL algorithm that learns an action-value function **Q(s, a)** — the expected cumulative discounted reward for taking action *a* in state *s* and following the optimal policy thereafter. A Q-table stores values for all (state, action) pairs. At each step the agent selects an action with an ε-greedy policy (explore randomly with probability ε, exploit the best known action otherwise), observes the reward and next state, then updates the table with the Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`. Q-values converge to optimal with sufficient exploration.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Discrete route optimisation** | Learn the shortest or cheapest path across a discretised ocean grid, avoiding shallow zones and traffic separation schemes |
| **Speed selection policy** | In a simplified state space (distance-to-port bin, weather bin), learn the optimal speed to minimise fuel + time cost |
| **Port entry sequencing** | Learn which berth queue to join when multiple options exist, based on estimated wait times |
| **Simple collision avoidance** | Train on a gridworld with other-vessel tokens to learn COLREGS-compliant give-way behaviour |

## State / Action / Reward Design
- **State**: grid cell (row, col) or discretised (distance, speed, weather) tuple
- **Action**: heading direction (N, NE, E, SE, S, SW, W, NW) or speed change (up, maintain, down)
- **Reward**: −1 per step (encourages shortest path), large negative for grounding / collision, large positive for reaching destination

## Notes
- Q-table size grows exponentially with state dimensions — use DQN for continuous or high-dimensional state spaces.
- Start here before DQN; simpler to debug and interpret.

## Run
```bash
python main.py
```
