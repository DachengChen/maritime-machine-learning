# REINFORCE (Policy Gradient) — Reinforcement Learning

## How It Works
REINFORCE is a Monte Carlo policy gradient algorithm. Instead of learning a value function, it directly parameterises the policy π(a|s; θ) as a neural network (outputting action probabilities) and updates θ to increase the log-probability of actions that led to high returns. A full episode is sampled, then the return *G_t* (discounted cumulative reward from step *t*) is computed and used to scale the policy gradient: `∇θ J ≈ Σ_t G_t · ∇θ log π(a_t|s_t; θ)`. Subtracting a baseline (e.g. the mean return) reduces variance. The approach is on-policy, simple, and works naturally with stochastic policies needed in uncertain maritime environments.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Stochastic voyage planning** | Learn a probabilistic routing policy that naturally handles uncertainty in weather, current, and port availability |
| **Multi-objective optimisation** | Optimise a trade-off between fuel, time, and safety by shaping the reward function; the stochastic policy explores the Pareto front |
| **Fishing strategy learning** | Learn a catch-maximising patrol policy over a fishing ground modelled as a stochastic resource environment |
| **Adaptive speed scheduling** | Derive a speed-profile policy that adapts in real time to stochastic ETA windows and charter-party penalties |

## State / Action / Reward Design
- **State**: continuous voyage state (position, speed, heading, time-to-deadline, weather proxy)
- **Action**: probability distribution over (heading, speed) pairs — sampled during exploration
- **Reward**: ETA bonus, fuel cost penalty, safety penalty; shaped to encourage smooth voyages

## Notes
- High gradient variance; use normalised returns and a value-function baseline (Actor-Critic) for more stable training.
- REINFORCE is the conceptual foundation for PPO, A2C, and other modern policy-gradient methods.

## Run
```bash
python main.py
```
