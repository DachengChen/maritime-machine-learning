# MLP Neural Network — Classification & Regression

## How It Works
A Multi-Layer Perceptron (MLP) stacks fully-connected (dense) layers separated by non-linear activation functions (e.g. ReLU). Each neuron computes a weighted sum of its inputs plus a bias, then applies the activation. During training, backpropagation computes the gradient of the loss with respect to every weight, and an optimiser (SGD, Adam) updates them iteratively. The depth and width of the network control its capacity to learn complex, non-linear input–output mappings. For classification, a softmax output and cross-entropy loss are used; for regression, a linear output neuron and MSE loss.

## Maritime Use Cases

| Use Case | Description |
|---|---|
| **Vessel type classification** | Learn complex non-linear decision boundaries over SOG, COG, heading, and dimensions to classify ship types |
| **ETA / SOG regression** | Predict continuous voyage metrics from multi-dimensional AIS feature vectors |
| **Anomaly scoring** | Train a binary classifier to distinguish normal vs. anomalous AIS reports |
| **Port congestion prediction** | Regress expected waiting time from multi-vessel traffic features |

## AIS Features Typically Used
- `sog`, `cog`, `heading` — Kinematic state
- `length`, `width`, `draught` — Physical dimensions
- `shiptype`, `navigationalstatus` — Categorical (one-hot encoded)

## Notes
- Requires feature normalisation (StandardScaler) for stable training.
- More data-hungry than tree-based methods; use regularisation (dropout, weight decay) on small AIS datasets.

## Run
```bash
python main.py
```
