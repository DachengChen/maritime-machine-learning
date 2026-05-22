from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for vessel trajectory anomaly detection.

    Encodes a sequence to a latent vector, then decodes it back to the
    original sequence length. High reconstruction error indicates an anomaly.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        Latent dimension. Default ``32``.
    num_layers : int
        Number of LSTM layers in encoder and decoder. Default ``1``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        _, (h, c) = self.encoder(x)
        # Repeat latent for each timestep and decode
        seq_len = x.size(1)
        latent  = h[-1].unsqueeze(1).expand(-1, seq_len, -1)
        decoded, _ = self.decoder(latent)
        return self.output_layer(decoded)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error (higher = more anomalous)."""
        recon = self(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
) -> list[float]:
    criterion = nn.MSELoss()
    model.train()
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for (X_batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), X_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


def evaluate(model: nn.Module, X: torch.Tensor, threshold: float) -> dict:
    """Flag sequences whose reconstruction error exceeds ``threshold``."""
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X)
    anomalies = int((errors > threshold).sum().item())
    return {
        "mean_error": round(errors.mean().item(), 4),
        "threshold":  threshold,
        "anomalies":  anomalies,
        "total":      len(X),
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    SEQ_LEN, N, FEATURES = 20, 500, 4

    # Normal trajectories
    X_normal = torch.randn(N, SEQ_LEN, FEATURES) * 0.5

    train_loader = DataLoader(
        TensorDataset(X_normal[:400]),
        batch_size=32, shuffle=True,
    )

    model     = LSTMAutoencoder(input_size=FEATURES, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, optimizer, epochs=5)

    # Mix normal + anomalous sequences for evaluation
    X_anomalous = torch.randn(50, SEQ_LEN, FEATURES) * 5.0
    X_eval = torch.cat([X_normal[400:], X_anomalous], dim=0)

    # Use 95th-percentile of training error as threshold
    model.eval()
    with torch.no_grad():
        train_errors = model.reconstruction_error(X_normal[:400])
    threshold = float(train_errors.quantile(0.95).item())

    print("eval:", evaluate(model, X_eval, threshold))
