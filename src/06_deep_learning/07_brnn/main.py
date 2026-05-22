from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BRNNPredictor(nn.Module):
    """Bidirectional RNN-based sequence predictor for vessel trajectories.

    Processes the sequence in both forward and backward directions and
    concatenates the two hidden states, giving the model access to both
    past and future context at each timestep.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep (e.g. sog, cog, lat, lon).
    hidden_size : int
        Number of hidden units per direction. Default ``64``.
    num_layers : int
        Number of stacked BRNN layers. Default ``2``.
    output_size : int
        Output dimension — 1 for regression, N for N-class output. Default ``1``.
    dropout : float
        Dropout between layers (ignored when num_layers=1). Default ``0.0``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.brnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional doubles the output feature dimension
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.brnn(x)
        # out: (batch, seq_len, hidden_size * 2) — concatenated fwd + bwd
        return self.fc(out[:, -1, :])  # use last timestep


def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
) -> list[float]:
    model.train()
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            total += criterion(model(X_batch), y_batch).item()
    return {"loss": round(total / len(loader), 4)}


if __name__ == "__main__":
    torch.manual_seed(42)

    # Synthetic AIS-like data: 200 sequences, length 20, 4 features
    seq_len, n_features = 20, 4
    X = torch.randn(200, seq_len, n_features)
    y = torch.randn(200, 1)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = BRNNPredictor(input_size=n_features, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training BRNN...")
    train(model, train_loader, criterion, optimizer, epochs=10)

    print("\nEvaluation:", evaluate(model, val_loader, criterion))
