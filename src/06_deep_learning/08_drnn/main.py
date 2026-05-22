from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DRNNPredictor(nn.Module):
    """Deep RNN sequence predictor for vessel trajectories.

    Stacks multiple RNN layers with configurable per-layer hidden sizes,
    passing the output of each layer as input to the next. Intermediate
    layers apply dropout for regularisation.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep (e.g. sog, cog, lat, lon).
    hidden_sizes : list[int]
        Hidden unit count for each RNN layer. Length determines depth.
        Default ``[128, 64, 32]``.
    output_size : int
        Output dimension — 1 for regression, N for N-class output. Default ``1``.
    dropout : float
        Dropout probability applied after each intermediate RNN layer. Default ``0.2``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_size = input_size
        for i, h in enumerate(hidden_sizes):
            self.layers.append(nn.RNN(in_size, h, batch_first=True))
            # Apply dropout after every layer except the last
            self.dropouts.append(
                nn.Dropout(dropout) if i < len(hidden_sizes) - 1 else nn.Identity()
            )
            in_size = h

        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out = x
        for rnn, drop in zip(self.layers, self.dropouts):
            out, _ = rnn(out)   # out: (batch, seq_len, hidden_size_i)
            out = drop(out)
        return self.fc(out[:, -1, :])  # use last timestep of deepest layer


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

    model = DRNNPredictor(input_size=n_features, hidden_sizes=[128, 64, 32], output_size=1, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training DRNN...")
    train(model, train_loader, criterion, optimizer, epochs=10)

    print("\nEvaluation:", evaluate(model, val_loader, criterion))
