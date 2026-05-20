from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CNN1DPredictor(nn.Module):
    """1-D Convolutional network for vessel sequence classification / regression.

    Applies two Conv1d layers over the time axis, then global average-pools
    to a fixed-size representation independent of sequence length.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    output_size : int
        Output dimension. Default ``1``.
    num_filters : int
        Number of filters in the first Conv1d layer. Doubled in the second. Default ``64``.
    kernel_size : int
        Convolutional kernel width. Default ``3``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        num_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=pad)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(num_filters * 2, output_size)
        self.relu  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) → Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


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
    SEQ_LEN, N, FEATURES = 20, 500, 4

    X = torch.randn(N, SEQ_LEN, FEATURES)
    y = X[:, -1, :1]

    split = int(N * 0.8)
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X[split:], y[split:]),  batch_size=32)

    model     = CNN1DPredictor(input_size=FEATURES, output_size=1, num_filters=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, criterion, optimizer, epochs=5)
    print("test:", evaluate(model, test_loader, criterion))
