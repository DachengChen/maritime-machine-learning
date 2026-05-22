from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerPredictor(nn.Module):
    """Transformer encoder for vessel sequence modelling.

    Projects raw features to ``d_model`` dimensions, passes them through a
    stack of multi-head self-attention layers, then reads off the last
    position for prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    output_size : int
        Output dimension. Default ``1``.
    d_model : int
        Internal embedding dimension (must be divisible by ``nhead``). Default ``64``.
    nhead : int
        Number of attention heads. Default ``4``.
    num_layers : int
        Number of TransformerEncoder layers. Default ``2``.
    dropout : float
        Dropout rate. Default ``0.1``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc      = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


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

    model     = TransformerPredictor(input_size=FEATURES, output_size=1, d_model=64, nhead=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, criterion, optimizer, epochs=5)
    print("test:", evaluate(model, test_loader, criterion))
