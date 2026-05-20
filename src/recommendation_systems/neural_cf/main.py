from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering (NCF) for vessel route recommendations.

    Combines a Generalised Matrix Factorisation (GMF) path with a
    Multi-Layer Perceptron (MLP) path, then fuses both for the final
    rating prediction.

    Parameters
    ----------
    n_users : int
        Total number of users (vessels).
    n_items : int
        Total number of items (routes / ports).
    embed_dim : int
        Embedding dimension for both GMF and MLP paths. Default ``16``.
    mlp_layers : list[int]
        Hidden layer sizes of the MLP path. Default ``[64, 32, 16]``.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 16,
        mlp_layers: list[int] | None = None,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_item = nn.Embedding(n_items, embed_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_item = nn.Embedding(n_items, embed_dim)

        # MLP tower
        layers: list[nn.Module] = []
        in_dim = embed_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        gmf_out = self.gmf_user(user_ids) * self.gmf_item(item_ids)

        # MLP path
        mlp_in  = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        return self.output(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)

    def recommend(self, user_id: int, n_items: int, top_n: int = 5) -> list[int]:
        """Return top-``top_n`` item indices for ``user_id``."""
        self.eval()
        with torch.no_grad():
            users = torch.full((n_items,), user_id, dtype=torch.long)
            items = torch.arange(n_items, dtype=torch.long)
            scores = self(users, items).numpy()
        return list(scores.argsort()[::-1][:top_n])


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
        for user_ids, item_ids, ratings in loader:
            optimizer.zero_grad()
            loss = criterion(model(user_ids, item_ids), ratings)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)

    N_USERS, N_ITEMS = 50, 20
    # Synthetic (user, item, rating) triples
    users   = torch.randint(0, N_USERS, (400,))
    items   = torch.randint(0, N_ITEMS, (400,))
    ratings = torch.randint(1, 6, (400,)).float()

    loader = DataLoader(TensorDataset(users, items, ratings), batch_size=32, shuffle=True)

    model     = NeuralCollaborativeFiltering(N_USERS, N_ITEMS, embed_dim=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, loader, criterion, optimizer, epochs=5)
    print("recommendations for vessel 0:", model.recommend(0, N_ITEMS, top_n=5))
