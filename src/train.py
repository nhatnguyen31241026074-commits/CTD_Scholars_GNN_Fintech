import random
from typing import List, Any

import torch
import torch.optim as optim
import torch.nn as nn

# Optional imports
try:
    from torch_geometric.loader import NeighborLoader
except Exception:
    NeighborLoader = None

try:
    import wandb
except Exception:
    wandb = None


class SimpleData:
    """Minimal stand-in for torch_geometric.data.Data when PyG is unavailable.

    Only provides .x (node features tensor) and .edge_index (2 x E tensor) and a .to(device) method.
    """

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.x = x
        self.edge_index = edge_index

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


def train_one_epoch(model: nn.Module, loader: List[Any], optimizer: optim.Optimizer, device: torch.device, sequence_length: int) -> float:
    """Train model for one epoch.

    Args:
        model: ST_GAD_Model instance
        loader: iterable that yields a sequence/list of Data snapshots per item
        optimizer: optimizer
        device: torch device
        sequence_length: number of most recent snapshots to use

    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    count = 0

    for batch in loader:
        # batch expected to be a list/sequence of snapshots (Data objects)
        if isinstance(batch, (list, tuple)):
            seq = list(batch)[-sequence_length:]
        else:
            # if loader yields a dataset object, try to get snapshots attribute
            try:
                seq = list(batch.snapshots)[-sequence_length:]
            except Exception:
                # fallback: treat batch itself as a single snapshot
                seq = [batch]

        # move snapshots to device
        seq_device = []
        for s in seq:
            try:
                s_dev = s.to(device)
            except Exception:
                # if object doesn't have to(), assume it's a SimpleData-like dict
                s_dev = s
            seq_device.append(s_dev)

        optimizer.zero_grad()
        H, H_recon = model(seq_device)
        loss = criterion(H, H_recon)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss


def main_training_loop(
    data_path: str = "../data",
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
    sequence_length: int = 3,
    gnn_config: dict = None,
    lstm_config: dict = None,
    ae_config: dict = None,
    wandb_project: str = None,
    wandb_config: dict = None,
):
    """Main training loop skeleton.

    This function is intentionally flexible: it attempts to import user modules lazily and will
    fall back to a small synthetic dataset if loading real snapshots is not implemented.
    """
    # init wandb if available and requested
    if wandb_project and wandb is not None:
        wandb.init(project=wandb_project, config=wandb_config or {})
    elif wandb_project:
        print("wandb requested but not installed; continuing without logging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Lazy import of model to avoid import-time dependency issues
    try:
        from models import ST_GAD_Model
    except Exception:
        try:
            from src.models import ST_GAD_Model
        except Exception as e:
            raise ImportError("Could not import ST_GAD_Model from src.models or models. Ensure src/models.py is on PYTHONPATH and dependencies are installed.") from e

    # Default configs if not provided
    gnn_config = gnn_config or {"node_feature_dim": 1, "gnn_hidden_dim": 64, "gnn_layers": 2}
    lstm_config = lstm_config or {"lstm_hidden_dim": 64, "lstm_layers": 2}
    ae_config = ae_config or {"ae_latent_dim": 16}

    # initialize model
    model = ST_GAD_Model(
        node_feature_dim=gnn_config.get("node_feature_dim", 1),
        gnn_hidden_dim=gnn_config.get("gnn_hidden_dim", 64),
        gnn_layers=gnn_config.get("gnn_layers", 2),
        aggregator=gnn_config.get("aggregator", "mean"),
        gnn_dropout=gnn_config.get("gnn_dropout", 0.3),
        lstm_hidden_dim=lstm_config.get("lstm_hidden_dim", 64),
        lstm_layers=lstm_config.get("lstm_layers", 2),
        lstm_dropout=lstm_config.get("lstm_dropout", 0.2),
        ae_latent_dim=ae_config.get("ae_latent_dim", 16),
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data loading: attempt to load saved snapshots using data_processing.create_graph_snapshot if available
    loader = []
    try:
        # try to import helper
        try:
            from data_processing import create_graph_snapshot, generate_synthetic_transactions
        except Exception:
            from src.data_processing import create_graph_snapshot, generate_synthetic_transactions

        # create a tiny synthetic dataset: sequence_length snapshots with slightly varying data
        num_nodes = 100
        for s in range(50):
            # generate synthetic transactions and build a graph snapshot (may raise if torch_geometric missing)
            df = generate_synthetic_transactions(num_users=num_nodes, num_transactions=500, start_date="2025-01-01", end_date="2025-01-02")
            try:
                snapshot = create_graph_snapshot(df, [f"user_{i+1}" for i in range(num_nodes)])
            except Exception:
                # fallback to SimpleData
                x = torch.randn((num_nodes, gnn_config.get("node_feature_dim", 1)), dtype=torch.float)
                # random edges
                e1 = torch.randint(0, num_nodes, (500,), dtype=torch.long)
                e2 = torch.randint(0, num_nodes, (500,), dtype=torch.long)
                edge_index = torch.stack([e1, e2], dim=0)
                snapshot = SimpleData(x, edge_index)

            loader.append([snapshot])

    except Exception:
        # If anything fails, build a dummy loader of random SimpleData sequences
        print("Falling back to synthetic SimpleData loader (no PyG available or data load failed)")
        num_nodes = 100
        for _ in range(50):
            seq = []
            for _ in range(sequence_length):
                x = torch.randn((num_nodes, gnn_config.get("node_feature_dim", 1)), dtype=torch.float)
                e1 = torch.randint(0, num_nodes, (500,), dtype=torch.long)
                e2 = torch.randint(0, num_nodes, (500,), dtype=torch.long)
                edge_index = torch.stack([e1, e2], dim=0)
                seq.append(SimpleData(x, edge_index))
            loader.append(seq)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, device, sequence_length)
        print(f"Epoch {epoch}/{num_epochs} - train_loss: {avg_loss:.6f}")
        if wandb is not None and wandb_project:
            wandb.log({"epoch": epoch, "train_loss": avg_loss})

    if wandb is not None and wandb_project:
        wandb.finish()

    # optional save
    try:
        torch.save(model.state_dict(), "st_gad_model.pth")
        print("Saved model to st_gad_model.pth")
    except Exception:
        print("Could not save model state (permission or missing torch).")


if __name__ == "__main__":
    # sample config
    main_training_loop(
        data_path="../data",
        num_epochs=2,
        learning_rate=1e-3,
        batch_size=1,
        sequence_length=3,
        gnn_config={"node_feature_dim": 1, "gnn_hidden_dim": 64, "gnn_layers": 2},
        lstm_config={"lstm_hidden_dim": 64, "lstm_layers": 2},
        ae_config={"ae_latent_dim": 16},
        wandb_project=None,
        wandb_config=None,
    )
