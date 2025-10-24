import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from typing import Optional


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder using PyTorch Geometric's SAGEConv layers.

    Parameters
    - in_channels: input node feature dimension
    - hidden_channels: hidden (and output) dimension for SAGE layers
    - num_layers: number of SAGE layers to stack (>=1)
    - aggregator: aggregation type for SAGEConv ('mean' or 'lstm')
    - dropout_rate: dropout probability applied after each layer

    The module applies: [SAGEConv -> ReLU -> Dropout -> (LayerNorm)] repeated num_layers times
    and returns the final node embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.dropout_rate = dropout_rate

        # build SAGEConv layers
        convs = []
        # first layer maps from in_channels -> hidden_channels
        convs.append(gnn.SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        # subsequent layers map hidden -> hidden
        for _ in range(1, num_layers):
            convs.append(gnn.SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        self.convs = nn.ModuleList(convs)

        # activation, dropout, optional layernorm
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        # LayerNorm on hidden dimension to stabilize training
        self.layernorms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [num_nodes, in_channels] or [num_nodes, hidden_channels] if continuing
            edge_index: [2, num_edges] adjacency index

        Returns:
            node embeddings tensor of shape [num_nodes, hidden_channels]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            # apply non-linearity and regularization
            h = self.act(h)
            h = self.dropout(h)
            # apply layer norm
            h = self.layernorms[i](h)

        return h


__all__ = ["GraphSAGEEncoder"]


class TemporalAggregatorLSTM(nn.Module):
    """Temporal aggregator using an LSTM to summarize a sequence of node embeddings.

    The module expects input shape (batch_size, sequence_length, input_dim) with batch_first=True.
    It returns the last time-step output from the top LSTM layer with shape (batch_size, hidden_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # dropout in LSTM is only used when num_layers > 1
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )

    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        # sequence_embeddings: (batch_size, sequence_length, input_dim)
        out, (h_n, c_n) = self.lstm(sequence_embeddings)
        # out: (batch_size, sequence_length, hidden_dim)
        # take the last time step
        last = out[:, -1, :]
        return last


class EmbeddingAutoencoder(nn.Module):
    """Simple autoencoder for embedding dimensionality reduction/reconstruction."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(latent_dim, input_dim // 2), latent_dim),
        )

        # Decoder: latent_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(latent_dim, input_dim // 2), input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class ST_GAD_Model(nn.Module):
    """Spatio-temporal graph anomaly detection model composed of:
    - GraphSAGEEncoder: per-snapshot spatial encoder
    - TemporalAggregatorLSTM: aggregates per-node embeddings across time
    - EmbeddingAutoencoder: reconstructs temporal embeddings for anomaly scoring

    Notes:
    - For simplicity we treat nodes as the batch dimension when feeding sequences to the LSTM.
      That is, for N nodes and T snapshots we arrange data as (batch_size=N, sequence_length=T, input_dim=gnn_hidden).
    - This assumes a consistent node ordering across snapshots (e.g., using a fixed user list).
    """

    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        aggregator: str = "mean",
        gnn_dropout: float = 0.3,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        ae_latent_dim: int = 32,
    ) -> None:
        super().__init__()
        # spatial encoder
        self.gnn = GraphSAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=gnn_hidden_dim,
            num_layers=gnn_layers,
            aggregator=aggregator,
            dropout_rate=gnn_dropout,
        )

        # temporal aggregator
        self.temporal = TemporalAggregatorLSTM(
            input_dim=gnn_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout_rate=lstm_dropout,
        )

        # autoencoder
        self.autoencoder = EmbeddingAutoencoder(input_dim=lstm_hidden_dim, latent_dim=ae_latent_dim)

    def forward(self, data_sequence: list) -> tuple:
        """Forward pass over a sequence of PyG Data snapshots.

        Args:
            data_sequence: list of Data objects (length T). Each Data must contain x and edge_index.

        Returns:
            H: tensor (num_nodes, lstm_hidden_dim) temporal embeddings
            H_recon: tensor (num_nodes, lstm_hidden_dim) reconstructed embeddings
        """
        if len(data_sequence) == 0:
            raise ValueError("data_sequence must contain at least one snapshot")

        # compute spatial embeddings for each snapshot
        embeddings = []
        for data in data_sequence:
            x = data.x
            edge_index = data.edge_index
            h_t = self.gnn(x, edge_index)  # [num_nodes, gnn_hidden_dim]
            embeddings.append(h_t)

        # stack embeddings -> [T, num_nodes, gnn_hidden_dim]
        stacked = torch.stack(embeddings, dim=1)  # [num_nodes, T, gnn_hidden_dim]

        # pass through temporal aggregator; treat nodes as batch
        H = self.temporal(stacked)  # [num_nodes, lstm_hidden_dim]

        # reconstruct with autoencoder
        H_recon = self.autoencoder(H)

        return H, H_recon


__all__ = ["GraphSAGEEncoder", "TemporalAggregatorLSTM", "EmbeddingAutoencoder", "ST_GAD_Model"]

