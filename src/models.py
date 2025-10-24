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
