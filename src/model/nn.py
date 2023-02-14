from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: nn = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = act

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = nn.BatchNorm1d(hidden_channels)
            else:
                norm = nn.Identity()
            self.norms.append(norm)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'

class GIN_virtual(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, edge_dim, depth, dropout, virtual: bool=True, pooling: str="mean", residual=True):
        super(GIN_virtual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_channels
        self.edge_dim = edge_dim
        self.depth = depth
        self.dropout = dropout
        self.residual = residual
        self.virtual = virtual
        self.pooling = pooling

        self.in_layer = MLP(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            out_channels = hidden_channels,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = dropout,
        )

        self.vitual_embed = nn.Embedding(1, hidden_channels)
        nn.init.constant_(self.vitual_embed.weight.data, 0)

        for i in range(depth):
            layer_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            layer = GINEConv(nn=layer_nn, train_eps=True, edge_dim=edge_dim)

            self.add_module(f"GNN_layer_{i}", layer)

            virtual_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            self.add_module(f"virtual_layer_{i}", virtual_layer)
        
        self.out_layer = MLP(
            in_channels = in_channels + hidden_channels * (depth + 1),
            hidden_channels = hidden_channels,
            out_channels = out_channels,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if self.virtual:
            v = torch.zeros(batch[-1].item() + 1, dtype=torch.long).to(x.device)
            virtual_embed = self.vitual_embed(v)

        x_list = [x, self.in_layer(x)]

        for i in range(self.depth):
            GNN_layer = getattr(self, f"GNN_layer_{i}")
            if self.virtual:
                virtual_layer = getattr(self, f"virtual_layer_{i}")
                x_list[-1] = x_list[-1] + virtual_embed[batch]

            x = GNN_layer(x_list[-1], edge_index, edge_attr)

            if self.residual:
                x = x + x_list[-1]
            
            x_list.append(x)

            if self.virtual:
                virtual_tmp = virtual_layer(global_add_pool(x_list[-1], batch) + virtual_embed)
                virtual_embed = virtual_embed + virtual_tmp if self.residual else virtual_tmp

        join_vecs = torch.cat(x_list, -1)
        nodes_reps = self.out_layer(join_vecs)

        if self.pooling == "mean":
            graph_reps = global_mean_pool(nodes_reps, batch)
        else:
            assert self.pooling == "add"
            graph_reps = global_add_pool(nodes_reps, batch)

        del x, x_list, join_vecs
        if self.virtual:
            del virtual_embed, virtual_tmp
        return nodes_reps, graph_reps