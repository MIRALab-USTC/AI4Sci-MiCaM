import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from model.mol_graph import MolGraph, MAX_NUM_ATOMS, NUM_EDGE_TYPES, ATOM_FEATURES
import networkx as nx
from typing import Tuple, List
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean

class Atom_Embedding(nn.Module):
    def __init__(self,
        atom_embed_size: List[int]
    ) -> None:
        super().__init__()
        assert len(atom_embed_size) == len(ATOM_FEATURES)
        for i in range(len(ATOM_FEATURES)):
            f_embed = nn.Embedding(ATOM_FEATURES[i].size(), atom_embed_size[i])
            self.add_module(f"f_embed_{i}", f_embed)

    def forward(self,
        atom_features: torch.Tensor
    ) -> torch.Tensor:
        features = torch.split(atom_features, 1, dim=-1)
        features = [f.long().view([-1]) for f in features]
        return torch.cat( [getattr(self, f"f_embed_{i}")(f) for i, f in enumerate(features)], dim=-1)

    def reset_parameters(self) -> None:
        for i in range(len(ATOM_FEATURES)):
            nn.init.xavier_normal_(getattr(self, f"f_embed_{i}").weight)

class Motif_Embedding(nn.Module):
    def __init__(self,
        motif_embed_size: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.f_embed1 = nn.Embedding(MolGraph.MOTIF_VOCAB.size()[0], motif_embed_size[1], max_norm=512, scale_grad_by_freq=True)
        self.f_embed2 = nn.Embedding(MolGraph.MOTIF_VOCAB.size()[1], motif_embed_size[1], max_norm=512, scale_grad_by_freq=True)
        self.dropout = nn.Dropout(dropout)
        self.idx = MolGraph.MOTIF_VOCAB.get_idx()
        self.reset_parameters()

    def forward(self) -> torch.Tensor:
        if not self.idx.is_cuda:
            self.idx = self.idx.cuda()
        weight1 = self.dropout(self.f_embed1.weight)
        weight2 = self.dropout(self.f_embed2.weight) + weight1[self.idx]
        return (self.idx, weight1, weight2)
    
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.f_embed1.weight)
        nn.init.zeros_(self.f_embed2.weight)
        
class Encoder(nn.Module):
    def __init__(self,
        atom_embedding: Atom_Embedding,
        edge_embedding: nn.Embedding,
        GNN: torch.nn,
    ) -> None:
        super(Encoder, self).__init__()

        self.atom_embedding = atom_embedding
        self.edge_embedding = edge_embedding
        self.GNN = GNN

        self.hidden_size = GNN.out_channels

    def embed_graph(self, x, edge_attr):
        
        x = self.atom_embedding(x)
        edge_attr = self.edge_embedding(edge_attr.long().view([-1]))

        return x, edge_attr

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_attr = self.embed_graph(data.x, data.edge_attr)
        nodes_reps, graph_reps = self.GNN(x, data.edge_index, edge_attr, data.batch)
        return nodes_reps, graph_reps