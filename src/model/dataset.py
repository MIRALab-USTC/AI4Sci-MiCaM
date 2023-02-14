import os
import os.path as path
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from model.mol_graph import MolGraph
from model.mydataclass import batch_train_data, mol_train_data, train_data
from model.vocab import SubMotifVocab


class MolsDataset(Dataset):

    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

    def __getitem__(self, index):
        file = path.join(self.data_dir, self.files[index])
        return torch.load(file)
    
    def __len__(self):
        return len(self.files)

def batch_collate(batch: List[mol_train_data]) -> batch_train_data:
    batch_mols_graphs: List[Data] = []
    batch_props: List[torch.Tensor] = []
    batch_start_labels: List[int] = []
    batch_train_data_list: List[List[train_data]] = []
    motif_lists: List[List[int]] = []

    for data in batch:
        batch_mols_graphs.append(data.mol_graph)
        batch_props.append(data.props)
        batch_start_labels.append(data.start_label)
        batch_train_data_list.append(data.train_data_list)
        motif_lists.append(data.motif_list)

    motifs_list = list(set(sum(motif_lists, [])))
    motif_vocab = SubMotifVocab(MolGraph.MOTIF_VOCAB, motifs_list)
    motif_conns_idx = motif_vocab.get_conns_idx()
    motif_conns_num = len(motif_conns_idx)
    batch_start_labels = [motif_vocab.motif_idx_in_sublist(idx) for idx in batch_start_labels]

    offset, G_offset, conn_offset= 0, 0, motif_conns_num
    batch_train_graphs: List[Data] = []
    mol_idx, graph_idx, query_idx, cyclize_cand_idx, labels = [], [], [], [], []
    for bid, data_list in enumerate(batch_train_data_list):
        for data in data_list:
            query_atom, cyclize_cand, (motif_idx, conn_idx) = data.query_atom, data.cyclize_cand, data.label

            mol_idx.append(bid)
            graph_idx.append(G_offset)
            query_idx.append(query_atom + offset)
            cyclize_cand_idx.extend([cand + offset for cand in cyclize_cand])

            if motif_idx == -1:
                labels.append(conn_offset)
            else:
                labels.append(motif_vocab.get_conn_label(motif_idx, conn_idx))

            batch_train_graphs.append(data.graph)
            offset += len(data.graph.x)
            G_offset += 1
            conn_offset += len(cyclize_cand)

    return batch_train_data(
        batch_mols_graphs = Batch.from_data_list(batch_mols_graphs),
        batch_props = torch.Tensor(batch_props),
        batch_start_labels = torch.LongTensor(batch_start_labels),
        motifs_list = torch.LongTensor(motifs_list),
        batch_train_graphs = Batch.from_data_list(batch_train_graphs),
        mol_idx = torch.LongTensor(mol_idx),
        graph_idx = torch.LongTensor(graph_idx),
        query_idx = torch.LongTensor(query_idx),
        cyclize_cand_idx = torch.LongTensor(cyclize_cand_idx),
        motif_conns_idx = torch.LongTensor(motif_conns_idx),
        labels = torch.LongTensor(labels),
    )