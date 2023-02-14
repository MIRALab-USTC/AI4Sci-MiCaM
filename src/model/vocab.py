import rdkit
import rdkit.Chem as Chem
from typing import List, Tuple, Dict
import torch
from model.utils import smiles2mol, get_conn_list
from collections import defaultdict


class Vocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vmap = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return len(self.vocab_list)

class MotifVocab(object):

    def __init__(self, pair_list: List[Tuple[str, str]]):
        self.motif_smiles_list = [motif for _, motif in pair_list]
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))

        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        conn_dict: Dict[int, Tuple[int, int]] = {}
        bond_type_motifs_dict = defaultdict(list)
        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            motif = smiles2mol(motif_smiles)
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))

            cur_orders = []
            vocab_conn_dict[motif_idx] = {}
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    bond_type = atom.GetBonds()[0].GetBondType()
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])
                    cur_orders.append(ranks[atom.GetIdx()])
                    bond_type_motifs_dict[bond_type].append(conn_offset)
                    nodes_idx.append(node_offset)
                    conn_offset += 1
                node_offset += 1
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()
        self.vocab_conn_dict = vocab_conn_dict
        self.conn_dict = conn_dict
        self.nodes_idx = nodes_idx
        self.num_atoms_dict = num_atoms_dict
        self.bond_type_conns_dict = bond_type_motifs_dict


    def __getitem__(self, smiles: str) -> int:
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1
    
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx
    
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]

class SubMotifVocab(object):

    def __init__(self, motif_vocab: MotifVocab, sublist: List[int]):
        self.motif_vocab = motif_vocab
        self.sublist = sublist
        self.idx2sublist_map = dict(zip(sublist, range(len(sublist))))

        node_offset, conn_offset, nodes_idx = 0, 0, []
        motif_idx_in_sublist = {}
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        for i, mid in enumerate(sublist):
            motif_idx_in_sublist[mid] = i
            vocab_conn_dict[mid] = {}
            for cid in motif_vocab.vocab_conn_dict[mid].keys():
                vocab_conn_dict[mid][cid] = conn_offset
                nodes_idx.append(node_offset + cid)
                conn_offset += 1
            node_offset += motif_vocab.num_atoms_dict[mid]
        self.vocab_conn_dict = vocab_conn_dict
        self.nodes_idx = nodes_idx
        self.motif_idx_in_sublist_map = motif_idx_in_sublist
    
    def motif_idx_in_sublist(self, motif_idx: int):
        return self.motif_idx_in_sublist_map[motif_idx]

    def get_conn_label(self, motif_idx: int, order_idx: int):
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self):
        return self.nodes_idx

    



