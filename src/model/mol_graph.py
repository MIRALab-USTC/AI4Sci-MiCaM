"""For molecular graph processing."""
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit.Chem import Descriptors
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from merging_operation_learning import merge_nodes
from model.mydataclass import batch_train_data, mol_train_data, train_data
from model.utils import (fragment2smiles, get_conn_list, graph2smiles,
                         networkx2data, smiles2mol)
from model.vocab import MotifVocab, SubMotifVocab, Vocab

RDContribDir = os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib')
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
import sascorer

sys.path.pop()

ATOM_SYMBOL_VOCAB = Vocab(['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F'])
ATOM_ISAROMATIC_VOCAB = Vocab([True, False])
ATOM_FORMALCHARGE_VOCAB = Vocab(["*", -1, 0, 1, 2, 3])
ATOM_NUMEXPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3])
ATOM_NUMIMPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3])
ATOM_FEATURES = [ATOM_SYMBOL_VOCAB, ATOM_ISAROMATIC_VOCAB, ATOM_FORMALCHARGE_VOCAB, ATOM_NUMEXPLICITHS_VOCAB, ATOM_NUMIMPLICITHS_VOCAB]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_VOCAB = Vocab(BOND_LIST)

LOGP_MEAN, LOGP_VAR = 3.481587226600002, 1.8185146774225027
MOLWT_MEAN, MOLWT_VAR = 396.7136355500001, 110.55283206754517
QED_MEAN, QED_VAR = 0.5533041888502863, 0.21397359224960685
SA_MEAN, SA_VAR = 2.8882909807901354, 0.8059540682960904

class MolGraph(object):

    @classmethod
    def load_operations(cls, operation_path: str, num_operations: int=500):
        MolGraph.NUM_OPERATIONS = num_operations
        MolGraph.OPERATIONS = [code.strip('\r\n') for code in open(operation_path)]
        MolGraph.OPERATIONS = MolGraph.OPERATIONS[:num_operations]
    
    @classmethod
    def load_vocab(cls, vocab_path: str):
        pair_list = [line.strip("\r\n").split() for line in open(vocab_path)]
        MolGraph.MOTIF_VOCAB = MotifVocab(pair_list)
        MolGraph.MOTIF_LIST = MolGraph.MOTIF_VOCAB.motif_smiles_list

    def __init__(self,
        smiles: str,
        tokenizer: str="graph",
    ):  
        assert tokenizer in ["graph", "motif"], \
            "The variable `process_level` should be 'graph' or 'motif'. "
        self.smiles = smiles
        self.mol = smiles2mol(smiles, sanitize=True)
        self.mol_graph = self.get_mol_graph()
        self.init_mol_graph = self.mol_graph.copy()
        
        if tokenizer == "motif":
            self.merging_graph = self.get_merging_graph()
            self.refragment()
            self.motifs = self.get_motifs()

    def get_mol_graph(self) -> nx.Graph:
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        for atom in self.mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['smarts'] = atom.GetSmarts()
            graph.nodes[atom.GetIdx()]['atom_indices'] = set([atom.GetIdx()])
            graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)

        for bond in self.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        return graph
    
    def get_merging_graph(self) -> nx.Graph:
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        merging_graph = mol_graph.copy()
        for code in self.OPERATIONS:
            for (node1, node2) in mol_graph.edges:
                if not merging_graph.has_edge(node1, node2):
                    continue
                atom_indices = merging_graph.nodes[node1]['atom_indices'].union(merging_graph.nodes[node2]['atom_indices'])
                pattern = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
                if pattern == code:
                    merge_nodes(merging_graph, node1, node2)
            mol_graph = merging_graph.copy()
        return nx.convert_node_labels_to_integers(merging_graph)

    def refragment(self) -> None:
        mol_graph = self.mol_graph.copy()
        merging_graph = self.merging_graph

        for node in merging_graph.nodes:
            atom_indices = self.merging_graph.nodes[node]['atom_indices']
            merging_graph.nodes[node]['motif_no_conn'] = fragment2smiles(self.mol, atom_indices)
            for atom_idx in atom_indices:
                mol_graph.nodes[atom_idx]['bpe_node'] = node

        for node1, node2 in self.mol_graph.edges:
            bpe_node1, bpe_node2 = mol_graph.nodes[node1]['bpe_node'], mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 != bpe_node2:
                conn1 = len(mol_graph)
                mol_graph.add_node(conn1)
                mol_graph.add_edge(node1, conn1)

                conn2 = len(mol_graph)
                mol_graph.add_node(conn2)
                mol_graph.add_edge(node2, conn2)
                
                mol_graph.nodes[conn1]['smarts'] = '*'
                mol_graph.nodes[conn1]['targ_atom'] = node2
                mol_graph.nodes[conn1]['merge_targ'] = conn2
                mol_graph.nodes[conn1]['anchor'] = node1
                mol_graph.nodes[conn1]['bpe_node'] = bpe_node1
                mol_graph[node1][conn1]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node1][conn1]['label'] = mol_graph[node1][node2]['label']
                merging_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node2][conn2]['label'] = mol_graph[node1][node2]['label']
                merging_graph.nodes[bpe_node2]['atom_indices'].add(conn2)
                mol_graph.nodes[conn2]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

        for node in merging_graph.nodes:
            atom_indices = merging_graph.nodes[node]['atom_indices']
            motif_graph = mol_graph.subgraph(atom_indices)
            merging_graph.nodes[node]['motif'] = graph2smiles(motif_graph)

        self.mol_graph = mol_graph

    def get_motifs(self) -> Set[str]:
        return [(self.merging_graph.nodes[node]['motif_no_conn'], self.merging_graph.nodes[node]['motif']) for node in self.merging_graph.nodes]

    def relabel(self):
        mol_graph = self.mol_graph
        bpe_graph = self.merging_graph

        for node in bpe_graph.nodes:
            bpe_graph.nodes[node]['internal_edges'] = []
            atom_indices = bpe_graph.nodes[node]['atom_indices']
            
            fragment_graph = mol_graph.subgraph(atom_indices)
            motif_smiles_with_idx = graph2smiles(fragment_graph, with_idx=True)
            motif_with_idx = smiles2mol(motif_smiles_with_idx)
            conn_list, ordermap = get_conn_list(motif_with_idx, use_Isotope=True)
           
            bpe_graph.nodes[node]['conn_list'] = conn_list
            bpe_graph.nodes[node]['ordermap'] = ordermap
            bpe_graph.nodes[node]['label'] = MolGraph.MOTIF_VOCAB[ bpe_graph.nodes[node]['motif'] ]
            bpe_graph.nodes[node]['num_atoms'] = len(atom_indices)

        for node1, node2 in bpe_graph.edges:
            self.merging_graph[node1][node2]['label'] = 0

        edge_dict = {}
        for edge, (node1, node2, attr) in enumerate(mol_graph.edges(data=True)):
            edge_dict[(node1, node2)] = edge_dict[(node2, node1)] = edge
            bpe_node1 = mol_graph.nodes[node1]['bpe_node']
            bpe_node2 = mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 == bpe_node2:
                bpe_graph.nodes[bpe_node1]['internal_edges'].append(edge)
        
        for node, attr in mol_graph.nodes(data=True): 
            if attr['smarts'] == '*':
                anchor = attr['anchor']
                targ_atom = attr['targ_atom']
                mol_graph.nodes[node]['edge_to_anchor'] = edge_dict[(node, anchor)]
                mol_graph.nodes[node]['merge_edge'] = edge_dict[(anchor, targ_atom)]
    
    def get_props(self) -> List[float]:
        mol = self.mol
        logP = (Descriptors.MolLogP(mol) - LOGP_MEAN) / LOGP_VAR
        Wt = (Descriptors.MolWt(mol) - MOLWT_MEAN) / MOLWT_VAR
        qed = (Descriptors.qed(mol) - QED_MEAN) / QED_VAR
        sa = (sascorer.calculateScore(mol) - SA_MEAN) / SA_VAR
        properties = [logP, Wt, qed, sa]
        return properties

    def get_data(self) -> mol_train_data:
        
        self.relabel()
        init_mol_graph, mol_graph, bpe_graph = self.init_mol_graph, self.mol_graph, self.merging_graph
        init_mol_graph_data, _ = networkx2data(init_mol_graph)
        motifs_list, conn_list = [], []
        train_data_list: List[train_data] = []

        nodes_num_atoms = dict(bpe_graph.nodes(data='num_atoms'))
        node = max(nodes_num_atoms, key=nodes_num_atoms.__getitem__)
        start_label = bpe_graph.nodes[node]['label']
        motifs_list.append(start_label)

        conn_list.extend(self.merging_graph.nodes[node]['conn_list'])
        subgraph = nx.Graph()
        subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

        while len(conn_list) > 0:
            query_atom = conn_list[0]
            targ = mol_graph.nodes[query_atom]['merge_targ']
            
            subgraph_data, mapping = networkx2data(subgraph)

            if targ in conn_list:  
                cur_mol_smiles_with_idx = graph2smiles(subgraph, with_idx=True)
                motif_with_idx = smiles2mol(cur_mol_smiles_with_idx)
                _, ordermap = get_conn_list(motif_with_idx, use_Isotope=True)

                cyc_cand = [mapping[targ]]
                for cand in conn_list[1:]:
                    if ordermap[cand] != ordermap[targ]:
                        cyc_cand.append(mapping[cand])
                
                train_data_list.append(train_data(
                    graph = subgraph_data,
                    query_atom = mapping[query_atom],
                    cyclize_cand = cyc_cand,
                    label = (-1, 0),
                ))
     
            else:
                node = mol_graph.nodes[targ]['bpe_node']
                motif_idx = bpe_graph.nodes[node]['label']
                motifs_list.append(motif_idx)
                ordermap = bpe_graph.nodes[node]['ordermap']
                conn_idx = ordermap[targ]
                cyc_cand = [mapping[cand] for cand in conn_list[1:]]

                train_data_list.append(train_data(
                    graph = subgraph_data,
                    query_atom = mapping[query_atom],
                    cyclize_cand = cyc_cand,
                    label = (motif_idx, conn_idx),
                ))

                conn_list.extend(bpe_graph.nodes[node]['conn_list'])
                subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

            anchor1 = mol_graph.nodes[query_atom]['anchor']
            anchor2 = mol_graph.nodes[targ]['anchor']
            subgraph.add_edge(anchor1, anchor2)
            subgraph[anchor1][anchor2]['bondtype'] = mol_graph[anchor1][anchor2]['bondtype']
            subgraph[anchor1][anchor2]['label'] = mol_graph[anchor1][anchor2]['label']
            subgraph.remove_node(query_atom)
            subgraph.remove_node(targ)
            conn_list.remove(query_atom)
            conn_list.remove(targ)

        props = self.get_props()
        motifs_list = list(set(motifs_list))
        return mol_train_data(
            mol_graph = init_mol_graph_data,
            props = props,
            start_label = start_label,
            train_data_list = train_data_list,
            motif_list = motifs_list,
        )        

    @staticmethod
    def preprocess_vocab() -> Batch:
        vocab_data = []
        for idx in tqdm(range(len(MolGraph.MOTIF_LIST))):
            graph, _, _ = MolGraph.motif_to_graph(MolGraph.MOTIF_LIST[idx])
            data, _ = networkx2data(graph)
            vocab_data.append(data)
        vocab_data = Batch.from_data_list(vocab_data)
        return vocab_data

    @staticmethod
    def get_atom_features(atom: Chem.rdchem.Atom=None, IsConn: bool=False, BondType: Chem.rdchem.BondType=None) -> Tuple[int, int, int, int, int]:
        if IsConn:
            Symbol, FormalCharge, NumExplicitHs, NumImplicitHs = 0, 0, 0, 0       
            IsAromatic = True if BondType == Chem.rdchem.BondType.AROMATIC else False
            IsAromatic = ATOM_ISAROMATIC_VOCAB[IsAromatic]
        else:
            Symbol = ATOM_SYMBOL_VOCAB[atom.GetSymbol()]
            IsAromatic = ATOM_ISAROMATIC_VOCAB[atom.GetIsAromatic()]
            FormalCharge = ATOM_FORMALCHARGE_VOCAB[atom.GetFormalCharge()]
            NumExplicitHs = ATOM_NUMEXPLICITHS_VOCAB[atom.GetNumExplicitHs()]
            NumImplicitHs = ATOM_NUMIMPLICITHS_VOCAB[atom.GetNumImplicitHs()]
        return (Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs)

    @staticmethod
    def motif_to_graph(smiles: str, motif_list: Optional[List[str]] = None) -> Tuple[nx.Graph, List[int], List[int]]:
        motif = smiles2mol(smiles)
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(motif))
        
        dummy_list = []
        for atom in motif.GetAtoms():
            idx = atom.GetIdx()
            graph.nodes[idx]['smarts'] = atom.GetSmarts()
            graph.nodes[idx]['motif'] = smiles
            if atom.GetSymbol() == '*':
                graph.nodes[idx]['dummy_bond_type'] = bondtype = atom.GetBonds()[0].GetBondType()
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                dummy_list.append(idx)
            else:
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)
        
        ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False))
        dummy_list = list(zip(dummy_list, [ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*']))
        if len(dummy_list) > 0:
            dummy_list.sort(key=lambda x: x[1])
            dummy_list, _ = zip(*dummy_list)       

        for bond in motif.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        return graph, list(dummy_list), ranks