"""For molecular graph processing."""
from ast import Global
import torch
import rdkit.Chem as Chem
import networkx as nx
from model.utils import graph2smiles, smiles2mol, get_conn_list, networkx2data, fragment2smiles
from merging_operation_learning import merge_nodes
from model.vocab import Vocab, MotifVocab, SubMotifVocab
from torch_geometric.data import Data, Batch
from model.mydataclass import train_data, mol_train_data, batch_train_data
from typing import Tuple, List, Dict, Set, Any, Optional, Union
from tqdm import tqdm
from rdkit.Chem import Descriptors
import os, sys
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
NUM_EDGE_TYPES = len(BOND_LIST)
MAX_NUM_ATOMS = 100

## guacamol
LOGP_MEAN, LOGP_VAR = 3.481587226600002, 1.8185146774225027
MOLWT_MEAN, MOLWT_VAR = 396.7136355500001, 110.55283206754517
QED_MEAN, QED_VAR = 0.5533041888502863, 0.21397359224960685
SA_MEAN, SA_VAR = 2.8882909807901354, 0.8059540682960904

class MolGraph(object):
    """
    To build a molecular graph.
    """

    @classmethod
    def load_operations(cls, operation_path: str, num_operations: int=500):
        """
        Load merging operations from a file.
        """
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
            self.bpe_graph = self.get_bpe_graph()
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
    
    def get_bpe_graph(self) -> nx.Graph:
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        bpe_graph = mol_graph.copy()
        for code in self.OPERATIONS:
            for (node1, node2) in mol_graph.edges:
                if not bpe_graph.has_edge(node1, node2):
                    continue
                atom_indices = bpe_graph.nodes[node1]['atom_indices'].union(bpe_graph.nodes[node2]['atom_indices'])
                pattern = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
                if pattern == code:
                    merge_nodes(bpe_graph, node1, node2)
            mol_graph = bpe_graph.copy()
        return nx.convert_node_labels_to_integers(bpe_graph)

    def refragment(self) -> None:
        '''
        In this function, the molecules are refragmented to motifs with dummy atoms.
        '''
        mol_graph = self.mol_graph.copy()
        bpe_graph = self.bpe_graph

        for node in bpe_graph.nodes:
            atom_indices = self.bpe_graph.nodes[node]['atom_indices']
            bpe_graph.nodes[node]['motif_no_conn'] = fragment2smiles(self.mol, atom_indices)
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
                bpe_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)
                
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node2][conn2]['label'] = mol_graph[node1][node2]['label']
                bpe_graph.nodes[bpe_node2]['atom_indices'].add(conn2)
                mol_graph.nodes[conn2]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

        for node in bpe_graph.nodes:
            atom_indices = bpe_graph.nodes[node]['atom_indices']
            motif_graph = mol_graph.subgraph(atom_indices)
            bpe_graph.nodes[node]['motif'] = graph2smiles(motif_graph)

        self.mol_graph = mol_graph

    def get_motifs(self) -> Set[str]:
        '''
        This function is applied when using a bpe code to learn the motif vocabulary.
        '''
        return [(self.bpe_graph.nodes[node]['motif_no_conn'], self.bpe_graph.nodes[node]['motif']) for node in self.bpe_graph.nodes]

    def relabel(self):
        '''
        Relabel the graphs for further preprocessing.
        '''
        mol_graph = self.mol_graph
        bpe_graph = self.bpe_graph

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
            self.bpe_graph[node1][node2]['label'] = 0

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
        init_mol_graph, mol_graph, bpe_graph = self.init_mol_graph, self.mol_graph, self.bpe_graph
        init_mol_graph_data, _ = networkx2data(init_mol_graph)
        motifs_list, conn_list = [], []
        train_data_list: List[train_data] = []

        nodes_num_atoms = dict(bpe_graph.nodes(data='num_atoms'))
        node = max(nodes_num_atoms, key=nodes_num_atoms.__getitem__)
        start_label = bpe_graph.nodes[node]['label']
        motifs_list.append(start_label)

        conn_list.extend(self.bpe_graph.nodes[node]['conn_list'])
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
    def preprocess(batch: List[str], raw_dir: str) -> train_data:
        if isinstance(mol, str):
            mol = MolGraph(mol, tokenizer="motif")
        return mol.get_data()

    @staticmethod
    def preprocess_batch(
        batch_data: Union[List[str], List['MolGraph']], 
        dev: bool = False,
        without_connection: bool = False,
    ) -> batch_train_data:
        '''
        This function is applied to preprocess a batch of smiles to get training data.
        '''   
        batch_mols_graphs: List[Data] = []
        batch_props: List[torch.Tensor] = []
        batch_start_labels: List[int] = []
        batch_train_data_list: List[List[train_data]] = []
        motif_lists: List[List[int]] = []

        if isinstance(batch_data[0], str):
            batch_smiles = batch_data
            batch_data = [MolGraph(smi, tokenizer="motif") for smi in batch_data]
        else:
            batch_smiles = [mol.smiles for mol in batch_data]

        for mol in batch_data:
            data = mol.get_data()
            batch_mols_graphs.append(data.mol_graph)
            batch_props.append(data.props)
            batch_start_labels.append(data.start_label)
            batch_train_data_list.append(data.train_data_list)
            motif_lists.append(data.motif_list)

        if dev:
            motifs_list = list(set(sum(motif_lists, [])))
            motif_vocab = MolGraph.MOTIF_VOCAB
            motif_conns_idx = motif_vocab.get_conns_idx()
            motif_conns_num = len(motif_conns_idx)
        else:
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
            batch_smiles = batch_smiles,
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
        '''
        Convert a motif index to the graph.
        '''
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

    @staticmethod
    def add_motif(
        graph: nx.Graph,
        motif_idx: int,
        cur_conn_list: List[int] = [],
        step_dict: dict = {},
        step: int = 0,
        query_idx: int = None,
        order_idx: int = None,
    ) -> Tuple[nx.Graph, Data, List[int], Dict[int, int], List[int]]:

        motif_graph, new_conn_list, ranks = MolGraph.motif_to_graph(motif_idx)
        if order_idx is not None:
            for i in range(len(motif_graph)):
                if ranks[i] == order_idx:
                    cand = i + len(graph)
                    break
        motif_graph = nx.convert_node_labels_to_integers(motif_graph, len(graph))
        
        for i, conn in enumerate(new_conn_list):
            new_conn_list[i] = conn = conn + len(graph)
            step_dict[conn] = step
        conn_list = cur_conn_list + new_conn_list
        graph = nx.union(graph, motif_graph)
        data, _ = networkx2data(graph)
        if order_idx is not None:
            graph, data, conn_list, step_dict =\
                MolGraph.merge_dummy_atoms(graph, query_idx, cand, conn_list, step_dict)
        
        return graph, data, conn_list, step_dict
    
    @staticmethod
    def merge_dummy_atoms(
        graph: nx.Graph,
        dummy: int,
        cand: int,
        dummy_list: List[int],
        step_dict: Dict[int, int],
    ) -> Tuple[nx.Graph, List[int]]:
        '''
        Merge two dummy atoms "dummy" and "cand" in the graph.
        :return: the new graph and the new list of dummy atoms in the graph.
        '''

        dummy_list.remove(dummy)
        dummy_list.remove(cand)
        step_dict.pop(dummy)
        step_dict.pop(cand)
        assert graph.nodes[dummy]['dummy_bond_type'] == graph.nodes[cand]['dummy_bond_type']
        bond_type = graph.nodes[dummy]['dummy_bond_type']
        node1 = list(graph.neighbors(dummy))[0]
        node2 = list(graph.neighbors(cand))[0]
        graph.add_edge(node1, node2)
        graph[node1][node2]['bondtype'] = bond_type
        graph[node1][node2]['label'] = BOND_VOCAB[bond_type]
        graph.remove_node(dummy)
        graph.remove_node(cand)

        mapping = dict(zip(graph.nodes(), range(0, len(graph))))
        graph = nx.relabel_nodes(graph, mapping)
        data, _ = networkx2data(graph)
        dummy_list = [mapping[i] for i in dummy_list]
        new_step_dict = dict()
        for key, value in step_dict.items():
            new_step_dict[mapping[key]] = value
        
        for i in dummy_list: assert i < len(graph)

        return graph, data, dummy_list, new_step_dict