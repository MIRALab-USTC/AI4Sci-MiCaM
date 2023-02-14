from typing import Dict, List, Tuple

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        return Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol

def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol()
    node2idx = {}
    for node in fragment_graph.nodes:
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node)
        node2idx[node] = idx
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
    return Chem.MolToSmiles(motif, allBondsExplicit=True)

def networkx2data(G: nx.Graph) -> Tuple[Data, Dict[int, int]]:
    num_nodes = G.number_of_nodes()
    mapping = dict(zip(G.nodes(), range(num_nodes)))
    
    G = nx.relabel_nodes(G, mapping)
    G = G.to_directed() if not nx.is_directed(G) else G

    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = torch.tensor([i for _, i in G.nodes(data='label')])
    edge_attr = torch.tensor([[i] for _, _, i in G.edges(data='label')], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, mapping

def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)

def mol_graph2smiles(graph: nx.Graph, postprocessing: bool=True) -> str:
    mol = Chem.RWMol()
    graph = nx.convert_node_labels_to_integers(graph)
    node2idx = {}
    for node in graph.nodes:
        idx = mol.AddAtom(smarts2atom(graph.nodes[node]['smarts']))
        node2idx[node] = idx
    for node1, node2 in graph.edges:
        mol.AddBond(node2idx[node1], node2idx[node2], graph[node1][node2]['bondtype'])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    return postprocess(smiles) if postprocessing else smiles
 
def postprocess(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and not atom.IsInRing():
                atom.SetIsAromatic(False)   
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        
        for _ in range(100):
            problems = Chem.DetectChemistryProblems(mol)
            flag = False
            for problem in problems:
                if problem.GetType() =='KekulizeException':
                    flag = True
                    for atom_idx in problem.GetAtomIndices():
                        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(False)
                    for bond in mol.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False)
            if flag: continue
            else: break
        
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"{smiles} not valid")
            return "CC"
        smi = Chem.MolToSmiles(mol)
        return smi

def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))
    if use_Isotope:
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))
    if not symm:
        conn_atoms = list(ordermap.keys())
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)
    return conn_atoms, ordermap


def label_attachment(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    dummy_atoms = [(atom.GetIdx(), ranks[atom.GetIdx()])for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    dummy_atoms.sort(key=lambda x: x[1])
    orders = []
    for (idx, order) in dummy_atoms:
        if order not in orders:
            orders.append(order)
            mol.GetAtomWithIdx(idx).SetIsotope(len(orders))
    return Chem.MolToSmiles(mol)

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()

    number, indices = torch.topk(scores, k=10, dim=-1)
    topk_acc = torch.eq(indices, labels.view(-1,1)).float()
    return torch.sum(acc) / labels.nelement(), torch.sum(topk_acc) / labels.nelement()

def sample_from_distribution(distribution: torch.Tensor, greedy: bool=False, topk: int=0):
    if greedy or topk == 1:
        motif_indices = torch.argmax(distribution, dim=-1)
    elif topk == 0 or len(torch.where(distribution > 0)) <= topk:
        motif_indices = torch.multinomial(distribution, 1)
    else:
        _, topk_idx = torch.topk(distribution, topk, dim=-1)
        mask = torch.zeros_like(distribution)
        ones = torch.ones_like(distribution)
        mask.scatter_(-1, topk_idx, ones)
        motif_indices = torch.multinomial(distribution * mask, 1)
    return motif_indices