import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from torch_geometric.data import Data
import rdkit.Chem.Draw as Draw
from PIL import Image
from guacamol.utils.chemistry import canonicalize
from tensorboardX import SummaryWriter
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA

EPSILON = 1e-7
INVALID = 0

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
    r"""Converts a :obj:`networkx.Graph`  to a
    :class:`torch_geometric.data.Data` instance, and the index mapping.
    """

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
    return regen_smiles(smiles) if postprocessing else smiles
 
def regen_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        global INVALID
        INVALID += 1
        print(f"invalid {INVALID}: {smiles}")
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
    '''
    Input a motif with connection sites, return the list of connection sites, and the ordermap.
    If with_idx==True: return the Isotope numbers, i.e., the indices in the full molecules.
    If with_idx==False: return the atom indices, i.e., the indices in the motifs.
    If symm==True: considering the symmetry issue.
    '''
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
    '''
    label the attachment atoms with their order as isotope (considering the symmetry issue)
    '''
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



def get_rec_acc(smiles: List[str], gen_smiles: List[str]) -> float:
    num = len(smiles)
    return sum([smiles[i] == gen_smiles[i] for i in range(num)]) / num

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

def get_accuracy_bin(scores: torch.Tensor, labels: torch.Tensor):
    preds = torch.ge(scores, 0).long()
    acc = torch.eq(preds, labels).float()
    
    tp = (labels * preds).sum().to(torch.float32)
    fp = ((1 - labels) * preds).sum().to(torch.float32)
    fn = (labels * (1 - preds)).sum().to(torch.float32)    
    
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)
    
    return torch.sum(acc) / labels.nelement(), f1, precision, recall

# def graph2motif(fragment_graph: nx.Graph, with_idx: bool=False) -> Chem.rdchem.Mol:
#     motif = Chem.RWMol()
#     node2idx = {}
#     for node in fragment_graph.nodes:
#         idx = motif.AddAtom(Chem.Atom(fragment_graph.nodes[node]['smarts']))
#         if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
#             motif.GetAtomWithIdx(idx).SetIsotope(node)
#         node2idx[node] = idx
#     for node1, node2 in fragment_graph.edges:
#         motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
#     motif_smiles = Chem.MolToSmiles(motif)
#     return Chem.MolFromSmiles(motif_smiles)


def motif_no_dummy(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != '*']
    if len(atom_indices) == 1:
        atom = mol.GetAtomWithIdx(atom_indices[0])
        return atom.GetSmarts()
    smi = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False))

def fragment_to_idx(fragment_smiles):
    for atom in Chem.MolFromSmiles(fragment_smiles, sanitize=False).GetAtoms():
        if atom.GetSymbol() == '*': return atom.GetIsotope()

def get_dummy_atoms(mol_graph: nx.Graph) -> List[Tuple[int, int]]:
    return [node for node in mol_graph.nodes if mol_graph.nodes[node]['smarts'] == '*']

def det_dummy_atoms_from_smiles(smiles: str) -> List[int]:
    mol = Chem.MolFromSmiles(smiles)
    return [atom.GetIdx() for atom in mol.GetAtoms()]






def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def fragment_no_label(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    return Chem.MolToSmiles(mol)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def get_accuracy_sym(scores, labels):
    max_scores,max_idx = torch.max(scores, dim=-1)
    lab_scores = scores[torch.arange(len(scores)), labels]
    acc = torch.eq(lab_scores, max_scores).float()
    return torch.sum(acc) / labels.nelement()

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)

def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)

def zip_tensors(tup_list):
    res = []
    tup_list = zip(*tup_list)
    for a in tup_list:
        if type(a[0]) is int: 
            res.append( torch.LongTensor(a).cuda() )
        else:
            res.append( torch.stack(a, dim=0) )
    return res

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


class RDFilter:
    def __init__(self, RULES_FILENAME, ALERT_FILENAME):
        with open(RULES_FILENAME) as json_file:
            self.rule_dict = json.load(json_file)

        rule_list = [
            x.replace("Rule_", "")
            for x in self.rule_dict.keys()
            if x.startswith("Rule") and self.rule_dict[x]
        ]

        rule_df = pd.read_csv(ALERT_FILENAME).dropna()
        rule_df = rule_df[rule_df.rule_set_name.isin(rule_list)]

        self.rule_list = []
        tmp_rule_list = rule_df[["rule_id", "smarts", "max", "description"]].values.tolist()
        for rule_id, smarts, max_val, desc in tmp_rule_list:
            smarts_mol = Chem.MolFromSmarts(smarts)
            self.rule_list.append([smarts_mol, max_val, desc])

    def __call__(self, smiles: str, log: bool=False):
        mol = Chem.MolFromSmiles(smiles)
        if not (self.rule_dict["MW"][0] <= MolWt(mol) <= self.rule_dict["MW"][1]):
            return False

        if not (self.rule_dict["LogP"][0] <= MolLogP(mol) <= self.rule_dict["LogP"][1]):
            return False

        if not (self.rule_dict["HBD"][0] <= NumHDonors(mol) <= self.rule_dict["HBD"][1]):
            return False

        if not (self.rule_dict["HBA"][0] <= NumHAcceptors(mol) <= self.rule_dict["HBA"][1]):
            return False

        if not (self.rule_dict["TPSA"][0] <= TPSA(mol) <= self.rule_dict["TPSA"][1]):
            return False

        for row in self.rule_list:
            patt, max_val, desc = row
            if len(mol.GetSubstructMatches(patt)) > max_val:
                if log: print(f"Pattern: {Chem.MolToSmiles(patt)}")
                return False

        return True










####################

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def is_aromatic_ring(mol):
    if mol.GetNumAtoms() == mol.GetNumBonds(): 
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False

def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

