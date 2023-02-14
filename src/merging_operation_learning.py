import multiprocessing as mp
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Dict, List, Tuple

import networkx as nx
from rdkit import Chem

from arguments import parse_arguments
from model.mydataclass import Paths


@dataclass
class MolGraph:
    idx: int
    mol_graph: Chem.rdchem.Mol
    merging_graph: nx.Graph

    def __init__(self, smiles: str, idx: int=0) -> "MolGraph":
        self.idx = idx
        self.mol_graph = Chem.MolFromSmiles(smiles)
        self.merging_graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol_graph))
        for atom in self.mol_graph.GetAtoms():
            self.merging_graph.nodes[atom.GetIdx()]["atom_indices"] = set([atom.GetIdx()])
    
    def apply_merging_operation(self, motif: str, stats: Dict[str, int], indices: Dict[str, Dict[int, int]]) -> None:
        if self.merging_graph.number_of_nodes() == 1:
            return
        new_graph = self.merging_graph.copy()
        for (node1, node2) in self.merging_graph.edges:
            if not new_graph.has_edge(node1, node2):
                continue
            atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[node2]["atom_indices"])
            motif_smiles = fragment2smiles(self, atom_indices)
            if motif_smiles == motif:
                graph_before_merge = new_graph.copy()
                merge_nodes(new_graph, node1, node2)
                update_stats(self, graph_before_merge, new_graph, node1, node2, stats, indices, self.idx)
        self.merging_graph = new_graph
        indices[motif][self.idx] = 0
    
    def apply_merging_operation_producer(self, motif: str, q: Queue) -> None:
        if self.merging_graph.number_of_nodes() == 1:
            return
        new_graph = self.merging_graph.copy()
        for (node1, node2) in self.merging_graph.edges:
            if not new_graph.has_edge(node1, node2):
                continue
            atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[node2]["atom_indices"])
            motif_smiles = fragment2smiles(self, atom_indices)
            if motif_smiles == motif:
                graph_before_merge = new_graph.copy()
                merge_nodes(new_graph, node1, node2)
                update_stats_producer(self, graph_before_merge, new_graph, node1, node2, q, self.idx)
        q.put((motif, self.idx, new_graph))

def load_batch_mols(batch: List[Tuple[int, str]]) -> List[MolGraph]:
    return [MolGraph(smi, idx) for (idx, smi) in batch]

def load_mols(train_path: str, num_workers: int) -> List[MolGraph]:
    print(f"[{datetime.now()}] Loading molecules...")
    smiles_list = [smi.strip("\n") for smi in open(train_path)]
    smiles_list = [(i, smi) for (i, smi) in enumerate(smiles_list)]
    
    batch_size = (len(smiles_list) - 1) // num_workers + 1
    batches = [smiles_list[i : i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    mols: List[MolGraph]= []
    with mp.Pool(num_workers) as pool:
        for mols_batch in pool.imap(load_batch_mols, batches):
            mols.extend(mols_batch)

    print(f"[{datetime.now()}] Loading molecules finished. Total: {len(mols)} molecules.\n")
    return mols

def fragment2smiles(mol: MolGraph, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol.mol_graph, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

def merge_nodes(graph: nx.Graph, node1: int, node2: int) -> None:
    neighbors = [n for n in graph.neighbors(node2)]
    atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[node2]["atom_indices"])
    for n in neighbors:
        if node1 != n and not graph.has_edge(node1, n):
            graph.add_edge(node1, n)
        graph.remove_edge(node2, n)
    graph.remove_node(node2)
    graph.nodes[node1]["atom_indices"] = atom_indices

def get_stats_producer(batch: List[MolGraph], q: Queue):
    for mol in batch:
        for (node1, node2) in mol.merging_graph.edges:
            atom_indices = mol.merging_graph.nodes[node1]["atom_indices"].union(mol.merging_graph.nodes[node2]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((mol.idx, motif_smiles))
    q.put(None)

def get_stats_consumer(stats: Dict[str, int], indices: Dict[str, Dict[int, int]], q: Queue, num_workers: int):
    num_tasks_done = 0
    while True:
        info = q.get()
        if info == None:
            num_tasks_done += 1
            if num_tasks_done == num_workers:
                break
        else:
            (idx, smi) = info
            stats[smi] += 1
            indices[smi][idx] += 1

def get_stats(mols: List[MolGraph], num_workers: int) -> Tuple[Dict[str, int], Dict[int, int]]:
    print(f"[{datetime.now()}] Begin getting statistics.")
    stats = defaultdict(int)
    indices = defaultdict(lambda: defaultdict(int))
    if num_workers == 1:
        for mol in mols:
            for (node1, node2) in mol.merging_graph.edges:
                atom_indices = mol.merging_graph.nodes[node1]["atom_indices"].union(mol.merging_graph.nodes[node2]["atom_indices"])
                motif_smiles = fragment2smiles(mol, atom_indices)
                stats[motif_smiles] += 1
                indices[motif_smiles][mol.idx] += 1
    else:
        batch_size = (len(mols) - 1) // num_workers + 1
        batches = [mols[i : i + batch_size] for i in range(0, len(mols), batch_size)]
        q = Queue()
        producers = [Process(target=get_stats_producer, args=(batches[i], q)) for i in range(num_workers)]
        [p.start() for p in producers]
        get_stats_consumer(stats, indices, q, num_workers)
        [p.join() for p in producers]
    return stats, indices

def update_stats(mol: MolGraph, graph: nx.Graph, new_graph: nx.Graph, node1: int, node2: int, stats: Dict[str, int], indices: Dict[str, Dict[int, int]], i: int):
    neighbors1 = [n for n in graph.neighbors(node1)]
    for n in neighbors1:
        if n != node2:
            atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            stats[motif_smiles] -= 1
            indices[motif_smiles][i] -= 1
    neighbors2 = [n for n in graph.neighbors(node2)]
    for n in neighbors2:
        if n != node1:
            atom_indices = graph.nodes[node2]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            stats[motif_smiles] -= 1
            indices[motif_smiles][i] -= 1
    neighbors = [n for n in new_graph.neighbors(node1)]
    for n in neighbors:
        atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[n]["atom_indices"])
        motif_smiles = fragment2smiles(mol, atom_indices)
        stats[motif_smiles] += 1
        indices[motif_smiles][i] += 1

def update_stats_producer(mol: MolGraph, graph: nx.Graph, new_graph: nx.Graph, node1: int, node2: int, q: Queue, i: int):
    neighbors1 = [n for n in graph.neighbors(node1)]
    for n in neighbors1:
        if n != node2:
            atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((motif_smiles, i, -1))
    neighbors2 = [n for n in graph.neighbors(node2)]
    for n in neighbors2:
        if n != node1:
            atom_indices = graph.nodes[node2]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((motif_smiles, i, -1))
    neighbors = [n for n in new_graph.neighbors(node1)]
    for n in neighbors:
        atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[n]["atom_indices"])
        motif_smiles = fragment2smiles(mol, atom_indices)
        q.put((motif_smiles, i, 1))

def apply_merging_operation_producer(motif: str, batch: List[MolGraph], q: Queue):
    [mol.apply_merging_operation_producer(motif, q) for mol in batch]
    q.put(None)

def apply_merging_operation_consumer(mols: List[MolGraph], stats: Dict[str, int], indices: Dict[str, Dict[int, int]], q: Queue, num_workers: int):
    num_tasks_done = 0
    while True:
        info = q.get()
        if info == None:
            num_tasks_done += 1
            if num_tasks_done == num_workers:
                break
        else:
            (motif, i, change) = info
            if isinstance(change, int):
                stats[motif] += change
                indices[motif][i] += change
            else:
                assert isinstance(change, nx.Graph)
                indices[motif][i] = 0
                mols[i].merging_graph = change
                
def apply_merging_operation(
    motif: str,
    mols: List[MolGraph],
    stats: Dict[str, int],
    indices: Dict[str, Dict[int, int]],
    num_workers: int = 1,
):
    mols_to_process = [mols[i] for i, freq in indices[motif].items() if freq > 0]
    if num_workers > 1:
        batch_size = (len(mols_to_process) -1 ) // num_workers + 1
        batches = [mols_to_process[i : i + batch_size] for i in range(0, len(mols_to_process), batch_size)]
        q = Queue()
        producers = [Process(target=apply_merging_operation_producer, args=(motif, batches[i], q)) for i in range(num_workers)]
        [p.start() for p in producers]
        apply_merging_operation_consumer(mols, stats, indices, q, num_workers)
        [p.join() for p in producers]
    else:
        [mol.apply_merging_operation(motif, stats, indices) for mol in mols_to_process]
    stats[motif] = 0

def merging_operation_learning(
    train_path: str,
    operation_path: str,
    num_iters: int,
    min_frequency: int,
    num_workers: int,
    mp_threshold: int,
):

    print(f"[{datetime.now()}] Learning merging operations from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.\n")

    mols = load_mols(train_path, num_workers)
    stats, indices = get_stats(mols, num_workers)

    trace = []
    dir = os.path.split(operation_path)[0]
    os.makedirs(dir, exist_ok=True)
    output = open(operation_path, "w")
    for i in range(num_iters):
        print(f"[{datetime.now()}] Iteration {i}.")
        motif = max(stats, key=lambda x: (stats[x], x))
        if stats[motif] < min_frequency:
            print(f"No motif has frequency >= {min_frequency}. Stopping.\n")
            break
        print(f"[Iteration {i}] Most frequent motif: {motif}, frequency: {stats[motif]}.\n")
        trace.append((motif, stats[motif]))
        
        apply_merging_operation(
            motif = motif,
            mols = mols,
            stats = stats,
            indices = indices,
            num_workers = num_workers if stats[motif] >= mp_threshold else 1,
        )

        output.write(f"{motif}\n")
    
    output.close()
    print(f"[{datetime.now()}] Merging operation learning finished.")
    print(f"The merging operations are in {operation_path}.\n\n")

    return trace

if __name__ == "__main__":

    args = parse_arguments()
    paths = Paths(args)

    learning_trace = merging_operation_learning(
        train_path = paths.train_path,
        operation_path = paths.operation_path,
        num_iters = args.num_iters,
        min_frequency = args.min_frequency,
        num_workers = args.num_workers,
        mp_threshold = args.mp_thd,
    )