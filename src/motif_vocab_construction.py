import multiprocessing as mp
import os
import os.path as path
import pickle
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple

from tqdm import tqdm

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths


def apply_operations(batch: List[Tuple[int, str]], mols_pkl_dir: str) -> Counter:
    vocab = Counter()
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            mol = MolGraph(smi, tokenizer="motif")
            with open(path.join(mols_pkl_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump(mol, f)
            vocab = vocab + Counter(mol.motifs)
            pbar.update()
    return vocab

def motif_vocab_construction(
    train_path: str,
    vocab_path: str,
    operation_path: str,
    num_operations: int,
    num_workers: int,
    mols_pkl_dir: str,
):

    print(f"[{datetime.now()}] Construcing motif vocabulary from {train_path}.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.")

    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(train_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f"Total: {len(data_set)} molecules.\n")

    print(f"Processing...")
    vocab = Counter()
    os.makedirs(mols_pkl_dir, exist_ok=True)
    MolGraph.load_operations(operation_path, num_operations)
    func = partial(apply_operations, mols_pkl_dir=mols_pkl_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(func, batches):
            vocab = vocab + batch_vocab

    atom_list = [x for (x, _) in vocab.keys() if x not in MolGraph.OPERATIONS]
    atom_list.sort()
    new_vocab = []
    full_list = atom_list + MolGraph.OPERATIONS
    for (x, y), value in vocab.items():
        assert x in full_list
        new_vocab.append((x, y, value))
        
    index_dict = dict(zip(full_list, range(len(full_list))))
    sorted_vocab = sorted(new_vocab, key=lambda x: index_dict[x[0]])
    with open(vocab_path, "w") as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")
    
    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
    print(f"The motif vocabulary is in {vocab_path}.\n\n")

if __name__ == "__main__":

    args = parse_arguments()
    paths = Paths(args)
    os.makedirs(paths.preprocess_dir, exist_ok=True)

    motif_vocab_construction(
        train_path = paths.train_path,
        vocab_path = paths.vocab_path,
        operation_path = paths.operation_path,
        num_operations = args.num_operations,
        mols_pkl_dir = paths.mols_pkl_dir,
        num_workers = args.num_workers,
    )

    
    
    