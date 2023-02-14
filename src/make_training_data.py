import multiprocessing as mp
import os
import os.path as path
import pickle
from datetime import datetime
from functools import partial
from typing import List, Tuple

import torch
from tqdm import tqdm

from arguments import parse_arguments
from model.mol_graph import MolGraph
from model.mydataclass import Paths


def process_train_batch(batch: List[str], raw_dir: str, save_dir: str):
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for file in batch:
            with open(path.join(raw_dir, file), "rb") as f:
                mol: MolGraph = pickle.load(f)
            data = mol.get_data()
            torch.save(data, path.join(save_dir, file.split()[0]+".pth"))
            pbar.update()

def process_valid_batch(batch: List[Tuple[int, str]], save_dir: str):
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(batch), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for idx, smi in batch:
            mol = MolGraph(smi, tokenizer="motif")
            data = mol.get_data()
            torch.save(data, path.join(save_dir, f"{idx}.pth"))
            pbar.update()

def make_trainig_data(
    mols_pkl_dir: str,
    valid_path: str,
    vocab_path: str,
    train_processed_dir: str,
    valid_processed_dir: str,
    vocab_processed_path: str,
    num_workers: int,
):

    print(f"[{datetime.now()}] Preprocessing traing data.")
    print(f"Number of workers: {num_workers}. Total number of CPUs: {mp.cpu_count()}.\n")


    print(f"[{datetime.now()}] Loading training set from {mols_pkl_dir}.\n")
    os.makedirs(train_processed_dir, exist_ok=True)
    data_set = os.listdir(mols_pkl_dir)
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    func = partial(process_train_batch, raw_dir=mols_pkl_dir, save_dir=train_processed_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)
    
    
    print(f"[{datetime.now()}] Preprocessing valid set from {valid_path}.\n")
    os.makedirs(valid_processed_dir, exist_ok=True)
    data_set = [(idx, smi.strip("\n")) for idx, smi in enumerate(open(valid_path))]
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    func = partial(process_valid_batch, save_dir=valid_processed_dir)
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        pool.map(func, batches)

    print(f"[{datetime.now()}] Preprocessing motif vocabulary from {vocab_path}.\n")
    vocab_data = MolGraph.preprocess_vocab()
    with open(vocab_processed_path, "wb") as f:
        torch.save(vocab_data, f)

    print(f"[{datetime.now()}] Preprocessing finished.\n\n")

if __name__ == "__main__":

    args = parse_arguments()
    paths  = Paths(args)

    MolGraph.load_operations(paths.operation_path, args.num_operations)
    MolGraph.load_vocab(paths.vocab_path)

    make_trainig_data(
        mols_pkl_dir = paths.mols_pkl_dir,
        valid_path = paths.valid_path,
        vocab_path = paths.vocab_path,
        train_processed_dir = paths.train_processed_dir,
        valid_processed_dir = paths.valid_processed_dir,
        vocab_processed_path = paths.vocab_processed_path,
        num_workers = args.num_workers,
    )
