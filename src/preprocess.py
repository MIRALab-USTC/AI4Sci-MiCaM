import os.path as path

import torch

from arguments import parse_arguments
from make_training_data import make_trainig_data
from merging_operation_learning import merging_operation_learning
from model.mol_graph import MolGraph
from model.mydataclass import Paths
from motif_vocab_construction import motif_vocab_construction

if __name__ == "__main__":

    args = parse_arguments()
    paths = Paths(args)

    if not path.exists(paths.operation_path):
        learning_trace = merging_operation_learning(
            train_path = paths.train_path,
            operation_path = paths.operation_path,
            num_iters = args.num_iters,
            min_frequency = args.min_frequency,
            num_workers = args.num_workers,
            mp_threshold = args.mp_thd,
        )

    MolGraph.load_operations(paths.operation_path, args.num_operations)

    if not path.exists(paths.vocab_path):
        mols, vocab = motif_vocab_construction(
            train_path = paths.train_path,
            vocab_path = paths.vocab_path,
            operation_path = paths.operation_path,
            num_operations = args.num_operations,
            mols_pkl_dir = paths.mols_pkl_dir,
            num_workers = args.num_workers,
        )
    
    MolGraph.load_vocab(paths.vocab_path)
    
    torch.multiprocessing.set_sharing_strategy("file_system")
    make_trainig_data(
        mols_pkl_dir = paths.mols_pkl_dir,
        valid_path = paths.valid_path,
        vocab_path = paths.vocab_path,
        train_processed_dir = paths.train_processed_dir,
        valid_processed_dir = paths.valid_processed_dir,
        vocab_processed_path = paths.vocab_processed_path,
        num_workers = args.num_workers,
    )
