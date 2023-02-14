import argparse
from typing import List


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--preprocess_dir', default='preprocess/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard/')

    parser.add_argument('--dataset', type=str, default="QM9")
    parser.add_argument('--job_name', type=str, default="")
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--generate_path', type=str, default="samples")

    # hyperparameters
    ## common
    parser.add_argument('--num_workers', type=int, default=60)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)

    ## merging operation learning, motif vocab construction
    parser.add_argument('--num_operations', type=int, default=500)
    parser.add_argument('--num_iters', type=int, default=3000)
    parser.add_argument('--min_frequency', type=int, default=0)
    parser.add_argument('--mp_thd', type=int, default=1e5)

    ## networks
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--atom_embed_size', type=List[int], default=[192, 16, 16, 16, 16])
    parser.add_argument('--edge_embed_size', type=int, default=256)
    parser.add_argument('--motif_embed_size', type=int, default=[256, 256])
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=15)
    parser.add_argument('--motif_depth', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--virtual', action='store_true')
    parser.add_argument('--pooling', type=str, default="add")

    ## training
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_anneal_iter', type=int, default=500)
    parser.add_argument('--lr_anneal_rate', type=float, default=0.99)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)

    parser.add_argument('--beta_warmup', type=int, default=3000)
    parser.add_argument('--beta_min', type=float, default=1e-3)
    parser.add_argument('--beta_max', type=float, default=0.6)
    parser.add_argument('--beta_anneal_period', type=int, default=20000)
    parser.add_argument('--prop_weight', type=float, default=0.5)
    
    # inference
    parser.add_argument('--num_sample', type=int, default=10000)
    
    args = parser.parse_args()
    return args