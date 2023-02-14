import random
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from arguments import parse_arguments
from model.MiCaM_VAE import MiCaM
from model.mydataclass import ModelParams, Paths

if __name__ == '__main__':
    
    args = parse_arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    paths = Paths(args)
    model_params = ModelParams(args)
    tb = SummaryWriter(log_dir=paths.tensorboard_dir)
    
    generator = MiCaM.load_generator(model_params, paths)
    print(f"[{datetime.now()}] Begin generating...")
    samples = generator.generate(args.num_sample)
    print(f"[{datetime.now()}] End generating...")

    with open(paths.generate_path, "w") as f:
        [f.write(f"{smi}\n") for smi in samples]
    
    print(f"See {paths.generate_path} for the samples.")

