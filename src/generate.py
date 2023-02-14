from arguments import parse_arguments
import torch, random
from src.model.MiCaM_VAE import MiCaM
from model.mol_graph import MolGraph
import sys
import os.path as path
import rdkit.Chem as Chem
from guacamol.utils.chemistry import is_valid
import torchvision
from rdkit.Chem import Draw
from PIL import Image
from tensorboardX import SummaryWriter

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from model.mydataclass import ModelParams, Paths

if __name__ == '__main__':
    
    args = parse_arguments()
    pathtool = Paths.from_arguments(args)
    model_params = ModelParams.from_arguments(args)
    tb = SummaryWriter(log_dir=pathtool.tensorboard_dir)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    generator = MiCaM.load_generator(model_params, pathtool)
    print(f"[{datetime.now()}] Begin generating...")
    samples = generator.generate(args.num_sample)
    print(f"[{datetime.now()}] End generating...")


    with open("acrylates.smiles", "w") as f:
        [f.write(f"{smi}\n") for smi in samples]

