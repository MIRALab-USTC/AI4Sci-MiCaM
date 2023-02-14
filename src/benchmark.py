import random

import torch
from guacamol.assess_distribution_learning import assess_distribution_learning

from arguments import parse_arguments
from model.benchmarks import QuickBenchGenerator
from model.MiCaM_VAE import MiCaM
from model.mydataclass import ModelParams, Paths

if __name__ == '__main__':
    
    args = parse_arguments()
    paths = Paths(args)
    model_params = ModelParams(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.cuda)

    
    generator = MiCaM.load_generator(model_params, paths)
    generator = QuickBenchGenerator(generator, number_samples=args.num_sample)
    
    with open(paths.generate_path, "w") as f:
        for smi in generator.molecules:
            f.write(f'{smi}\n')

    assess_distribution_learning(
        generator,
        chembl_training_file=paths.train_path,
        json_output_file=paths.benchmark_path,
    )
    