"""Connection Quary Variational AutoEncoder."""
import torch
import torch.nn as nn
from argparse import Namespace
from model.encoder import Atom_Embedding, Motif_Embedding, Encoder
from model.decoder import Decoder
from model.mol_graph import MolGraph
from model.benchmarks import QuickBenchmark
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from typing import Any, List, Tuple
from model.utils import networkx2data
from model.nn import GIN_virtual, MLP
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from model.mydataclass import ModelParams, Paths, batch_train_data, VAE_Output, Decoder_Output
from collections import defaultdict
import torch.multiprocessing as mp
import os.path as path
import numpy as np
from guacamol.goal_directed_generator import GoalDirectedGenerator
from typing import List, Optional
from guacamol.scoring_function import ScoringFunction
import random

class MiCaM(nn.Module):
    """
    [C]onnection [Q]uary VAE model.
    """
    def __init__(self, model_params: ModelParams):
        super(MiCaM, self).__init__()
        
        self.motif_vocab = MolGraph.MOTIF_VOCAB

        self.model_params = model_params
        self.atom_embed_size = model_params.atom_embed_size
        self.edge_embed_size = model_params.edge_embed_size
        self.motif_embed_size = model_params.motif_embed_size
        self.dropout = model_params.dropout
        self.virtual = model_params.virtual
        self.pooling = model_params.pooling
        self.hidden_size = model_params.hidden_size
        self.latent_size = model_params.latent_size
        self.depth = model_params.depth
        self.motif_depth = model_params.motif_depth
        
        self.atom_embedding = nn.Sequential(
            Atom_Embedding(self.atom_embed_size),
            nn.Dropout(self.dropout),
        )

        self.edge_embedding = nn.Sequential(
            nn.Embedding(4, self.edge_embed_size),
            nn.Dropout(self.dropout),
        )

        self.encoder_gnn = GIN_virtual(
            in_channels = sum(self.atom_embed_size),
            out_channels = self.hidden_size,
            hidden_channels = self.hidden_size,
            edge_dim = self.edge_embed_size,
            depth = self.depth,
            dropout = self.dropout,
            virtual = self.virtual,
            pooling = model_params.pooling,
        )
        self.decoder_gnn = GIN_virtual(
            in_channels = sum(self.atom_embed_size),
            out_channels = self.hidden_size,
            hidden_channels = self.hidden_size,
            edge_dim = self.edge_embed_size,
            depth = self.depth,
            dropout = self.dropout,
            virtual = self.virtual,
            pooling = model_params.pooling,
        )

        self.motif_graphs: Batch = torch.load(model_params.vocab_processed_path)
        self.motif_gnn = GIN_virtual(
            in_channels = sum(self.atom_embed_size),
            out_channels = self.hidden_size,
            hidden_channels = self.hidden_size,
            edge_dim = self.edge_embed_size,
            depth = self.motif_depth,
            dropout = self.dropout,
            virtual = self.virtual,
            pooling = model_params.pooling
        )
        
        self.encoder = Encoder(
            atom_embedding = self.atom_embedding,
            edge_embedding = self.edge_embedding,
            GNN = self.encoder_gnn,
        )
        self.decoder = Decoder(
            atom_embedding = self.atom_embedding,
            edge_embedding = self.edge_embedding,
            decoder_gnn = self.decoder_gnn,
            motif_gnn = self.motif_gnn,
            motif_vocab = self.motif_vocab,
            motif_graphs = self.motif_graphs,
            motif_embed_size = self.motif_embed_size,
            hidden_size = self.hidden_size,
            latent_size = self.latent_size,
            dropout = self.dropout,
        )

        self.prop_pred = MLP(
            in_channels = self.latent_size,
            hidden_channels = self.hidden_size,
            out_channels = model_params.num_props,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = self.dropout,
        )
        self.pred_loss = nn.MSELoss()

        self.z_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.z_log_var = nn.Linear(self.hidden_size, self.latent_size)


    def rsample(self, z: torch.Tensor, perturb: bool=True, alpha: float=1.0): 
        batch_size = len(z)
        z_mean = self.z_mean(z)
        z_log_var = torch.clamp_max(self.z_log_var(z), max=10)
        kl_loss = - 0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean, device=z.device)
        z = z_mean + alpha * torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z, kl_loss

    def sample(self, num_sample:int=100, greedy:bool=False, beam_top: int=10):
        init_vecs = torch.randn(num_sample, self.latent_size).cuda()
        return self.decode(init_vecs, greedy=greedy, beam_top=beam_top)

    def decode(self, z: torch.Tensor, greedy:bool=False, beam_top: int=10, batch_size: int=1000):
        num_sample = len(z)
        num_batches = (num_sample - 1) // batch_size + 1
        batches = [z[i * batch_size: i * batch_size + batch_size] for i in range(num_batches)]
        results = []
        for batch in tqdm(batches):
            results.extend(self.decoder.decode(batch, greedy=greedy, max_decode_step=20, beam_top=beam_top))
        return results

    def benchmark(self, train_path:str):
        train_set = [smi.strip("\n") for smi in open(train_path)]
        benchmarks = QuickBenchmark(training_set=train_set, num_samples=10000)
        generator = GeneratorFromModel(self)
        return benchmarks.assess_model(generator)

    def save_motifs_embed(self, path):
        self.decoder.save_motifs_embed(path)
    
    def load_motifs_embed(self, path, cuda_: bool=True):
        self.decoder.load_motifs_embed(path, cuda_=cuda_)

    @staticmethod
    def load_model(model_params: ModelParams, paths: Paths, load_log_model: bool=False):
        MolGraph.load_vocab(paths.vocab_path)
        model = MiCaM(model_params).cuda()

        model_path = "model_log.ckpt" if load_log_model else "model_best.ckpt"
        model_path = path.join(paths.model_dir, model_path)
        motif_embed_path = "motifs_embed_log.ckpt" if load_log_model else "motifs_embed_best.ckpt"
        motif_embed_path = path.join(paths.model_dir, motif_embed_path)
        model.load_state_dict(torch.load(model_path)[0])
        model.load_motifs_embed(motif_embed_path)
        model.eval()
        return model

    @staticmethod
    def load_generator(model_params: ModelParams, paths: Paths, load_log_model: bool=False):
        model = MiCaM.load_model(model_params, paths, load_log_model)
        return GeneratorFromModel(model)

    def forward(self,
        input: batch_train_data,
        beta: float,
        prop_weight: float,
        dev: bool=False
    ) -> VAE_Output:

        _, z = self.encoder(input.batch_mols_graphs)
        z, kl_div = self.rsample(z, perturb=False) if dev else self.rsample(z, perturb=True)
        
        pred = self.prop_pred(z)
        pred_loss = self.pred_loss(pred, input.batch_props)

        decoder_output: Decoder_Output = self.decoder(z, input, dev)

        return VAE_Output(
            total_loss = beta * kl_div + decoder_output.decoder_loss + prop_weight * pred_loss,
            kl_div = kl_div,
            decoder_loss = decoder_output.decoder_loss,
            start_loss = decoder_output.start_loss,
            query_loss = decoder_output.query_loss,
            start_acc = decoder_output.tart_acc,
            start_topk_acc = decoder_output.start_topk_acc,
            query_acc = decoder_output.query_acc,
            query_topk_acc = decoder_output.query_topk_acc,
            pred_loss = pred_loss,
        )

class GeneratorFromModel(DistributionMatchingGenerator):

    def __init__(self, model: MiCaM, greedy: bool=True, beam_size: int=0):
        self.model = model
        self.greedy = greedy
        self.beam_size = beam_size
    
    def generate(self, number_samples: int) -> List[str]:
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(number_samples, greedy=self.greedy, beam_top=self.beam_size)
        return samples


class emb2seq:
    def __init__(self, motif_vocab, motif_list, decode_func, greedy, max_decode_step, beam_top, trace):
        self.decode = decode_func
        self.greedy = greedy
        self.max_decode_step = max_decode_step
        self.beam_top = beam_top
        self.trace = trace
        self.motif_vocab = motif_vocab
        self.motif_list = motif_list
    
    def __call__(self, z):
        z = torch.tensor(z)
        MolGraph.MOTIF_VOCAB = self.motif_vocab
        MolGraph.MOTIF_LIST = self.motif_list
        return self.decode(z, greedy=self.greedy, max_decode_step=self.max_decode_step, beam_top=self.beam_top, trace=self.trace)

class Wrapped_VAE(nn.Module):
    def __init__(self, args, greedy: bool=False, beam_top: int=0, cuda_: bool=True, max_decode_step: int=20):
        super(Wrapped_VAE, self).__init__()
        MolGraph.load_operations(args.code_file, num_operations=500)
        MolGraph.load_vocab(args.vocab_file)
        model = MiCaM(args)
        model.prop_pred = MLP(
            in_channels = model.latent_size,
            hidden_channels = model.hidden_size,
            out_channels = 1,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = model.dropout,
        )
        self.cuda_ = cuda_
        self.model = model.cuda() if cuda_ else model

        self.model.load_state_dict(torch.load(path.join(args.model_save_dir, f'model.ckpt.{args.ckpt}'))[0])
        self.model.load_motifs_embed(path.join(args.model_save_dir, f'motifs_embed.ckpt.{args.ckpt}'), cuda_=cuda_)
        self.model.eval()

        self.greedy = greedy
        self.beam_top = beam_top
        self.max_decode_step = max_decode_step
    
    def seq_to_emb(self, smiles: List[str]):
        batch_size = 200
        num_batches = ( len(smiles)-1 ) // batch_size + 1
        batches = [smiles[i*batch_size: (i+1)*batch_size] for i in range(num_batches)]
        zz = []
        for batch in batches:
            graphs = [networkx2data(MolGraph(smi, evaluate=True).mol_graph)[0] for smi in batch]
            
            data = Batch.from_data_list(graphs).cuda() if self.cuda_ else Batch.from_data_list(graphs)
            _, z = self.model.encoder(data)
            z, _ = self.model.rsample(z, perturb=False)
            z = z.cpu().detach().numpy() if self.cuda_ else z.detach().numpy()
            zz.append(z)
        return np.concatenate(zz, axis=0)

    def emb_to_seq(self, z, trace: bool=False):
        if self.cuda_:
            batch_size = 200
            num_batches = ( len(z)-1 ) // batch_size + 1
            batches = [z[i*batch_size: (i+1)*batch_size] for i in range(num_batches)]
            smis = []
            for batch in batches:
                zz = torch.tensor(batch).cuda()
                smis.extend(self.model.decoder.decode(zz, greedy=self.greedy, max_decode_step=self.max_decode_step, beam_top=self.beam_top, trace=trace))
        else:
            batch_size = 200
            num_batches = ( len(z)-1 ) // batch_size + 1
            batches = [z[i*batch_size: (i+1)*batch_size] for i in range(num_batches)]
            smis = []
            with mp.Pool(mp.cpu_count()) as pool:
                smis = list(tqdm(pool.imap(emb2seq(motif_vocab=MolGraph.MOTIF_VOCAB, motif_list=MolGraph.MOTIF_LIST, \
                    decode_func=self.model.decoder.decode, greedy=self.greedy, max_decode_step=self.max_decode_step, beam_top=self.beam_top, trace=trace), \
                    batches), desc="Batch", total=num_batches))
            smis = [smi for batch in smis for smi in batch]
        return smis

class Goal_VAE(GoalDirectedGenerator):

    def __init__(self, dir):
        sols = []
        for i in [1, 2, 3, 4, 5, 6]:
            sol = [line.strip("\r\n").split() for line in open(path.join(dir, f"solutions_iter{i}.txt"))]
            sol = [(float(x[0]), x[1]) for x in sol]
            sols.extend(sol)
        sols = list(set(sols))
        sols.sort(key=lambda x: x[0],reverse=True)
        self.smis = [smi for (s, smi) in sols]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        
        return self.smis[:number_molecules]