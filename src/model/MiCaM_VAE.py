import os.path as path
from typing import List

import torch
import torch.nn as nn
from guacamol.distribution_matching_generator import \
    DistributionMatchingGenerator
from torch_geometric.data import Batch
from tqdm import tqdm

from model.benchmarks import QuickBenchmark
from model.decoder import Decoder
from model.encoder import Atom_Embedding, Encoder
from model.mol_graph import MolGraph
from model.mydataclass import (Decoder_Output, ModelParams, Paths, VAE_Output,
                               batch_train_data)
from model.nn import MLP, GIN_virtual


class MiCaM(nn.Module):

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


    def rsample(self, z: torch.Tensor, perturb: bool=True): 
        batch_size = len(z)
        z_mean = self.z_mean(z)
        z_log_var = torch.clamp_max(self.z_log_var(z), max=10)
        kl_loss = - 0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean, device=z.device)
        z = z_mean +  torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z, kl_loss

    def sample(self, num_sample: int=100, greedy: bool=True, beam_top: int=1):
        init_vecs = torch.randn(num_sample, self.latent_size).cuda()
        return self.decode(init_vecs, greedy=greedy, beam_top=beam_top)

    def decode(self, z: torch.Tensor, greedy: bool=True, beam_top: int=1, batch_size: int=1000):
        num_sample = len(z)
        num_batches = (num_sample - 1) // batch_size + 1
        batches = [z[i * batch_size: i * batch_size + batch_size] for i in range(num_batches)]
        results = []
        for batch in tqdm(batches):
            results.extend(self.decoder.decode(batch, greedy=greedy, max_decode_step=20, beam_top=beam_top))
        return results

    def benchmark(self, train_path: str):
        train_set = [smi.strip("\n") for smi in open(train_path)]
        benchmarks = QuickBenchmark(training_set=train_set, num_samples=10000)
        generator = GeneratorFromModel(self)
        return benchmarks.assess_model(generator)

    def save_motifs_embed(self, path):
        self.decoder.save_motifs_embed(path)
    
    def load_motifs_embed(self, path):
        self.decoder.load_motifs_embed(path)

    @staticmethod
    def load_model(model_params: ModelParams, paths: Paths):
        MolGraph.load_vocab(paths.vocab_path)
        model = MiCaM(model_params).cuda()

        model_path = path.join(paths.model_dir, "model.ckpt")
        motif_embed_path = path.join(paths.model_dir, "motifs_embed.ckpt")
        model.load_state_dict(torch.load(model_path)[0])
        model.load_motifs_embed(motif_embed_path)
        model.eval()
        return model

    @staticmethod
    def load_generator(model_params: ModelParams, paths: Paths):
        model = MiCaM.load_model(model_params, paths)
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

    def __init__(self, model: MiCaM):
        self.model = model
    
    def generate(self, number_samples: int) -> List[str]:
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(number_samples)
        return samples