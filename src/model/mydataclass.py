import logging
import os
import os.path as path
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from email.generator import Generator
from typing import Any, Dict, List, Optional, Tuple

from guacamol.distribution_learning_benchmark import \
    DistributionLearningBenchmarkResult
from tensorboardX import SummaryWriter
from torch import LongTensor, Tensor
from torch_geometric.data import Batch, Data


@dataclass
class train_data:
    graph: Data
    query_atom: int
    cyclize_cand: List[int]
    label: Tuple[int, int]

@dataclass
class mol_train_data:
    mol_graph: Data
    props: Tensor
    start_label: int
    train_data_list: List[train_data]
    motif_list: List[int]

@dataclass
class batch_train_data:
    batch_mols_graphs: Batch
    batch_props: Tensor
    batch_start_labels: LongTensor
    motifs_list: LongTensor
    batch_train_graphs: Batch
    mol_idx: LongTensor
    graph_idx: LongTensor
    query_idx: LongTensor
    cyclize_cand_idx: LongTensor
    motif_conns_idx: LongTensor
    labels: LongTensor

    def cuda(self):
        self.batch_mols_graphs = self.batch_mols_graphs.cuda()
        self.batch_props, self.batch_start_labels = self.batch_props.cuda(), self.batch_start_labels.cuda()
        self.batch_train_graphs = self.batch_train_graphs.cuda()
        self.mol_idx, self.graph_idx, self.query_idx = self.mol_idx.cuda(), self.graph_idx.cuda(), self.query_idx.cuda()
        self.cyclize_cand_idx, self.motif_conns_idx = self.cyclize_cand_idx.cuda(), self.motif_conns_idx.cuda()
        self.labels = self.labels.cuda()
        return self

@dataclass
class Paths:
    data_dir: str
    preprocess_dir: str
    output_dir: str
    model_save_dir: str
    train_path: str
    valid_path: str
    operation_path: str
    vocab_path: str
    mols_pkl_dir: str
    train_processed_dir: str
    valid_processed_dir: str
    vocab_processed_path: str
    generate_path: str
    job_name: str
    tensorboard_dir: str
    model_path: Optional[str] = None

    def __init__(self, args: Namespace):
        self.data_dir = args.data_dir
        self.preprocess_dir = args.preprocess_dir
        self.output_dir = args.output_dir

        self.data_dir = path.join(self.data_dir, args.dataset)
        assert path.exists(self.data_dir), print(f"Cannot find the dataset {self.data_dir}.")
        self.train_path = path.join(self.data_dir, "train.smiles")
        self.valid_path = path.join(self.data_dir, "valid.smiles")

        self.preprocess_dir = path.join(self.preprocess_dir, args.dataset)
       
        self.operation_path = path.join(self.preprocess_dir, "merging_operation.txt")
        self.preprocess_dir = path.join(self.preprocess_dir, f"num_ops_{args.num_operations}")
        self.vocab_path = path.join(self.preprocess_dir, "vocab.txt")
        self.mols_pkl_dir = path.join(self.preprocess_dir, "mol_graphs")
        self.train_processed_dir = path.join(self.preprocess_dir, "train")
        self.valid_processed_dir = path.join(self.preprocess_dir, "valid")
        self.vocab_processed_path = path.join(self.preprocess_dir, "vocab.pth")

        date_str = datetime.now().strftime("%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        self.job_name = time_str + "-" + args.job_name
        self.output_dir = path.join(self.output_dir, date_str, self.job_name)

        self.model_save_dir = path.join(self.output_dir, "ckpt")
        if args.model_dir is not None:
            self.model_dir = path.join(args.model_dir, "ckpt")
            self.generate_path = path.join(args.model_dir, args.generate_path+".smiles")
            self.benchmark_path = path.join(args.model_dir, args.generate_path+".json")
        
        self.tensorboard_dir = path.join(args.tensorboard_dir, date_str, self.job_name)

@dataclass
class ModelParams:
    atom_embed_size: List[int]
    edge_embed_size: int
    motif_embed_size: int
    hidden_size: int
    latent_size: int
    depth: int
    motif_depth: int
    virtual: bool
    pooling: str
    dropout: float
    num_props: int
    vocab_processed_path: str

    def __init__(self, args: Namespace):
        self.atom_embed_size = args.atom_embed_size
        self.edge_embed_size = args.edge_embed_size
        self.motif_embed_size = args.motif_embed_size
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.depth = args.depth
        self.motif_depth = args.motif_depth
        self.virtual = args.virtual
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.num_props = 4
        self.vocab_processed_path = path.join(args.preprocess_dir, args.dataset, f"num_ops_{args.num_operations}", "vocab.pth")

    def __repr__(self) -> str:
        return f"""
        Model Parameters:
        atom_embed_size         |       {self.atom_embed_size}
        edge_embed_size         |       {self.edge_embed_size}
        motif_embed_size        |       {self.motif_embed_size}
        hidden_size             |       {self.hidden_size}
        latent_size             |       {self.latent_size}
        depth                   |       {self.depth}
        motif_depth             |       {self.motif_depth}
        virtual_node            |       {self.virtual}
        pooling                 |       {self.pooling}
        dropout                 |       {self.dropout}
        """
    
@dataclass
class TrainingParams:
    lr: float
    lr_anneal_iter: int
    lr_anneal_rate: float
    grad_clip_norm: float
    batch_size: int
    steps: int
    beta_warmup: int
    beta_min: float
    beta_max: float
    beta_anneal_period: int
    prop_weight: float

    def __init__(self, args: Namespace):
        self.lr = args.lr
        self.lr_anneal_iter = args.lr_anneal_iter
        self.lr_anneal_rate = args.lr_anneal_rate
        self.grad_clip_norm = args.grad_clip_norm
        self.batch_size = args.batch_size
        self.steps = args.steps
        self.beta_warmup = args.beta_warmup
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.beta_anneal_period = args.beta_anneal_period
        self.prop_weight = args.prop_weight

    def __repr__(self) -> str:
        return f"""
        Training Parameters:
        lr                      |       {self.lr}
        lr_anneal_iter          |       {self.lr_anneal_iter}
        lr_anneal_rate          |       {self.lr_anneal_rate}
        grad_clip_norm          |       {self.grad_clip_norm}
        steps                   |       {self.steps}
        beta_warmup             |       {self.beta_warmup}
        beta_min                |       {self.beta_min}
        beta_max                |       {self.beta_max}
        beta_anneal_period      |       {self.beta_anneal_period}
        prop_weight             |       {self.prop_weight}
        """

@dataclass
class Decoder_Output:
    decoder_loss: Any = None
    start_loss: Any = None
    query_loss: Any = None
    tart_acc: Any = None
    start_topk_acc: Any = None
    query_acc: Any = None
    query_topk_acc: Any = None


@dataclass
class BenchmarkResults:
    validity: DistributionLearningBenchmarkResult
    uniqueness: DistributionLearningBenchmarkResult
    novelty: DistributionLearningBenchmarkResult
    kl_div: DistributionLearningBenchmarkResult
    fcd: DistributionLearningBenchmarkResult

    def __repr__(self):
        return f"""
        ==============================================================
        | Metrics | Validity | Uniqueness | Novelty | KL Div |  FCD  |
        --------------------------------------------------------------
        | Scores  |  {self.validity.score:.3f}   |   {self.uniqueness.score:.3f}    |  {self.novelty.score:.3f}  | {self.kl_div.score:.3f}  | {self.fcd.score:.3f} |
        ==============================================================
        """

@dataclass
class VAE_Output:
    total_loss: Tensor = None
    kl_div: Tensor = None
    decoder_loss: Tensor = None
    start_loss: Tensor = None
    query_loss: Tensor = None
    pred_loss: Tensor = None
    start_acc: float = None
    start_topk_acc: float = None
    query_acc: float = None
    query_topk_acc: float = None

    def print_results(self, total_step: int, lr:float, beta: float) -> None:
        logging.info(f"[Step {total_step:5d}] | Loss. KL: {self.kl_div:3.3f}, decoder_loss: {self.decoder_loss:3.3f}, pred_loss: {self.pred_loss:2.5f} \
| Start_acc. top1: {self.start_acc: .3f}, top10: {self.start_topk_acc:.3f} | Query_acc. top1: {self.query_acc:.3f}, top10: {self.query_topk_acc:.3f} \
| Params. lr: {lr:.6f}, beta: {beta:.6f}.")

    def log_tb_results(self, total_step: int, tb:SummaryWriter, beta, lr) -> None:
    
        tb.add_scalar(f"Loss/Total_Loss", self.total_loss, total_step)
        tb.add_scalar(f"Loss/Decoder_loss", self.decoder_loss, total_step)
        tb.add_scalar(f"Loss/KL_div", self.kl_div, total_step)
        tb.add_scalar(f"Loss/Start_loss", self.start_loss, total_step)
        tb.add_scalar(f"Loss/Query_loss", self.query_loss, total_step)
        tb.add_scalar(f"Loss/Prop_pred_loss", self.pred_loss, total_step)

        tb.add_scalar("Hyperparameters/beta", beta, total_step)
        tb.add_scalar("Hyperparameters/lr", lr, total_step)
       