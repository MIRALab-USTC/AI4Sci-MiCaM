# MiCaM: De Novo Molecular Generation via Connection-aware Motif Mining

This is the code of paper **De Novo Molecular Generation via Connection-aware Motif Mining**. *Zijie Geng, Shufang Xie, Yingce Xia, Lijun Wu, Tao Qin, Jie Wang, Yongdong Zhang, Feng Wu, Tie-Yan Liu.* ICLR 2023. [[arXiv](https://arxiv.org/pdf/2302.01129.pdf)]

## Environment

- Python 3.7
- Pytorch
- rdkit
- networkx
- torch-geometric
- guacamol

## Workflow

Put the dataset under the `./data` directory. Name the training set and avlid set as `train.smiles` and `valid.smiles`, respectively. An example of the working directory is as following.
```
AI4Sci-MiCaM
├── data
│   └── QM9
│       ├── train.smiles
│       └── valid.smiles
├── output/
├── preprocess/
├── src/
└── README.md
```

### 1. Mining connection-aware motifs

It consists of two phases: merging operation learning and motif vocabulary construction.

For merging operation learning, run the commands in form of

```
python src/merging_operation_learning.py \
    --dataset QM9 \
    --num_workers 60
```

For motif vocabulary constraction, run the commands in form of

```
python src/motif_vocab_construction.py \
    --dataset QM9 \
    --num_operations 1000 \
    --num_workers 60
```

### 2. Preprocess

To generate training data, using a given motif vocabulary, run the commands in form of

```
python src/make_training_data.py \
    --dataset QM9 \
    --num_operations 1000 \
    --num_workers 60
```

Alternatively, to run the entire preprocessing workflow, which includes mining motifs and generating training data, just run the commands in form of

```
python src/preprocess.py \
    --dataset QM9 \
    --num_operations 1000 \
    --num_workers 60
```

### 3. Training **MiCaM**

To train the MiCaM model, run a command in form of

```
python src/train.py \
    --job_name train_micam \
    --dataset QM9 \
    --num_operations 1000 \
    --batch_size 2000 \
    --depth 15 \
    --motif_depth 6 \
    --latent_size 64 \
    --hidden_size 256 \
    --dropout 0.3 \
    --steps 30000 \
    --lr 0.005 \
    --lr_anneal_iter 50 \
    --lr_anneal_rate 0.99 \
    --beta_warmup 3000 \
    --beta_min 0.001 \
    --beta_max 0.3 \
    --beta_anneal_period 40000 \
    --prop_weight 0.2 \
    --cuda 0
```

Benchmarking will be automatically conduct during the training process.

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{
geng2023de,
title={De Novo Molecular Generation via Connection-aware Motif Mining},
author={Zijie Geng and Shufang Xie and Yingce Xia and Lijun Wu and Tao Qin and Jie Wang and Yongdong Zhang and Feng Wu and Tie-Yan Liu},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=Q_Jexl8-qDi}
}
```




