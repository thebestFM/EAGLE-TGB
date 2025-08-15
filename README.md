# EAGLE-TGB
This repository contains our implementation and commands for reproducing the results reported in the TGB leaderboard. Thank you so much for your checking!

---

## Link Prediction

```bash
cd link_prediction
nohup python train.py --dataset_name tgbl-wiki --load_best_params --gpu 0 > output_wiki.log 2>&1 &
nohup python train.py --dataset_name tgbl-review --load_best_params --gpu 1 > output_review.log 2>&1 &
nohup python train.py --dataset_name tgbl-coin --load_best_params --gpu 2 > output_coin.log 2>&1 &
nohup python train.py --dataset_name tgbl-comment --load_best_params --gpu 3 > output_comment.log 2>&1 &
nohup python train.py --dataset_name tgbl-flight --load_best_params --gpu 4 > output_flight.log 2>&1 &
```

---

## Node Classification

```bash
cd node_classification
nohup python train.py --dataset_name tgbn-trade --load_best_params --gpu 0 > output_trade.log 2>&1 &
nohup python train.py --dataset_name tgbn-genre --load_best_params --gpu 1 > output_genre.log 2>&1 &
nohup python train.py --dataset_name tgbn-reddit --load_best_params --gpu 2 > output_reddit.log 2>&1 &
nohup python train.py --dataset_name tgbn-token --load_best_params --gpu 3 > output_token.log 2>&1 &
```

---

## Environment (for reference)

Hardware: 

```bash
NVIDIA A100-PCIE-40GB
```

Software: 

```bash
CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch_geometric==2.5.3
pip install matplotlib==3.8.4 networkx==3.2.1 numba==0.60.0 numpy==1.26.4 ogb==1.3.6 pandas==1.5.3 scikit-learn==1.5.2 scipy==1.13.1 py-tgb==2.0.0
```

---

## Contact

If you have any questions or need additional details, please feel free to reach out. ^_^

```bash
Yuming Xu
martin.xu@connect.polyu.hk
```

---
