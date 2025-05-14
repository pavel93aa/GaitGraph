# GaitGraph: Psychological Trait Estimation from Skeleton Data

This project implements a deep learning model for estimating psychological traits (e.g., self-esteem) based on gait
analysis using the **PsyMo dataset**. The model, named `GaitGraph`, is built using **Graph Convolutional Networks (GCN)
** to analyze 2D skeleton sequences extracted from video recordings.

The system supports two evaluation protocols:

- **Run-level**: classification of each walking sequence independently.
- **Subject-level**: aggregation of predictions across all runs per subject to estimate overall psychological state.

---

## Goal

To build and train a graph-based model that can classify human gait into psychological categories using skeletal motion
data. This task is part of a broader effort in Ambient Assisted Living (AAL), where computer vision is used for
non-intrusive monitoring of physical and mental states.

---

## Dataset Description

We use the **PsyMo dataset** (Psychological traits from Motion), which contains:

- 312 participants
- 7 walking styles per participant
- 6 camera views
- Annotated with psychological scores including RSE_Score (Rosenberg Self-Esteem Score)

For this implementation:

- We use only **IDs 0–7 for training**
- And **IDs 8–9 for testing**

The full dataset includes:

- `metadata_raw_scores_v3.csv`: raw psychological scores.
- `semantic_data/skeletons/`: directory containing `.json` files with skeleton keypoints.

Each `.json` file contains a list of frames with 17 body joints in COCO format:

- `[x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]` — total of 51 values per frame.

---

## Model Overview

### Model Name: `GaitGraph`

A Graph Neural Network (GNN) based on GCN layers designed to process 2D skeleton data as a graph.

### Architecture:

- Input: 17 joints × 2 coordinates (x, y)
- GCNConv → ReLU
- GCNConv → ReLU
- Global Mean Pooling
- Linear classifier (output: Low / Normal / High)

> See `models/gaitgraph.py` for full architecture.

---

## Quick Start Guide

### You can run the full pipeline step-by-step using the following commands

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Train the model

```bash
python -m training.train
```

#### Evaluate at Run-Level

```bash
python -c "from training.evaluate import evaluate; from models.gaitgraph import GaitGraph; from utils.data_loader import
PsyMoDataset; from torch_geometric.data import DataLoader; loader = PsyMoDataset(); _, test_data =
loader.get_datasets(); evaluate(DataLoader(test_data, batch_size=1), GaitGraph())"
```

#### Evaluate at Subject-Level

```bash
python -c "from training.subject_level_eval import subject_level_evaluate; from models.gaitgraph import GaitGraph; from
utils.data_loader import PsyMoDataset; from torch_geometric.data import DataLoader; loader = PsyMoDataset(); _,
test_data = loader.get_datasets(); subject_level_evaluate(DataLoader(test_data, batch_size=1), GaitGraph())"
```