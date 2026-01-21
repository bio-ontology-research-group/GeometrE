# Fully Geometric Multi-Hop Reasoning on Knowledge Graphs With Transitive Relations

---

## Requirements

- Python 3.10.15
- PyTorch 2.4.1
- CUDA-capable GPU (recommended)

## Installation

Install all dependencies using conda:

```bash
conda env create -f environment.yml
conda activate geometre
```

---

## Data

Download the datasets:

- **WN18RR**: Download from https://github.com/nelsonhuangzijian/WN18RR-QA
- **NELL & FB15k-237**: Download from BetaE:
  ```bash
  wget http://snap.stanford.edu/betae/KG_data.zip
  unzip KG_data.zip
  ```

Place the data under `src/data/` so that the file structure looks like this:

```
.
├── environment.yml
├── README.md
├── requirements.txt
└── src
    ├── main.py
    ├── data
    │   ├── FB15k-237-betae
    │   ├── NELL-betae
    │   └── WN18RR-QA
    └── ...
```

---

## Scripts Overview

| Script | Description |
|--------|-------------|
| `main.py` | **Entry point for training and evaluation.** Parses command-line arguments, loads datasets, manages the training loop with checkpointing, runs evaluation (MRR, Hits@1/3/10), and integrates with Weights & Biases for experiment tracking. |
| `models.py` | **Core neural network model.** Adaptation of the `KGReasoning` class used in previous methods (BetaE, ConE, etc.) with entity/relation embeddings (centers and offsets), loss computation (positive/negative samples, membership, transitive regularization), and evaluation methods. |
| `embeddings.py` | **Query embedding functions.** Implements geometric construction of GeometrE embeddings for all query types. | 
| `box.py` | **Geometric box operations.** Defines the `Box` class with methods for intersection, transformation, projection, and scoring functions (inclusion, exclusion, order preservation). |
| `dataloader.py` | **Data handling.** Provides PyTorch Dataset classes for training and testing, implements negative sampling strategies, and manages data iteration. |
| `util.py` | **Utility functions.** Helper functions for data structure conversions, random seed setting, and mappings for transitive/inverse relations per dataset. |

---

## Hyperparameters

### Grid Search Ranges

The following hyperparameter ranges were used during grid search:

| Hyperparameter | Values |
|----------------|--------|
| `alpha` | 0, 0.1, 0.2, 0.5 |
| `gamma` | 10, 20, 40 |
| `hidden_dim` (embedding size) | 100, 200, 400 |
| `learning_rate` | 0.001, 0.0005, 0.0001 |
| `with_answer_embedding` | yes, no |

For the transitive loss function, we used `lambda = 0.1`.

### Optimal Hyperparameters per Dataset

| Hyperparameter | WN18RR | NELL | FB15k-237 |
|----------------|--------|------|-----------|
| `alpha` | 0.5 | 0.2 | 0.2 |
| `gamma` | 20 | 10 | 20 |
| `hidden_dim` | 400 | 400 | 400 |
| `learning_rate` | 0.001 | 0.0005 | 0.0005 |
| `with_answer_embedding` | yes | no | no |
| `transitive` | yes | no | no |
| `batch_size` | 1024 | 1024 | 1024 |
| `negative_sample_size` | 64 | 64 | 64 |

---

## Reproduce the Results

### Setup

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. We use [Weights and Biases](https://wandb.ai/) for experiment tracking. You need to provide your W&B username via the `--wandb_username` argument.

   **To disable W&B logging**, set the environment variable before running:
   ```bash
   export WANDB_MODE=disabled
   ```

### Run Experiments

**WN18RR:**
```bash
python main.py --do_train --do_test --alpha=0.5 --batch_size=1024 \
    --data_path=data/WN18RR-QA --gamma=20 --hidden_dim=400 \
    --learning_rate=0.001 --negative_sample_size=64 --transitive=yes \
    --with_answer_embedding -ns -desc "reproduce wn18rr" \
    --wandb_username your_username
```

**NELL:**
```bash
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 \
    --data_path=data/NELL-betae --gamma=10 --hidden_dim=400 \
    --learning_rate=0.0005 --negative_sample_size=64 --transitive=no \
    -ns -desc "reproduce nell" --wandb_username your_username
```

**FB15k-237:**
```bash
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 \
    --data_path=data/FB15k-237-betae --gamma=20 --hidden_dim=400 \
    --learning_rate=0.0005 --negative_sample_size=64 --transitive=no \
    -ns -desc "reproduce fb237" --wandb_username your_username
```

---

## Command-Line Arguments

### Main Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--do_train` | Enable training | - |
| `--do_test` | Enable evaluation on test set | - |
| `--do_valid` | Enable evaluation on validation set | - |
| `--data_path` | Path to the dataset | `data/WN18RR-QA` |
| `--wandb_username` | Your Weights & Biases username | - |
| `-desc`, `--description` | Description of the run (used as W&B run name) | `default` |

### Model Hyperparameters

| Argument | Description | Default |
|----------|-------------|---------|
| `-d`, `--hidden_dim` | Embedding dimension | 500 |
| `-g`, `--gamma` | Margin in the loss function | 12.0 |
| `-a`, `--alpha` | Balance between in-box and out-box distance | 0.5 |
| `-ts`, `--transitive` | Enable transitive relation handling (`yes`/`no`) | `no` |
| `--with_answer_embedding` | Use answer embeddings instead of box embeddings | - |

### Training Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `-b`, `--batch_size` | Batch size for training | 1024 |
| `-n`, `--negative_sample_size` | Number of negative samples per query | 128 |
| `-lr`, `--learning_rate` | Learning rate | 0.0001 |
| `--max_steps` | Maximum training iterations | 200001 |
| `--valid_steps` | Evaluate every N steps | 5000 |
| `--save_checkpoint_steps` | Save checkpoint every N steps | 5000 |

### Other Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-ns`, `--no_sweep` | Not a W&B sweep (logs hyperparams directly) | - |
| `--checkpoint_path` | Path to load a saved checkpoint | - |
| `--seed` | Random seed for reproducibility | 0 |
| `--print_on_screen` | Print logs to console | - |

---

## Output

- **Checkpoints** are saved to `logs/<dataset>/<tasks>/<model>/`
- **Logs** include training metrics and evaluation results (MRR, Hits@1, Hits@3, Hits@10)
- If W&B is enabled, metrics are also logged to your W&B dashboard
