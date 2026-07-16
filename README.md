# CoRe

**Learning Event-Conditioned Drug Roles via Reciprocal Motif–Context Refinement for Drug–Drug Interaction Prediction**

## Overview

CoRe consists of three main components:

1. **Pairwise fragment interaction encoding.** Each drug is decomposed into BRICS fragments represented by radius-2 Morgan fingerprints. Intra-drug edges preserve fragment topology, while cross-drug edges support bilateral information exchange.
2. **Reciprocal motif–context refinement.** Pair-context and drug-side summary tokens recalibrate motif representations, and the refined motifs are subsequently used to update the pair context.
3. **Event-conditioned bilateral role inference.** Each candidate DDI event independently retrieves motif-level evidence from the two drugs, constructs event- and partner-conditioned drug roles, and scores the resulting event–drug–drug triplet.

## Requirements

The implementation requires Python 3.11.

A minimal environment can be created as follows:

```bash
python -m pip install -r requirements.txt
```

## Repository Structure

```text
CoRe-main/
├── code/
│   ├── chem_graph.py      # BRICS fragmentation and pair-graph construction
│   ├── data.py            # Dataset loading and mini-batch collation
│   ├── model.py           # CoRe architecture and auxiliary objectives
│   ├── train.py           # Training and evaluation entry point
│   └── cold_splits.py     # S2/S3 cold-start split construction
└── README.md
```

## Data Preparation

CoRe supports the **Deng** and **Ryu** fine-grained DDI benchmarks. Each dataset directory must contain a headerless drug–SMILES file:

```text
drug_smiles.csv
```

Each row contains:

```text
drug_id,smiles
```

Interaction files must be CSV files with the following header:

```text
d1,type,d2
```

where `d1` and `d2` form an ordered drug pair and `type` is the integer event label.

For the standard S1 setting, organize each fold as follows:

```text
data/
├── Deng/
│   ├── drug_smiles.csv
│   ├── 0/
│   │   ├── ddi_training1xiao.csv
│   │   ├── ddi_validation1xiao.csv
│   │   └── ddi_test1xiao.csv
│   └── ...
└── Ryu/
    ├── drug_smiles.csv
    ├── 0/
    │   ├── ddi_training1.csv
    │   ├── ddi_validation1.csv
    │   └── ddi_test1.csv
    └── ...
```

Repeat the fold directory for folds `0`–`4`. For S2 and S3, the training script invokes the cold-split construction utility and stores the generated partitions under directories such as `cold_s2_seed42/` and `cold_s3_seed42/`.

## Training

Run CoRe from the repository root.

### Standard setting: S1

```bash
python -m code.train \
  --dataset-root data/Deng \
  --scenario s1 \
  --fold 0 \
  --device cuda
```

Replace `data/Deng` with `data/Ryu` to run experiments on Ryu.

To evaluate all five folds:

```bash
for fold in 0 1 2 3 4; do
  python -m code.train \
    --dataset-root data/Deng \
    --scenario s1 \
    --fold ${fold} \
    --device cuda
done
```

## Default Configuration

The released configuration follows the main experimental setup:

| Hyperparameter | Value |
|---|---:|
| Hidden dimension | 256 |
| Pairwise interaction layers | 3 |
| Reciprocal refinement layers | 3 |
| Attention heads | 4 |
| Dropout | 0.2 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Maximum epochs | 100 |
| Early-stopping patience | 30 |
| Class weighting | Inverse square root |

All major hyperparameters can be changed through command-line arguments. Run the following command for the complete option list:

```bash
python -m code.train --help
```

## Outputs

By default, training artifacts are written to:

```text
run/     # Logs and validation-selected checkpoints
cache/   # Persistent per-drug BRICS fragment graph caches
```

The training log reports Accuracy, Macro-F1, Macro-Recall, and Macro-Precision. Model selection is based on validation Macro-F1, and the final checkpoint name includes the test Macro-F1 score.
