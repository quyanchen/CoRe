from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from .chem_graph import FragmentGraphBank
from .cold_splits import build_cold_splits


@dataclass(frozen=True)
class DDIExample:
    drug_a: str
    relation: int
    drug_b: str


def read_smiles(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))


def read_triples(path: str) -> List[DDIExample]:
    df = pd.read_csv(path)
    expected = {"d1", "type", "d2"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(expected)}")
    return [
        DDIExample(str(row.d1), int(row.type), str(row.d2))
        for row in df.itertuples(index=False)
    ]


def split_paths(dataset_root: str, fold: int, scenario: str = "s1", cold_seed: int = 42) -> Tuple[str, str, str]:
    scenario = str(scenario).lower().strip()
    if scenario in {"s2", "s3"}:
        fold_dir = os.path.join(dataset_root, f"cold_{scenario}_seed{int(cold_seed)}", str(fold))
        return (
            os.path.join(fold_dir, "ddi_training1.csv"),
            os.path.join(fold_dir, "ddi_validation1.csv"),
            os.path.join(fold_dir, "ddi_test1.csv"),
        )
    if scenario != "s1":
        raise ValueError(f"Unsupported scenario: {scenario}")

    direct_paths = tuple(os.path.join(dataset_root, name) for name in ("train.csv", "valid.csv", "test.csv"))
    if all(os.path.exists(path) for path in direct_paths):
        return direct_paths

    fold_dir = os.path.join(dataset_root, str(fold))
    if "Deng" in os.path.basename(os.path.normpath(dataset_root)):
        return (
            os.path.join(fold_dir, "ddi_training1xiao.csv"),
            os.path.join(fold_dir, "ddi_validation1xiao.csv"),
            os.path.join(fold_dir, "ddi_test1xiao.csv"),
        )
    return (
        os.path.join(fold_dir, "ddi_training1.csv"),
        os.path.join(fold_dir, "ddi_validation1.csv"),
        os.path.join(fold_dir, "ddi_test1.csv"),
    )


def infer_num_relations(*triple_files: str) -> int:
    max_rel = -1
    for file in triple_files:
        if os.path.exists(file):
            df = pd.read_csv(file, usecols=["type"])
            max_rel = max(max_rel, int(df["type"].max()))
    return max_rel + 1


class DrugPairDataset(Dataset):
    def __init__(self, triples: List[DDIExample], graph_bank: FragmentGraphBank):
        self.triples = triples
        self.graph_bank = graph_bank

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        ex = self.triples[idx]
        graph = self.graph_bank.pair_graph(ex.drug_a, ex.drug_b)
        return graph, ex.relation, ex.drug_a, ex.drug_b


def collate_drug_pairs(batch):
    graphs, rels, drug_a, drug_b = zip(*batch)
    return {
        "graph": Batch.from_data_list(list(graphs)),
        "relation": torch.tensor(rels, dtype=torch.long),
        "drug_a": list(drug_a),
        "drug_b": list(drug_b),
    }


def build_datasets(
    dataset_root: str,
    fold: int,
    n_bits: int = 2048,
    cache_dir: Optional[str] = None,
    scenario: str = "s1",
    cold_seed: int = 42,
    cold_val_ratio: float = 0.2,
):
    smiles = read_smiles(os.path.join(dataset_root, "drug_smiles.csv"))
    scenario = str(scenario).lower().strip()
    if scenario in {"s2", "s3"}:
        build_cold_splits(dataset_root, scenario, seed=cold_seed, val_ratio=cold_val_ratio)
    train_path, valid_path, test_path = split_paths(dataset_root, fold, scenario=scenario, cold_seed=cold_seed)
    cache_path = None
    if cache_dir:
        root = os.path.normpath(dataset_root)
        dataset_name = os.path.basename(root)
        if os.path.exists(os.path.join(root, "train.csv")):
            dataset_name = f"{os.path.basename(os.path.dirname(root))}_{dataset_name}"
        cache_path = os.path.join(cache_dir, f"{dataset_name}_drug_fragments_{n_bits}.pt")
    graph_bank = FragmentGraphBank(smiles, n_bits=n_bits, cache_path=cache_path)
    if cache_path:
        graph_bank.precompute_all()
    return (
        DrugPairDataset(read_triples(train_path), graph_bank),
        DrugPairDataset(read_triples(valid_path), graph_bank),
        DrugPairDataset(read_triples(test_path), graph_bank),
        infer_num_relations(train_path, valid_path, test_path),
    )
