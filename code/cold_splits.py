from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


Triple = Tuple[str, int, str]


@dataclass
class ColdSplitStats:
    scenario: str
    seed: int
    fold: int
    unknown_drugs: int
    train: int
    valid: int
    test: int
    cold_pool: int
    train_drugs: int
    valid_drugs: int
    test_drugs: int


def read_all_triples(dataset_root: str) -> List[Triple]:
    path = os.path.join(dataset_root, "DDIN.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source DDI file for cold split: {path}")
    df = pd.read_csv(path)
    expected = {"d1", "type", "d2"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(expected)}")
    return [(str(row.d1), int(row.type), str(row.d2)) for row in df.itertuples(index=False)]


def write_triples(path: str, triples: Sequence[Triple]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(triples, columns=["d1", "type", "d2"]).to_csv(path, index=False)


def write_drug_list(path: str, drugs: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"drug": sorted(str(drug) for drug in drugs)}).to_csv(path, index=False)


def split_validation_from_cold(cold_rows: List[Triple], val_ratio: float, seed: int):
    if len(cold_rows) < 2:
        raise ValueError(f"Cold pool too small for validation/test split: {len(cold_rows)}")
    y = np.asarray([row[1] for row in cold_rows])
    unique, counts = np.unique(y, return_counts=True)
    stratify = y if len(unique) > 1 and int(counts.min()) >= 2 else None
    val_idx, test_idx = train_test_split(
        np.arange(len(cold_rows)),
        test_size=1.0 - float(val_ratio),
        random_state=int(seed),
        stratify=stratify,
    )
    valid = [cold_rows[int(i)] for i in val_idx]
    test = [cold_rows[int(i)] for i in test_idx]
    return valid, test


def build_cold_splits(
    dataset_root: str,
    scenario: str,
    seed: int = 42,
    folds: int = 5,
    val_ratio: float = 0.2,
    output_root: str | None = None,
    overwrite: bool = False,
) -> str:
    """Build S2/S3 cold-start splits.

    S2: old-drug/new-drug valid/test pairs contain exactly one unknown drug.
    S3: new-drug/new-drug valid/test pairs contain two unknown drugs.

    Training contains only known-known pairs. Validation and test are split from
    the cold pool so both follow the requested cold-start distribution.
    """
    scenario = str(scenario).lower().strip()
    if scenario not in {"s2", "s3"}:
        raise ValueError(f"scenario must be s2 or s3, got: {scenario}")
    if folds < 2:
        raise ValueError("folds must be >= 2")

    triples = read_all_triples(dataset_root)
    all_drugs = sorted({d for d1, _, d2 in triples for d in (d1, d2)})
    rng = np.random.RandomState(int(seed))
    shuffled_drugs = np.asarray(all_drugs, dtype=object)
    rng.shuffle(shuffled_drugs)
    unknown_splits = [set(part.tolist()) for part in np.array_split(shuffled_drugs, int(folds))]

    split_root = output_root or os.path.join(dataset_root, f"cold_{scenario}_seed{int(seed)}")
    os.makedirs(split_root, exist_ok=True)
    stats = []
    for fold, unknown in enumerate(unknown_splits):
        fold_dir = os.path.join(split_root, str(fold))
        train_path = os.path.join(fold_dir, "ddi_training1.csv")
        valid_path = os.path.join(fold_dir, "ddi_validation1.csv")
        test_path = os.path.join(fold_dir, "ddi_test1.csv")
        unknown_path = os.path.join(fold_dir, "unknown_drugs.csv")
        if not overwrite and os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
            continue

        train_rows: List[Triple] = []
        cold_rows: List[Triple] = []
        for d1, rel, d2 in triples:
            c1 = d1 in unknown
            c2 = d2 in unknown
            if not c1 and not c2:
                train_rows.append((d1, rel, d2))
            elif scenario == "s2" and (c1 ^ c2):
                cold_rows.append((d1, rel, d2))
            elif scenario == "s3" and c1 and c2:
                cold_rows.append((d1, rel, d2))

        if len(train_rows) == 0:
            raise ValueError(f"Fold {fold} has empty train split for {scenario}")
        if len(cold_rows) < 2:
            raise ValueError(f"Fold {fold} has too few cold samples for {scenario}: {len(cold_rows)}")

        valid_rows, test_rows = split_validation_from_cold(cold_rows, val_ratio=val_ratio, seed=int(seed) + fold)
        if len(valid_rows) == 0 or len(test_rows) == 0:
            raise ValueError(f"Fold {fold} produced empty valid/test for {scenario}")

        write_triples(train_path, train_rows)
        write_triples(valid_path, valid_rows)
        write_triples(test_path, test_rows)
        write_drug_list(unknown_path, unknown)
        stats.append(
            ColdSplitStats(
                scenario=scenario,
                seed=int(seed),
                fold=fold,
                unknown_drugs=len(unknown),
                train=len(train_rows),
                valid=len(valid_rows),
                test=len(test_rows),
                cold_pool=len(cold_rows),
                train_drugs=len({d for d1, _, d2 in train_rows for d in (d1, d2)}),
                valid_drugs=len({d for d1, _, d2 in valid_rows for d in (d1, d2)}),
                test_drugs=len({d for d1, _, d2 in test_rows for d in (d1, d2)}),
            )
        )

    if stats:
        with open(os.path.join(split_root, "split_stats.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in stats], f, indent=2)
    return split_root


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build S2/S3 cold-start splits for Deng/Ryu DDI triples.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True, choices=["s2", "s3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    split_root = build_cold_splits(
        args.dataset_root,
        args.scenario,
        seed=args.seed,
        folds=args.folds,
        val_ratio=args.val_ratio,
        output_root=args.output_root or None,
        overwrite=args.overwrite,
    )
    print(f"cold split root: {split_root}")


if __name__ == "__main__":
    main()
