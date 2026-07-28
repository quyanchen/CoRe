import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

Row = tuple[str, int, str]

IMBALANCE_FACTOR = 100.0
VAL_RATIO = 0.1
TEST_RATIO = 0.2
MIN_VAL = 1
MIN_TEST = 1
MIN_TRAIN = 3
MIN_TAIL = 3

DEFAULT_DATASETS = {
    "deng": {
        "name": "Deng",
        "pairs": Path("./data/primekg+deng/ddi/newddixiao-1.csv"),
        "smiles": Path("./data/Deng/drug_smiles.csv"),
    },
    "ryu": {
        "name": "Ryu",
        "pairs": Path("./data/primekg+ryu/ddi/newddixiao-1.csv"),
        "smiles": Path("./data/Ryu/drug_smiles.csv"),
    },
}


def read_pairs(path: Path) -> list[Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            (str(row["d1"]).strip(), int(row.get("type", row.get("label"))), str(row["d2"]).strip())
            for row in csv.DictReader(file)
            if row.get("d1") and row.get("d2") and row.get("type", row.get("label")) not in {None, ""}
        ]
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def write_pairs(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["d1", "type", "d2"])
        writer.writerows(rows)


def split_by_class(rows: list[Row], seed: int) -> tuple[dict[int, list[Row]], list[Row], list[Row]]:
    rng = random.Random(seed)
    grouped: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)

    train_pool: dict[int, list[Row]] = {}
    valid: list[Row] = []
    test: list[Row] = []

    for relation, class_rows in sorted(grouped.items()):
        rng.shuffle(class_rows)
        n_valid = max(MIN_VAL, round(len(class_rows) * VAL_RATIO))
        n_test = max(MIN_TEST, round(len(class_rows) * TEST_RATIO))
        while len(class_rows) - n_valid - n_test < MIN_TRAIN:
            if n_test > MIN_TEST:
                n_test -= 1
            elif n_valid > MIN_VAL:
                n_valid -= 1
            else:
                raise ValueError(f"Relation {relation} has too few samples")

        valid.extend(class_rows[:n_valid])
        test.extend(class_rows[n_valid:n_valid + n_test])
        train_pool[relation] = class_rows[n_valid + n_test:]

    rng.shuffle(valid)
    rng.shuffle(test)
    return train_pool, valid, test


def build_lt100(train_pool: dict[int, list[Row]], seed: int) -> tuple[list[Row], dict[int, int]]:
    rng = random.Random(seed)
    relations = sorted(train_pool, key=lambda relation: (-len(train_pool[relation]), relation))
    head_count = min(int(IMBALANCE_FACTOR * MIN_TAIL), len(train_pool[relations[0]]))
    train: list[Row] = []
    counts: dict[int, int] = {}

    for rank, relation in enumerate(relations):
        class_rows = train_pool[relation]
        rng.shuffle(class_rows)
        exponent = 0.0 if len(relations) == 1 else rank / (len(relations) - 1)
        target = round(head_count * IMBALANCE_FACTOR ** (-exponent))
        keep = min(len(class_rows), max(MIN_TAIL, target))
        train.extend(class_rows[:keep])
        counts[relation] = keep

    rng.shuffle(train)
    return train, counts


def write_mapping(path: Path, mapping: dict[int, int], source_counts: Counter) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_relation_id", "relation_id", "source_count"])
        writer.writerows(
            (source, target, source_counts[source])
            for source, target in sorted(mapping.items(), key=lambda item: item[1])
        )


def build_dataset(name: str, pair_path: Path, smiles_path: Path, output_root: Path, seed: int) -> None:
    rows = read_pairs(pair_path)
    source_counts = Counter(relation for _, relation, _ in rows)
    minimum_count = MIN_VAL + MIN_TEST + MIN_TRAIN
    kept_relations = sorted(relation for relation, count in source_counts.items() if count >= minimum_count)
    mapping = {source: target for target, source in enumerate(kept_relations)}
    remapped = [(d1, mapping[relation], d2) for d1, relation, d2 in rows if relation in mapping]

    train_pool, valid, test = split_by_class(remapped, seed)
    train, train_counts = build_lt100(train_pool, seed + 100)

    output_dir = output_root / name / f"LT100_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(output_dir / "train.csv", train)
    write_pairs(output_dir / "valid.csv", valid)
    write_pairs(output_dir / "test.csv", test)
    write_mapping(output_dir / "label_mapping.csv", mapping, source_counts)
    shutil.copyfile(smiles_path, output_dir / "drug_smiles.csv")

    actual_if = max(train_counts.values()) / min(train_counts.values())
    print(
        f"[{name}] train={len(train)} valid={len(valid)} test={len(test)} "
        f"classes={len(mapping)} IF={actual_if:.2f} output={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-root", type=Path, default=Path("./data/controlled_longtail"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deng-pairs", type=Path, default=DEFAULT_DATASETS["deng"]["pairs"])
    parser.add_argument("--deng-smiles", type=Path, default=DEFAULT_DATASETS["deng"]["smiles"])
    parser.add_argument("--ryu-pairs", type=Path, default=DEFAULT_DATASETS["ryu"]["pairs"])
    parser.add_argument("--ryu-smiles", type=Path, default=DEFAULT_DATASETS["ryu"]["smiles"])
    args = parser.parse_args()

    paths = {
        "deng": (args.deng_pairs, args.deng_smiles),
        "ryu": (args.ryu_pairs, args.ryu_smiles),
    }
    for dataset in args.datasets:
        pair_path, smiles_path = paths[dataset]
        build_dataset(
            DEFAULT_DATASETS[dataset]["name"],
            pair_path,
            smiles_path,
            args.output_root,
            args.seed,
        )


if __name__ == "__main__":
    main()
