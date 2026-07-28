import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


Row = Tuple[str, int, str]


DEFAULT_DATASETS = {
    "deng": {
        "pairs": "./data/primekg+deng/ddi/newddixiao-1.csv",
        "relation_family": "./data/primekg+deng/ddi/relation_family.csv",
    },
    "ryu": {
        "pairs": "./data/primekg+ryu/ddi/newddixiao-1.csv",
        "relation_family": "./data/primekg+ryu/ddi/relation_family.csv",
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_pairs(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d1 = str(row.get("d1", "")).strip()
            d2 = str(row.get("d2", "")).strip()
            label_raw = row.get("type", row.get("label", ""))
            if not d1 or not d2 or str(label_raw).strip() == "":
                continue
            rows.append((d1, int(label_raw), d2))
    if not rows:
        raise ValueError(f"No valid DDI rows loaded from {path}")
    return rows


def write_pairs(path: str, rows: Iterable[Row]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["d1", "type", "d2"])
        writer.writeheader()
        for d1, rel, d2 in rows:
            writer.writerow({"d1": d1, "type": int(rel), "d2": d2})


def read_relation_family(path: str) -> Dict[int, Tuple[int, str]]:
    mapping: Dict[int, Tuple[int, str]] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid_raw = row.get("relation_id", "")
            if str(rid_raw).strip() == "":
                continue
            rid = int(rid_raw)
            fam_raw = row.get("family_id", "")
            fam_name = str(row.get("family_name", row.get("mechanism_family", "unknown_mixed"))).strip()
            if fam_name == "":
                fam_name = "unknown_mixed"
            try:
                fam_id = int(fam_raw)
            except Exception:
                fam_id = 0
            mapping[rid] = (fam_id, fam_name)
    return mapping


def write_relation_family(path: str, old_to_new: Dict[int, int], source_family: Dict[int, Tuple[int, str]]) -> None:
    family_name_to_new_id: Dict[str, int] = {}
    rows = []
    for old_rel, new_rel in sorted(old_to_new.items(), key=lambda kv: kv[1]):
        _, fam_name = source_family.get(old_rel, (0, "unknown_mixed"))
        if fam_name not in family_name_to_new_id:
            family_name_to_new_id[fam_name] = len(family_name_to_new_id)
        rows.append(
            {
                "relation_id": new_rel,
                "family_id": family_name_to_new_id[fam_name],
                "family_name": fam_name,
                "source_relation_id": old_rel,
            }
        )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation_id", "family_id", "family_name", "source_relation_id"])
        writer.writeheader()
        writer.writerows(rows)


def write_label_mapping(path: str, old_to_new: Dict[int, int], original_counts: Counter) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_relation_id", "relation_id", "original_count"],
        )
        writer.writeheader()
        for old_rel, new_rel in sorted(old_to_new.items(), key=lambda kv: kv[1]):
            writer.writerow(
                {
                    "source_relation_id": old_rel,
                    "relation_id": new_rel,
                    "original_count": int(original_counts[old_rel]),
                }
            )


def remap_rows(rows: Iterable[Row], old_to_new: Dict[int, int]) -> List[Row]:
    return [(d1, old_to_new[rel], d2) for d1, rel, d2 in rows if rel in old_to_new]


def allocate_holdout_counts(
    total: int,
    val_ratio: float,
    test_ratio: float,
    min_val: int,
    min_test: int,
    min_train: int,
) -> Tuple[int, int]:
    if total < min_val + min_test + min_train:
        raise ValueError(
            f"total={total} is too small for min_val={min_val}, min_test={min_test}, min_train={min_train}"
        )
    n_val = max(min_val, int(round(total * val_ratio)))
    n_test = max(min_test, int(round(total * test_ratio)))
    overflow = n_val + n_test + min_train - total
    if overflow > 0:
        reduce_test = min(overflow, max(0, n_test - min_test))
        n_test -= reduce_test
        overflow -= reduce_test
    if overflow > 0:
        reduce_val = min(overflow, max(0, n_val - min_val))
        n_val -= reduce_val
        overflow -= reduce_val
    if overflow > 0:
        raise ValueError(f"Cannot allocate holdout counts for total={total}")
    return n_val, n_test


def split_by_class(
    rows: List[Row],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_val: int,
    min_test: int,
    min_train: int,
) -> Tuple[Dict[int, List[Row]], List[Row], List[Row]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Row]] = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)

    train_pool: Dict[int, List[Row]] = {}
    val_rows: List[Row] = []
    test_rows: List[Row] = []

    for rel in sorted(by_label):
        cls_rows = list(by_label[rel])
        rng.shuffle(cls_rows)
        n_val, n_test = allocate_holdout_counts(
            total=len(cls_rows),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_val=min_val,
            min_test=min_test,
            min_train=min_train,
        )
        val_rows.extend(cls_rows[:n_val])
        test_rows.extend(cls_rows[n_val : n_val + n_test])
        train_pool[rel] = cls_rows[n_val + n_test :]

    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_pool, val_rows, test_rows


def build_longtail_train(
    train_pool: Dict[int, List[Row]],
    imbalance_factor: float,
    max_head_train: int,
    min_tail_train: int,
    seed: int,
) -> Tuple[List[Row], Dict[int, int], List[int]]:
    rng = random.Random(seed)
    labels_by_frequency = sorted(train_pool.keys(), key=lambda r: (-len(train_pool[r]), r))
    class_num = len(labels_by_frequency)
    if class_num == 0:
        raise ValueError("No classes available for long-tail construction")
    head_available = len(train_pool[labels_by_frequency[0]])
    if max_head_train > 0:
        head_count = min(int(max_head_train), head_available)
    else:
        head_count = min(int(round(float(imbalance_factor) * float(min_tail_train))), head_available)
    head_count = max(1, head_count)

    train_rows: List[Row] = []
    target_counts: Dict[int, int] = {}
    for rank, rel in enumerate(labels_by_frequency):
        available = list(train_pool[rel])
        rng.shuffle(available)
        if class_num == 1:
            target = head_count
        else:
            target = head_count * math.pow(float(imbalance_factor), -float(rank) / float(class_num - 1))
        keep = int(round(target))
        keep = max(int(min_tail_train), keep)
        keep = min(keep, len(available))
        target_counts[rel] = keep
        train_rows.extend(available[:keep])

    rng.shuffle(train_rows)
    return train_rows, target_counts, labels_by_frequency


def make_balanced_test(test_rows: List[Row], cap_per_class: int, seed: int) -> List[Row]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Row]] = defaultdict(list)
    for row in test_rows:
        by_label[row[1]].append(row)
    min_count = min(len(v) for v in by_label.values())
    per_class = min(min_count, int(cap_per_class)) if cap_per_class > 0 else min_count
    balanced: List[Row] = []
    for rel in sorted(by_label):
        cls_rows = list(by_label[rel])
        rng.shuffle(cls_rows)
        balanced.extend(cls_rows[:per_class])
    rng.shuffle(balanced)
    return balanced


def shot_group(count: int, many_threshold: int, medium_threshold: int) -> str:
    if count >= many_threshold:
        return "many"
    if count >= medium_threshold:
        return "medium"
    return "few"


def write_relation_frequency(
    path: str,
    labels_by_frequency: List[int],
    original_counts: Counter,
    pool_counts: Counter,
    train_counts: Dict[int, int],
    val_counts: Counter,
    test_counts: Counter,
    many_threshold: int,
    medium_threshold: int,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_by_train_pool",
                "relation_id",
                "original_count",
                "train_pool_count",
                "longtail_train_count",
                "val_count",
                "test_count",
                "shot_group",
            ],
        )
        writer.writeheader()
        for rank, rel in enumerate(labels_by_frequency):
            lt_count = int(train_counts.get(rel, 0))
            writer.writerow(
                {
                    "rank_by_train_pool": rank,
                    "relation_id": rel,
                    "original_count": int(original_counts[rel]),
                    "train_pool_count": int(pool_counts[rel]),
                    "longtail_train_count": lt_count,
                    "val_count": int(val_counts[rel]),
                    "test_count": int(test_counts[rel]),
                    "shot_group": shot_group(lt_count, many_threshold, medium_threshold),
                }
            )


def count_labels(rows: Iterable[Row]) -> Counter:
    return Counter(rel for _, rel, _ in rows)


def build_for_dataset(
    dataset_name: str,
    pair_path: str,
    source_relation_family_path: str,
    output_root: str,
    imbalance_factors: List[float],
    seed: int,
    min_class_count: int,
    val_ratio: float,
    test_ratio: float,
    min_val_per_class: int,
    min_test_per_class: int,
    min_train_per_class: int,
    max_head_train: int,
    min_tail_train: int,
    balanced_test_cap: int,
    many_threshold: int,
    medium_threshold: int,
) -> None:
    rows = read_pairs(pair_path)
    original_counts = count_labels(rows)
    kept_old_labels = sorted([rel for rel, cnt in original_counts.items() if cnt >= min_class_count])
    dropped_old_labels = sorted([rel for rel, cnt in original_counts.items() if cnt < min_class_count])
    old_to_new = {old: new for new, old in enumerate(kept_old_labels)}
    filtered_rows = remap_rows(rows, old_to_new)
    remapped_original_counts = count_labels(filtered_rows)

    dataset_out = os.path.join(output_root, dataset_name)
    ensure_dir(dataset_out)
    write_label_mapping(os.path.join(dataset_out, "label_mapping.csv"), old_to_new, original_counts)
    write_relation_family(
        os.path.join(dataset_out, "relation_family.csv"),
        old_to_new=old_to_new,
        source_family=read_relation_family(source_relation_family_path),
    )

    print(
        f"[{dataset_name}] source_rows={len(rows)} kept_rows={len(filtered_rows)} "
        f"kept_classes={len(kept_old_labels)} dropped_classes={len(dropped_old_labels)}"
    )
    if dropped_old_labels:
        print(
            f"[{dataset_name}] dropped labels with count < {min_class_count}: "
            + ", ".join(f"{rel}:{original_counts[rel]}" for rel in dropped_old_labels)
        )

    train_pool, val_rows, test_rows = split_by_class(
        filtered_rows,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_val=min_val_per_class,
        min_test=min_test_per_class,
        min_train=min_train_per_class,
    )
    pool_counts = Counter({rel: len(v) for rel, v in train_pool.items()})
    val_counts = count_labels(val_rows)
    test_counts = count_labels(test_rows)
    balanced_test_rows = make_balanced_test(test_rows, cap_per_class=balanced_test_cap, seed=seed + 17)

    for imbalance_factor in imbalance_factors:
        lt_rows, train_counts, labels_by_frequency = build_longtail_train(
            train_pool=train_pool,
            imbalance_factor=imbalance_factor,
            max_head_train=max_head_train,
            min_tail_train=min_tail_train,
            seed=seed + int(round(imbalance_factor * 100)),
        )
        if_tag = str(int(imbalance_factor)) if float(imbalance_factor).is_integer() else str(imbalance_factor)
        split_out = os.path.join(dataset_out, f"LT{if_tag}_seed{seed}")
        ensure_dir(split_out)
        write_pairs(os.path.join(split_out, "ddi_training1.csv"), lt_rows)
        write_pairs(os.path.join(split_out, "ddi_validation1.csv"), val_rows)
        write_pairs(os.path.join(split_out, "ddi_test1.csv"), test_rows)
        write_pairs(os.path.join(split_out, "ddi_test_balanced.csv"), balanced_test_rows)
        write_label_mapping(os.path.join(split_out, "label_mapping.csv"), old_to_new, original_counts)
        write_relation_family(
            os.path.join(split_out, "relation_family.csv"),
            old_to_new=old_to_new,
            source_family=read_relation_family(source_relation_family_path),
        )
        write_relation_frequency(
            os.path.join(split_out, "relation_frequency.csv"),
            labels_by_frequency=labels_by_frequency,
            original_counts=remapped_original_counts,
            pool_counts=pool_counts,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
            many_threshold=many_threshold,
            medium_threshold=medium_threshold,
        )
        train_count_values = list(train_counts.values())
        summary = {
            "dataset": dataset_name,
            "source_pair_path": pair_path,
            "imbalance_factor": imbalance_factor,
            "seed": seed,
            "class_num": len(kept_old_labels),
            "dropped_source_relation_ids": dropped_old_labels,
            "train_rows": len(lt_rows),
            "validation_rows": len(val_rows),
            "test_rows": len(test_rows),
            "balanced_test_rows": len(balanced_test_rows),
            "max_train_class_count": max(train_count_values),
            "min_train_class_count": min(train_count_values),
            "actual_train_imbalance": max(train_count_values) / max(1, min(train_count_values)),
            "max_head_train": max_head_train,
            "head_count_mode": "fixed_cap" if max_head_train > 0 else "imbalance_factor_times_min_tail_train",
            "min_tail_train": min_tail_train,
            "many_threshold": many_threshold,
            "medium_threshold": medium_threshold,
        }
        with open(os.path.join(split_out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(
            f"[{dataset_name} LT{if_tag}] train={len(lt_rows)} val={len(val_rows)} "
            f"test={len(test_rows)} classes={len(kept_old_labels)} "
            f"actual_IF={summary['actual_train_imbalance']:.2f} out={split_out}"
        )


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("empty float list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled long-tailed DDI relation prediction splits.")
    parser.add_argument("--datasets", type=str, default="deng,ryu", help="Comma-separated names: deng, ryu.")
    parser.add_argument("--output_root", type=str, default="./data/controlled_longtail")
    parser.add_argument("--imbalance_factors", type=str, default="10,50,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_class_count", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_val_per_class", type=int, default=1)
    parser.add_argument("--min_test_per_class", type=int, default=1)
    parser.add_argument("--min_train_per_class", type=int, default=3)
    parser.add_argument(
        "--max_head_train",
        type=int,
        default=0,
        help="Fixed cap for the head relation. Use 0 to set head_count ~= imbalance_factor * min_tail_train.",
    )
    parser.add_argument("--min_tail_train", type=int, default=3)
    parser.add_argument("--balanced_test_cap", type=int, default=20)
    parser.add_argument("--many_threshold", type=int, default=100)
    parser.add_argument("--medium_threshold", type=int, default=20)
    parser.add_argument("--deng_pairs", type=str, default=DEFAULT_DATASETS["deng"]["pairs"])
    parser.add_argument("--deng_relation_family", type=str, default=DEFAULT_DATASETS["deng"]["relation_family"])
    parser.add_argument("--ryu_pairs", type=str, default=DEFAULT_DATASETS["ryu"]["pairs"])
    parser.add_argument("--ryu_relation_family", type=str, default=DEFAULT_DATASETS["ryu"]["relation_family"])
    args = parser.parse_args()

    selected = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    imbalance_factors = parse_float_list(args.imbalance_factors)
    ensure_dir(args.output_root)

    for dataset_name in selected:
        if dataset_name == "deng":
            pair_path = args.deng_pairs
            relation_family_path = args.deng_relation_family
        elif dataset_name == "ryu":
            pair_path = args.ryu_pairs
            relation_family_path = args.ryu_relation_family
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        build_for_dataset(
            dataset_name=dataset_name,
            pair_path=pair_path,
            source_relation_family_path=relation_family_path,
            output_root=args.output_root,
            imbalance_factors=imbalance_factors,
            seed=args.seed,
            min_class_count=args.min_class_count,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_val_per_class=args.min_val_per_class,
            min_test_per_class=args.min_test_per_class,
            min_train_per_class=args.min_train_per_class,
            max_head_train=args.max_head_train,
            min_tail_train=args.min_tail_train,
            balanced_test_cap=args.balanced_test_cap,
            many_threshold=args.many_threshold,
            medium_threshold=args.medium_threshold,
        )


if __name__ == "__main__":
    main()
