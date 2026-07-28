from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .data import build_datasets, collate_drug_pairs
    from .model import (
        TripletRoleDDIModel,
        base_role_regularization,
        build_counterfactual_masks,
        context_role_separation_loss,
        counterfactual_motif_fidelity_loss,
        weighted_triplet_role_alignment_loss,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code.data import build_datasets, collate_drug_pairs
    from code.model import (
        TripletRoleDDIModel,
        base_role_regularization,
        build_counterfactual_masks,
        context_role_separation_loss,
        counterfactual_motif_fidelity_loss,
        weighted_triplet_role_alignment_loss,
    )


class TeeLogger:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "a", encoding="utf-8")

    def log(self, message: str):
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    true = labels.detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_rec": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_pre": float(precision_score(true, pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        graph = batch["graph"].to(device)
        labels = batch["relation"].to(device)
        out = model(graph)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())
    return metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))


def build_class_weights(train_set, num_relations: int, mode: str, beta: float, device: torch.device):
    if mode == "none":
        return None
    counts = torch.zeros(num_relations, dtype=torch.float32)
    for ex in train_set.triples:
        counts[int(ex.relation)] += 1.0
    counts = counts.clamp_min(1.0)
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "effective":
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts))
    else:
        raise ValueError("--class-weight must be none, inverse, sqrt_inverse, or effective")
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.to(device)


def metric_tag(value: float) -> str:
    return f"{int(round(float(value) * 10000)):04d}"


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 2
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def metric_checkpoint_path(output_dir: str, dataset_name: str, scenario: str, fold: int, macro_f1: float) -> str:
    tag = metric_tag(macro_f1)
    if scenario == "s1":
        name = f"{dataset_name}_fold{fold}_{tag}.pt"
    else:
        name = f"{dataset_name}_{scenario}_fold{fold}_{tag}.pt"
    return unique_path(os.path.join(output_dir, name))


def train_one_epoch(model, loader, optimizer, device, args, class_weights):
    model.train()
    total = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        graph = batch["graph"].to(device)
        labels = batch["relation"].to(device)
        out = model(graph, labels)

        loss_pred = F.cross_entropy(out["logits"], labels, weight=class_weights)
        loss_align = weighted_triplet_role_alignment_loss(
            out,
            labels,
            batch["drug_a"],
            batch["drug_b"],
            temperature=args.align_temperature,
            partner_weight=args.neg_partner_weight,
            relation_weight=args.neg_relation_weight,
            direction_weight=args.neg_direction_weight,
        )
        loss_role = base_role_regularization(out)
        loss_role_sep = context_role_separation_loss(
            out,
            labels,
            batch["drug_a"],
            batch["drug_b"],
            margin=args.role_sep_margin,
        )

        if args.cf_every > 0 and step % args.cf_every == 0:
            top_mask, random_mask, low_mask = build_counterfactual_masks(
                out["node_attention"].detach(),
                out["graph_id"],
                side=out["side"],
                top_ratio=args.cf_top_ratio,
            )
            top_out = model(graph, labels, node_keep_mask=top_mask)
            random_out = model(graph, labels, node_keep_mask=random_mask)
            low_out = model(graph, labels, node_keep_mask=low_mask)
            loss_cf = counterfactual_motif_fidelity_loss(
                out["logits"],
                top_out["logits"],
                random_out["logits"],
                low_out["logits"],
                labels,
                margin=args.cf_margin,
                random_tolerance=args.cf_random_tolerance,
            )
        else:
            loss_cf = out["logits"].new_tensor(0.0)

        loss = (
            loss_pred
            + args.lambda_align * loss_align
            + args.lambda_role * loss_role
            + args.lambda_role_sep * loss_role_sep
            + args.lambda_cf * loss_cf
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(description="Triplet-conditioned molecular role DDI training")
    parser.add_argument("--dataset-root", type=str, default="Ryu", help="Path to Deng or Ryu directory")
    parser.add_argument("--scenario", type=str, default="s3", choices=["s1", "s2", "s3"], help="S1 random split, S2 old-new cold start, or S3 new-new cold start.")
    parser.add_argument("--cold-seed", type=int, default=42)
    parser.add_argument("--cold-val-ratio", type=float, default=0.2, help="Validation fraction from the cold pool for S2/S3.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hig-layers", type=int, default=3)
    parser.add_argument("--role-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion-dim", type=int, default=8, choices=[3, 5, 7, 8], help="Number of hidden-dimension feature blocks used by the final classifier.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-role", type=float, default=1e-3)
    parser.add_argument("--lambda-role-sep", type=float, default=0.03)
    parser.add_argument("--lambda-cf", type=float, default=0.05)
    parser.add_argument("--align-temperature", type=float, default=0.1)
    parser.add_argument("--role-sep-margin", type=float, default=0.45)
    parser.add_argument("--neg-partner-weight", type=float, default=2.0)
    parser.add_argument("--neg-relation-weight", type=float, default=2.0)
    parser.add_argument("--neg-direction-weight", type=float, default=2.0)
    parser.add_argument("--cf-top-ratio", type=float, default=0.2)
    parser.add_argument("--cf-every", type=int, default=3, help="Compute counterfactual motif loss every N batches. 1 means every batch; 0 disables it.")
    parser.add_argument("--cf-margin", type=float, default=0.15)
    parser.add_argument("--cf-random-tolerance", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--class-weight", type=str, default="sqrt_inverse", choices=["none", "inverse", "sqrt_inverse", "effective"])
    parser.add_argument("--cb-beta", type=float, default=0.9999, help="Beta for effective-number class-balanced weights.")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience on validation macro-F1. <=0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation macro-F1 improvement required to reset patience.")
    parser.add_argument("--no-eval-train", default="no-eval-train", help="Skip training-set metrics after each epoch.")
    parser.add_argument("--output-dir", type=str, default="run", help="Directory for timestamped logs and best checkpoints.")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory for persistent per-drug BRICS fragment graph caches. Empty disables disk cache.")
    parser.add_argument("--save-path", type=str, default="", help="Optional explicit checkpoint path. Defaults to <dataset>_fold<fold>_best.pt in output-dir.")
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.dataset_root))
    run_name = (
        f"{dataset_name}_{args.scenario}_fold{args.fold}_fusion{args.fusion_dim}"
        if args.scenario != "s1"
        else f"{dataset_name}_fold{args.fold}_fusion{args.fusion_dim}"
    )
    explicit_save_path = bool(args.save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{run_name}_log_{timestamp}.log")
    save_path = args.save_path or unique_path(os.path.join(args.output_dir, f"{run_name}_best_{timestamp}.pt"))
    args.save_path = save_path
    logger = TeeLogger(log_path)
    logger.log(f"log_path={log_path}")
    logger.log(f"save_path={save_path}")
    logger.log("args=" + json.dumps(vars(args), indent=2, sort_keys=True))

    cache_dir = args.cache_dir if args.cache_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.log(f"cache_dir={cache_dir}")
    else:
        logger.log("cache_dir=disabled")
    train_set, valid_set, test_set, num_relations = build_datasets(
        args.dataset_root,
        args.fold,
        cache_dir=cache_dir,
        scenario=args.scenario,
        cold_seed=args.cold_seed,
        cold_val_ratio=args.cold_val_ratio,
    )
    logger.log(
        f"dataset={dataset_name} scenario={args.scenario} fold={args.fold} train={len(train_set)} valid={len(valid_set)} test={len(test_set)} num_relations={num_relations}"
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_drug_pairs
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_drug_pairs
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_drug_pairs
    )

    device = torch.device(args.device)
    logger.log(f"device={device}")
    model = TripletRoleDDIModel(
        num_relations=num_relations,
        hidden_dim=args.hidden_dim,
        hig_layers=args.hig_layers,
        role_layers=args.role_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        fusion_dim=args.fusion_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = build_class_weights(train_set, num_relations, args.class_weight, args.cb_beta, device)
    if class_weights is None:
        logger.log("class_weights=none")
    else:
        logger.log(
            f"class_weights mode={args.class_weight} min={float(class_weights.min()):.4f} "
            f"max={float(class_weights.max()):.4f} mean={float(class_weights.mean()):.4f}"
        )

    best_valid = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args, class_weights)
        train_metrics = None if args.no_eval_train else evaluate(model, train_loader, device)
        valid = evaluate(model, valid_loader, device)
        train_part = ""
        if train_metrics is not None:
            train_part = (
                f"train_acc={train_metrics['acc']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
                f"train_macro_rec={train_metrics['macro_rec']:.4f} train_macro_pre={train_metrics['macro_pre']:.4f} "
            )
        logger.log(
            f"epoch={epoch:03d} loss={loss:.4f} "
            f"{train_part}"
            f"valid_acc={valid['acc']:.4f} valid_macro_f1={valid['macro_f1']:.4f} "
            f"valid_macro_rec={valid['macro_rec']:.4f} valid_macro_pre={valid['macro_pre']:.4f} "
            f"best_macro_f1={best_valid:.4f}"
        )
        if valid["macro_f1"] > best_valid + args.min_delta:
            best_valid = valid["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "num_relations": num_relations,
                    "dataset": dataset_name,
                    "scenario": args.scenario,
                    "fold": args.fold,
                    "best_epoch": best_epoch,
                    "best_valid_macro_f1": best_valid,
                },
                args.save_path,
            )
            test_at_best = evaluate(model, test_loader, device)
            logger.log(
                f"saved_best epoch={epoch:03d} valid_macro_f1={best_valid:.4f} path={args.save_path} "
                f"test_acc={test_at_best['acc']:.4f} test_macro_f1={test_at_best['macro_f1']:.4f} "
                f"test_macro_rec={test_at_best['macro_rec']:.4f} test_macro_pre={test_at_best['macro_pre']:.4f}"
            )
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                logger.log(
                    f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_valid_macro_f1={best_valid:.4f} patience={args.patience}"
                )
                break

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    logger.log(
        f"best_epoch={ckpt.get('best_epoch', best_epoch)} best_valid_macro_f1={ckpt.get('best_valid_macro_f1', best_valid):.4f}"
    )
    logger.log(
        f"test_acc={test['acc']:.4f} test_macro_f1={test['macro_f1']:.4f} "
        f"test_macro_rec={test['macro_rec']:.4f} test_macro_pre={test['macro_pre']:.4f}"
    )
    logger.log(
        f"FINAL\tAcc.\tMacro-F1\tMacro-Rec.\tMacro-Pre.\n"
        f"FINAL\t{test['acc']:.4f}\t{test['macro_f1']:.4f}\t{test['macro_rec']:.4f}\t{test['macro_pre']:.4f}"
    )
    final_save_path = save_path
    ckpt["test_metrics"] = test
    if not explicit_save_path:
        final_save_path = metric_checkpoint_path(args.output_dir, dataset_name, args.scenario, args.fold, test["macro_f1"])
    ckpt["final_checkpoint"] = final_save_path
    torch.save(ckpt, final_save_path)
    if final_save_path != save_path and os.path.exists(save_path):
        os.remove(save_path)
    logger.log(f"final_checkpoint={final_save_path}")
    logger.close()


if __name__ == "__main__":
    main()
