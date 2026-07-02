"""
Training script for Amyloid PET Centiloid Prediction.

Usage:
    python train.py --train_csv /projectnb/medaihack/ABPET/data/train.csv --val_csv /projectnb/medaihack/ABPET/data/val.csv --patience 10
"""

"""
Updated:
    python train.py \
  --train_csv /projectnb/medaihack/ABPET/data/train.csv \
  --val_csv /projectnb/medaihack/ABPET/data/val.csv \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --resnet_depth 18 \
  --scheduler plateau \
  --lr_factor 0.5 \
  --lr_patience 2 \
  --lr_cooldown 1 \
  --min_lr 1e-6 \
  --patience 10
"""

"""
Training script for Amyloid PET Centiloid Prediction.
"""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PETDataset
from models.model import ResNet3D, BasicBlock3D, Bottleneck3D
from models.losses import get_criterion

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def setup_logger(log_dir: Path, results_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_dir, results_dir, timestamp


def safe_pearson(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if len(preds) < 2:
        return float("nan")
    if preds.std() == 0 or targets.std() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()


def get_resnet_config(depth: int):
    if depth == 10:
        return BasicBlock3D, [1, 1, 1, 1]
    elif depth == 18:
        return BasicBlock3D, [2, 2, 2, 2]
    elif depth == 34:
        return BasicBlock3D, [3, 4, 6, 3]
    elif depth == 50:
        return Bottleneck3D, [3, 4, 6, 3]
    else:
        raise ValueError(f"Unsupported resnet_depth={depth}. Choose from 10, 18, 34, 50.")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n = 0

    for images, centiloids, tracers in loader:
        images = images.to(device, non_blocking=True)
        centiloids = centiloids.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            preds = model(images, tracers)
            loss = criterion(preds, centiloids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, amp_enabled):
    model.eval()
    all_preds, all_targets, all_tracers = [], [], []

    for images, centiloids, tracers in loader:
        images = images.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            preds = model(images, tracers)

        all_preds.append(preds.cpu())
        all_targets.append(centiloids.cpu())
        all_tracers.append(tracers.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    tracer_ids = torch.cat(all_tracers)

    mae = (preds - targets).abs().mean().item()
    corr = safe_pearson(preds, targets)
    return mae, corr, preds, targets, tracer_ids


def save_val_report(preds, targets, tracer_ids, tracer_map, results_dir, timestamp):
    id_to_name = {v: k for k, v in tracer_map.items()}
    rows = []

    def metrics(p, t):
        mae = (p - t).abs().mean().item()
        corr = safe_pearson(p, t)
        return mae, corr, len(p)

    mae, corr, n = metrics(preds, targets)
    rows.append({"tracer": "ALL", "n": n, "mae": mae, "pearson_r": corr})

    for tid in sorted(tracer_ids.unique().tolist()):
        mask = tracer_ids == tid
        mae, corr, n = metrics(preds[mask], targets[mask])
        rows.append({"tracer": id_to_name[tid], "n": n, "mae": mae, "pearson_r": corr})

    report_path = results_dir / f"val_report_{timestamp}.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tracer", "n", "mae", "pearson_r"])
        writer.writeheader()
        writer.writerows(rows)

    return report_path


def save_plots(history, results_dir, timestamp):
    if not HAS_MPL or len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_mae"] for h in history], label="Val MAE", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (CL)")
    axes[1].set_title("Validation MAE")
    axes[1].legend()

    axes[2].plot(epochs, [h["val_corr"] for h in history], label="Val r", color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("Validation Correlation")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"curves_{timestamp}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="/projectnb/medaihack/ABPET/data/train.csv")
    parser.add_argument("--val_csv", type=str, default="/projectnb/medaihack/ABPET/data/val.csv")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--resnet_depth", type=int, default=18, choices=[10, 18, 34, 50])
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="LR reduction factor for plateau scheduler")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Epochs with no improvement before LR reduction")
    parser.add_argument("--lr_cooldown", type=int, default=0,
                        help="Cooldown epochs after LR reduction")
    parser.add_argument("--min_lr", type=float, default=1e-8)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    logger, log_dir, results_dir, timestamp = setup_logger(Path(args.log_dir), Path(args.results_dir))
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = PETDataset(args.train_csv, cache=args.cache)
    val_ds = PETDataset(args.val_csv, tracer_map=train_ds.tracer_map, cache=args.cache)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    block, layers = get_resnet_config(args.resnet_depth)
    mean_cl = train_ds.centiloids.mean().item()

    model = ResNet3D(
        block=block,
        layers=layers,
        in_channels=1,
        num_tracers=len(train_ds.tracer_map),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        mean_centiloid=float(mean_cl),
    ).to(device)

    raw_model = model

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"ResNet depth: {args.resnet_depth}")

    if args.compile and hasattr(torch, "compile"):
        logger.info("Using torch.compile()")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = get_criterion(args.loss)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=args.lr_cooldown,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    else:
        scheduler = None

    metrics_path = results_dir / f"metrics_{timestamp}.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "val_mae", "val_corr", "lr", "epoch_time_s", "is_best"])

    best_mae = float("inf")
    best_preds, best_targets, best_tracer_ids = None, None, None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            amp_enabled=amp_enabled, max_grad_norm=args.max_grad_norm
        )

        val_mae, val_corr, val_preds, val_targets, val_tracer_ids = validate(
            model, val_loader, device, amp_enabled=amp_enabled
        )

        prev_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_mae)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        lr_changed = lr != prev_lr
        epoch_time = time.time() - t0

        is_best = val_mae < best_mae
        tag_parts = []

        if is_best:
            best_mae = val_mae
            best_preds, best_targets, best_tracer_ids = val_preds, val_targets, val_tracer_ids
            epochs_without_improvement = 0

            save_model = raw_model if not hasattr(model, "_orig_mod") else model._orig_mod

            torch.save({
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "tracer_map": train_ds.tracer_map,
                "num_tracers": len(train_ds.tracer_map),
                "resnet_depth": args.resnet_depth,
                "embed_dim": args.embed_dim,
                "dropout": args.dropout,
                "epoch": epoch,
                "best_mae": best_mae,
            }, checkpoint_path)
            tag_parts.append("*")
        else:
            epochs_without_improvement += 1

            # Important: when LR drops, reset early stopping counter
            # so the lower LR gets a fair chance to improve validation MAE.
            if lr_changed:
                epochs_without_improvement = 0
                tag_parts.append("lr-reset")

        if lr_changed:
            tag_parts.append("lr↓")

        tag = f" [{' '.join(tag_parts)}]" if tag_parts else ""

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"MAE: {val_mae:.2f} CL | "
            f"r: {val_corr:.4f} | "
            f"LR: {lr:.2e} | "
            f"{epoch_time:.1f}s | "
            f"ES patience: {epochs_without_improvement}/{args.patience}"
            f"{tag}"
        )

        metrics_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_mae:.4f}",
            f"{val_corr:.6f}" if val_corr == val_corr else "nan",
            f"{lr:.2e}",
            f"{epoch_time:.1f}",
            int(is_best),
        ])
        metrics_file.flush()

        history.append(dict(
            epoch=epoch,
            train_loss=train_loss,
            val_mae=val_mae,
            val_corr=val_corr,
        ))

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(f"Early stopping: no improvement for {args.patience} epochs")
            break

    metrics_file.close()

    save_plots(history, results_dir, timestamp)

    if best_preds is not None:
        report_path = save_val_report(
            best_preds, best_targets, best_tracer_ids,
            train_ds.tracer_map, results_dir, timestamp
        )

        logger.info(f"Best validation MAE: {best_mae:.2f} centiloid units")
        logger.info(f"Model saved to {checkpoint_path}")
        logger.info(f"Metrics: {metrics_path}")
        logger.info(f"Val report: {report_path}")
        if HAS_MPL:
            logger.info(f"Plots: {results_dir / f'curves_{timestamp}.png'}")


if __name__ == "__main__":
    main()
