"""Bias-variance analysis and training history visualization."""

import json
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm

from config import get_args
from dataset import (
    CrackDataset, build_file_list, get_train_transforms, get_val_transforms,
)
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_history(history_path, out_path="training_curves.png"):
    """Plot train/val loss and accuracy from a saved history JSON."""
    with open(history_path) as f:
        h = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(h["train_loss"]) + 1)

    ax1.plot(epochs, h["train_loss"], label="Train Loss")
    ax1.plot(epochs, h["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, h["train_acc"], label="Train Acc")
    ax2.plot(epochs, h["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        all_preds.append(outputs.argmax(1).cpu().numpy())
    return np.concatenate(all_preds)


def bias_variance_analysis(args):
    """Bootstrap-based bias-variance decomposition for the CNN.

    Trains the model `bv_runs` times on different random subsets, collects
    predictions on a held-out set, and computes bias^2, variance, and error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files, labels = build_file_list(args.data_dir)
    labels = np.array(labels)

    # hold out a fixed test portion
    train_idx, test_idx = train_test_split(
        range(len(files)), test_size=0.3, stratify=labels, random_state=args.seed,
    )
    test_files = [files[i] for i in test_idx]
    test_labels = np.array([labels[i] for i in test_idx])
    test_ds = CrackDataset(test_files, test_labels.tolist(),
                           transform=get_val_transforms(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    pool_files = [files[i] for i in train_idx]
    pool_labels = np.array([labels[i] for i in train_idx])

    n_test = len(test_files)
    all_preds = np.zeros((args.bv_runs, n_test), dtype=np.int64)

    for run in range(args.bv_runs):
        print(f"\n=== Bias-Variance Run {run + 1}/{args.bv_runs} ===")
        set_seed(args.seed + run)

        # bootstrap subsample
        n_sub = int(len(pool_files) * args.bv_train_ratio)
        sub_idx = np.random.choice(len(pool_files), size=n_sub, replace=False)
        sub_files = [pool_files[i] for i in sub_idx]
        sub_labels = pool_labels[sub_idx].tolist()

        train_ds = CrackDataset(sub_files, sub_labels,
                                transform=get_train_transforms(args.img_size))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        model = build_model(args).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )

        # quick train
        for epoch in range(1, args.epochs + 1):
            model.train()
            for imgs, lbls in tqdm(train_loader,
                                   desc=f"  run {run+1} epoch {epoch}",
                                   leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()

        all_preds[run] = get_predictions(model, test_loader, device)

    # compute bias-variance (0-1 loss)
    mode_preds = stats.mode(all_preds, axis=0).values.flatten()
    bias_sq = np.mean(mode_preds != test_labels)
    variance = np.mean([
        np.mean(all_preds[r] != mode_preds) for r in range(args.bv_runs)
    ])
    error = np.mean([
        np.mean(all_preds[r] != test_labels) for r in range(args.bv_runs)
    ])

    print(f"\n--- Bias-Variance Results ({args.bv_runs} runs) ---")
    print(f"  Avg Error : {error:.4f}")
    print(f"  Bias^2    : {bias_sq:.4f}")
    print(f"  Variance  : {variance:.4f}")

    return {"error": error, "bias_sq": bias_sq, "variance": variance}


def main():
    args = get_args()

    if args.bv_runs > 0:
        bias_variance_analysis(args)
    elif args.checkpoint:
        hist_path = args.checkpoint.replace("_best.pt", "_history.json")
        plot_history(hist_path)
    else:
        print("Provide --bv_runs N for bias-variance analysis, "
              "or --checkpoint PATH to plot training curves.")


if __name__ == "__main__":
    main()
