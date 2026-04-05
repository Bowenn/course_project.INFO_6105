import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
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


def build_optimizer(model, args):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                               weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                gamma=args.gamma)
    return None


def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def run_val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
    files, labels = build_file_list(args.data_dir)
    train_idx, val_idx = train_test_split(
        range(len(files)), test_size=args.val_ratio,
        stratify=labels, random_state=args.seed,
    )
    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = CrackDataset(train_files, train_labels,
                            transform=get_train_transforms(args.img_size))
    val_ds = CrackDataset(val_files, val_labels,
                          transform=get_val_transforms(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # model
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    # checkpoint dir
    os.makedirs(args.save_dir, exist_ok=True)
    run_tag = args.run_name or f"{args.model}_lr{args.lr}_bs{args.batch_size}"

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc = run_val_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss  ={val_loss:.4f}  val_acc  ={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, f"{run_tag}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> saved best model ({val_acc:.4f})")

    # save training history
    hist_path = os.path.join(args.save_dir, f"{run_tag}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
