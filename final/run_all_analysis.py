"""Run bias-variance analysis and plot training curves for all checkpoints.

For each *_best.pt file in the checkpoints directory:
  1. Plot train/val loss and accuracy curves
  2. Run bootstrap bias-variance analysis

Then generate two comparison plots across all runs:
  - Val accuracy curves
  - Bias-variance bar chart
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import bias_variance_analysis, plot_history


def get_cli_args():
    parser = argparse.ArgumentParser(description="Run analysis on all checkpoints")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--bv_runs", type=int, default=10,
                        help="bootstrap runs per checkpoint")
    parser.add_argument("--bv_train_ratio", type=float, default=0.5,
                        help="fraction of training data per bootstrap sample")
    parser.add_argument("--bv_epochs", type=int, default=10,
                        help="training epochs per BV run (can be fewer than full training)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_bv", action="store_true",
                        help="only plot training curves, skip BV analysis")
    return parser.parse_args()


def plot_val_acc_comparison(histories, out_path):
    """Overlay val accuracy curves for all runs on one figure."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for run_tag, h in histories.items():
        epochs = range(1, len(h["val_acc"]) + 1)
        ax.plot(epochs, h["val_acc"], marker=".", label=run_tag)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Validation Accuracy — All Runs")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {out_path}")


def plot_bv_comparison(bv_results, out_path):
    """Grouped bar chart: bias^2, variance, error for each run."""
    run_tags = list(bv_results.keys())
    errors   = [bv_results[r]["error"]    for r in run_tags]
    biases   = [bv_results[r]["bias_sq"]  for r in run_tags]
    variances= [bv_results[r]["variance"] for r in run_tags]

    x = np.arange(len(run_tags))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(run_tags) * 1.5), 5))
    ax.bar(x - width, errors,    width, label="Avg Error",  color="tomato")
    ax.bar(x,         biases,    width, label="Bias²",      color="steelblue")
    ax.bar(x + width, variances, width, label="Variance",   color="mediumseagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(run_tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Bias-Variance Decomposition — All Runs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved BV comparison plot to {out_path}")


def main():
    cli = get_cli_args()
    ckpt_dir = cli.checkpoints_dir

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*_best.pt")))
    if not ckpt_files:
        print(f"No *_best.pt files found in '{ckpt_dir}'")
        return
    print(f"Found {len(ckpt_files)} checkpoints\n")

    histories   = {}
    bv_results  = {}

    for ckpt_path in ckpt_files:
        run_tag = os.path.basename(ckpt_path).replace("_best.pt", "")
        print(f"\n{'='*60}")
        print(f"  {run_tag}")
        print(f"{'='*60}")

        # ---- 1. Training curves ----
        hist_path = ckpt_path.replace("_best.pt", "_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[run_tag] = json.load(f)
            curve_out = os.path.join(ckpt_dir, f"{run_tag}_curves.png")
            plot_history(hist_path, out_path=curve_out)
        else:
            print(f"  [WARN] history file not found: {hist_path}")

        if cli.skip_bv:
            continue

        # ---- 2. BV analysis using the args saved inside the checkpoint ----
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_args = ckpt.get("args", {})
        if not saved_args:
            print(f"  [WARN] no saved args in checkpoint, skipping BV")
            continue

        args = SimpleNamespace(**saved_args)
        # override BV-specific settings
        args.bv_runs       = cli.bv_runs
        args.bv_train_ratio= cli.bv_train_ratio
        args.epochs        = cli.bv_epochs
        args.num_workers   = cli.num_workers

        result = bias_variance_analysis(args)
        bv_results[run_tag] = result

        # save per-run BV result
        bv_path = os.path.join(ckpt_dir, f"{run_tag}_bv.json")
        with open(bv_path, "w") as f:
            json.dump(result, f, indent=2)

    # ---- Comparison plots ----
    if histories:
        plot_val_acc_comparison(
            histories,
            out_path=os.path.join(ckpt_dir, "comparison_val_acc.png"),
        )

    if bv_results:
        plot_bv_comparison(
            bv_results,
            out_path=os.path.join(ckpt_dir, "comparison_bv.png"),
        )

        # save combined BV summary
        summary_path = os.path.join(ckpt_dir, "bv_summary.json")
        with open(summary_path, "w") as f:
            json.dump(bv_results, f, indent=2)
        print(f"\nBV summary saved to {summary_path}")

        # print table
        print(f"\n{'Run':<45} {'Error':>8} {'Bias²':>8} {'Variance':>10}")
        print("-" * 75)
        for tag, res in bv_results.items():
            print(f"{tag:<45} {res['error']:>8.4f} {res['bias_sq']:>8.4f} {res['variance']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
