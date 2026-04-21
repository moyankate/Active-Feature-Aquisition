"""
Experiment runner for AFA — Decision Tree base learner variant.

Compares:
  - Uniform baseline
  - SEU-US  (log-gain utility, uniform candidate sampling)
  - SEU-ES  (log-gain utility, error/uncertainty sampling)
  - SEU-Accuracy (accuracy gain, ablation)

Usage:
    python run_experiments.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import load_breast_cancer_data
from seu import run_acquisition, run_fully_observed_baseline

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
N_TRIALS        = 5
MISSING_RATES   = [0.50, 0.70]
SAMPLE_FRACTION = 0.05
BATCH_SIZE      = 10
RESULT_PREFIX   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "results", "results_dt")

STRATEGIES = ["uniform", "seu-us", "seu-es", "seu-accuracy"]
STRATEGY_LABELS = {
    "uniform":      "Uniform",
    "seu-us":       "SEU-US (log-gain)",
    "seu-es":       "SEU-ES (log-gain)",
    "seu-accuracy": "SEU-Accuracy",
}
COLORS = {
    "uniform":      "#888888",
    "seu-us":       "#1f77b4",
    "seu-es":       "#d62728",
    "seu-accuracy": "#ff7f0e",
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def interpolate_to_grid(cost_curve, acc_curve, grid):
    return np.interp(grid, cost_curve, acc_curve)


def run_all(X_full, y, missing_rate, n_trials=N_TRIALS):
    n_missing = int(X_full.shape[0] * X_full.shape[1] * missing_rate)
    grid = np.linspace(0, n_missing, 200)

    results = {}
    for strategy in STRATEGIES:
        print(f"  {strategy} ...", end=" ", flush=True)
        trial_curves = []
        trial_aucs   = []
        for seed in range(n_trials):
            cost_c, acc_c = run_acquisition(
                X_full, y,
                missing_rate=missing_rate,
                strategy=strategy,
                seed=seed,
                sample_fraction=SAMPLE_FRACTION,
                batch_size=BATCH_SIZE,
            )
            interp = interpolate_to_grid(cost_c, acc_c, grid)
            trial_curves.append(interp)
            trial_aucs.append(np.trapz(interp, grid) / grid[-1])

        arr  = np.stack(trial_curves)
        aucs = np.array(trial_aucs)
        results[strategy] = (
            grid,
            arr.mean(axis=0), arr.std(axis=0),
            aucs.mean(), aucs.std()
        )
        print(f"done  final={arr[:,-1].mean():.3f}±{arr[:,-1].std():.3f}  "
              f"AUC={aucs.mean():.4f}±{aucs.std():.4f}")

    return results


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_learning_curves(results_by_rate, ceiling_acc, prefix=RESULT_PREFIX):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mr in zip(axes, MISSING_RATES):
        results = results_by_rate[mr]
        for strategy in STRATEGIES:
            grid, mean_acc, std_acc, *_ = results[strategy]
            ax.plot(grid, mean_acc, label=STRATEGY_LABELS[strategy],
                    color=COLORS[strategy], linewidth=2)
            ax.fill_between(grid, mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=COLORS[strategy])
        ax.axhline(ceiling_acc, color="black", linestyle="--", linewidth=1.5,
                   label=f"Fully Observed ({ceiling_acc:.3f})")
        ax.set_xlabel("Cumulative Acquisition Cost (# features acquired)", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"AFA (DT) — {int(mr*100)}% Missing  (n={N_TRIALS} trials)", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_learning_curves.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_auc_bars(results_by_rate, prefix=RESULT_PREFIX):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    x = np.arange(len(STRATEGIES))
    width = 0.55

    for ax, mr in zip(axes, MISSING_RATES):
        results = results_by_rate[mr]
        means = [results[s][3] for s in STRATEGIES]
        stds  = [results[s][4] for s in STRATEGIES]
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=[COLORS[s] for s in STRATEGIES],
                      error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Mean AUC (normalised)", fontsize=11)
        ax.set_title(f"AUC per Strategy — {int(mr*100)}% Missing  (n={N_TRIALS})", fontsize=12)
        ax.set_ylim(min(means) * 0.97, max(means) * 1.02)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.0003,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fname = f"{prefix}_auc_bars.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_early_stage(results_by_rate, ceiling_acc, X_full, prefix=RESULT_PREFIX):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mr in zip(axes, MISSING_RATES):
        results = results_by_rate[mr]
        total_cost = int(X_full.shape[0] * X_full.shape[1] * mr)
        cutoff = int(0.5 * total_cost)
        for strategy in STRATEGIES:
            grid, mean_acc, std_acc, *_ = results[strategy]
            m = grid <= cutoff
            ax.plot(grid[m], mean_acc[m], label=STRATEGY_LABELS[strategy],
                    color=COLORS[strategy], linewidth=2)
            ax.fill_between(grid[m], (mean_acc-std_acc)[m], (mean_acc+std_acc)[m],
                            alpha=0.15, color=COLORS[strategy])
        ax.axhline(ceiling_acc, color="black", linestyle="--", linewidth=1.5,
                   label=f"Fully Observed ({ceiling_acc:.3f})")
        ax.set_xlabel("Cumulative Acquisition Cost", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"Early Stage (DT) — {int(mr*100)}% Missing (first 50% budget)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_early_stage.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

def print_summary_table(results_by_rate, ceiling_acc, X_full):
    checkpoints = [0.25, 0.50, 0.75, 1.0]
    for mr in MISSING_RATES:
        print(f"\n{'='*72}")
        print(f"Missing rate: {int(mr*100)}%")
        total_cost = int(X_full.shape[0] * X_full.shape[1] * mr)
        header = f"{'Strategy':<22}" + "".join(f"  @{int(c*100)}%cost" for c in checkpoints)
        header += "        AUC"
        print(header)
        print("-" * 72)
        for strategy in STRATEGIES:
            grid, mean_acc, std_acc, mean_auc, std_auc = results_by_rate[mr][strategy]
            row = f"{STRATEGY_LABELS[strategy]:<22}"
            for cp in checkpoints:
                a = np.interp(cp * total_cost, grid, mean_acc)
                row += f"  {a:.4f}  "
            row += f"  {mean_auc:.4f}±{std_auc:.4f}"
            print(row)
        print("-" * 72)
        print(f"{'Fully Observed':<22}  {ceiling_acc:.4f}  (upper bound)")
    print(f"\n{'='*72}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data ...")
    X_full, y = load_breast_cancer_data()

    print(f"Dataset: {X_full.shape[0]} instances, {X_full.shape[1]} features, "
          f"{len(np.unique(y))} classes")

    ceiling_acc = run_fully_observed_baseline(X_full, y)
    print(f"Fully observed upper bound (DT): {ceiling_acc:.4f}")

    results_by_rate = {}
    for mr in MISSING_RATES:
        print(f"\nRunning experiments — {int(mr*100)}% missingness ...")
        results_by_rate[mr] = run_all(X_full, y, mr)

    print("\nGenerating plots ...")
    plot_learning_curves(results_by_rate, ceiling_acc)
    plot_auc_bars(results_by_rate)
    plot_early_stage(results_by_rate, ceiling_acc, X_full)

    print_summary_table(results_by_rate, ceiling_acc, X_full)
    print("\nDone.")
