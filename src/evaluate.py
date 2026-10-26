"""Report figures and tables from a run's validation predictions (questions b, d, e).

  .venv/Scripts/python src/evaluate.py --run ettin400m_final [--out-dir docs/figures/ettin400m_final]

Reads outputs/runs/<run>/val_predictions.csv (columns reviews, label, pred, p_pos) and writes:
  confusion.png    counts and row-normalised confusion matrix
  roc.png          ROC curve with AUC
  pr.png           precision-recall curve for the NEGATIVE class (the minority class we report F1 for)
  calibration.png  reliability diagram (10 equal-width bins) + histogram of P(positive)
  per_class.csv    precision / recall / F1 / support per class
  summary.json     accuracy, macro-F1, ROC-AUC, PR-AUC (negative), Brier score, ECE
No model or GPU needed.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT  # noqa: E402

NAMES = ["negative (0)", "positive (1)"]
N_BINS = 10


def expected_calibration_error(y, p_pos, n_bins=N_BINS):
    """Weighted mean |accuracy - confidence| over equal-width bins of P(positive)."""
    bins = np.minimum((p_pos * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p_pos[m].mean())
    return float(ece)


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, mat, title, fmt in [(axes[0], cm, "Counts", "{:d}"), (axes[1], norm, "Row-normalised (recall)", "{:.3f}")]:
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=12)
        ax.set_xticks([0, 1], NAMES)
        ax.set_yticks([0, 1], NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(y, p_pos, auc, path):
    fpr, tpr, _ = roc_curve(y, p_pos)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (positive = class 1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_negative(y, p_pos, ap, path):
    y_neg, s_neg = 1 - y, 1 - p_pos
    prec, rec, _ = precision_recall_curve(y_neg, s_neg)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(rec, prec, lw=2, label=f"Model (AP = {ap:.4f})")
    ax.axhline(y_neg.mean(), ls="--", c="grey", label=f"Chance (prevalence {y_neg.mean():.3f})")
    ax.set_xlabel("Recall (negative)")
    ax.set_ylabel("Precision (negative)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall, negative class")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_calibration(y, p_pos, brier, ece, path):
    frac_pos, mean_pred = calibration_curve(y, p_pos, n_bins=N_BINS, strategy="uniform")
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(4.8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=f"Model (Brier {brier:.4f}, ECE {ece:.4f})")
    counts = np.histogram(p_pos, bins=N_BINS, range=(0, 1))[0]
    for x, f, n in zip(mean_pred, frac_pos, counts[counts > 0]):  # calibration_curve drops empty bins
        ax.annotate(f"n={n}", (x, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_ylabel("Fraction actually positive")
    ax.set_title("Reliability diagram (n = reviews per bin)")
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.05, 1.3)  # headroom so the legend clears the n= labels
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper left", fontsize=8)
    hx.hist([p_pos[y == 0], p_pos[y == 1]], bins=20, range=(0, 1), stacked=True, label=NAMES)
    hx.set_yscale("log")
    hx.set_xlabel("Predicted P(positive)")
    hx.set_ylabel("Reviews (log)")
    hx.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name under outputs/runs/")
    p.add_argument("--out-dir", default=None, help="default: docs/figures/<run>")
    a = p.parse_args()

    df = pd.read_csv(ROOT / "outputs" / "runs" / a.run / "val_predictions.csv")
    y, pred, p_pos = df["label"].to_numpy(), df["pred"].to_numpy(), df["p_pos"].to_numpy()
    out = Path(a.out_dir) if a.out_dir else ROOT / "docs" / "figures" / a.run
    out.mkdir(parents=True, exist_ok=True)

    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    per_class = pd.DataFrame({"class": NAMES, "precision": prec, "recall": rec, "f1": f1, "support": sup}).round(4)
    per_class.to_csv(out / "per_class.csv", index=False)

    roc_auc = roc_auc_score(y, p_pos)
    pr_auc_neg = average_precision_score(1 - y, 1 - p_pos)
    brier = brier_score_loss(y, p_pos)
    ece = expected_calibration_error(y, p_pos)
    summary = {
        "run": a.run, "n_val": int(len(df)),
        "accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "roc_auc": round(roc_auc, 4), "pr_auc_negative": round(pr_auc_neg, 4),
        "brier": round(brier, 4), "ece": round(ece, 4),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_confusion(y, pred, out / "confusion.png")
    plot_roc(y, p_pos, roc_auc, out / "roc.png")
    plot_pr_negative(y, p_pos, pr_auc_neg, out / "pr.png")
    plot_calibration(y, p_pos, brier, ece, out / "calibration.png")

    print(per_class.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main()
