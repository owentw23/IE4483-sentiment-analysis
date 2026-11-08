"""(f) Build Table F1 and the two figures from outputs/results_f.csv plus the fine-tuned Ettin runs.

  python src/fgh/report_f.py
Writes outputs/report_f/table_f1.csv, table_f1.md, f1_cost_vs_accuracy.png and f2_dimension_vs_accuracy.png.
No model or GPU needed.
Fine-tuned Ettin rows come from outputs/runs/<run>/metrics.json (src/train.py), labelled with where they ran.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LEDGER, REPORT, ROOT  # noqa: E402

COLUMNS = ["arm", "accuracy", "macro_f1", "neg_f1", "macro_f1_contrast", "feature_dim", "trainable_params",
           "fit_seconds", "predict_seconds_per_1k", "peak_rss_gb", "peak_gpu_gb", "model_size_mb", "mcnemar_p",
           "gpu", "cpu"]


def ettin_rows(runs):
    rows = []
    for run, (arm, where) in runs.items():
        f = ROOT / "outputs" / "runs" / run / "metrics.json"
        if not f.exists():
            print(f"skip {run}: no metrics.json")
            continue
        m = json.loads(f.read_text())
        rows.append({"arm": arm, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "neg_f1": m["neg_f1"],
                     "feature_dim": 1024 if "400m" in run else 768,
                     "trainable_params": round(m.get("params_m", 0) * 1e6),
                     "fit_seconds": round(m["train_minutes"] * 60, 1),
                     "predict_seconds_per_1k": round(m["val_seconds"] / m["n_val"] * 1000, 3),
                     "peak_gpu_gb": m.get("peak_gpu_gb"), "model_size_mb": round(m.get("params_m", 0) * 4, 1),
                     "gpu": where})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f11-run", default="ettin150m_lr2e-5")
    p.add_argument("--f11-hardware", default="RTX 4090 (rented server)")
    a = p.parse_args()
    arms = pd.read_csv(LEDGER)
    extra = ettin_rows({"ettin400m_final": ("f8_ettin400m_finetuned (submitted)", "RTX 4060 Laptop 8 GB"),
                        a.f11_run: ("f11_ettin150m_finetuned", a.f11_hardware)})
    table = pd.concat([arms, pd.DataFrame(extra)], ignore_index=True)
    table = table[[c for c in COLUMNS if c in table.columns]].sort_values("macro_f1").reset_index(drop=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT / "table_f1.csv", index=False)
    (REPORT / "table_f1.md").write_text(table.to_markdown(index=False))
    print(table.to_string(index=False))

    figs = REPORT
    timed = table[table["fit_seconds"].fillna(0) > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    size = 20 + 40 * np.log10(timed["trainable_params"].clip(lower=10))
    ax.scatter(timed["fit_seconds"], timed["macro_f1"], s=size, alpha=0.7)
    for _, r in timed.iterrows():
        ax.annotate(r["arm"].split("_", 1)[0], (r["fit_seconds"], r["macro_f1"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log scale)")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Feature format: training cost vs accuracy (point size ~ log trainable parameters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "f1_cost_vs_accuracy.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    tf = table[table["arm"].str.startswith(("f3_", "f2_tfidf_12gram_balanced"))].sort_values("feature_dim")
    gl = table[table["arm"].str.startswith("f4_")].sort_values("feature_dim")
    ax.plot(tf["feature_dim"], tf["macro_f1"], "o-", label="TF-IDF 1–2-gram (vocabulary size)")
    ax.plot(gl["feature_dim"], gl["macro_f1"], "s-", label="GloVe mean (vector size)")
    ax.set_xscale("log")
    ax.set_xlabel("feature dimension (log scale)")
    ax.set_ylabel("validation macro-F1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "f2_dimension_vs_accuracy.png", dpi=200)
    print(f"wrote Table F1 and both figures to {REPORT}")


if __name__ == "__main__":
    main()
