"""(f) Build Table F1 and the two figures from outputs/results_f.csv plus the fine-tuned Ettin runs.

  python src/fgh/report_f.py
Writes outputs/report_f/table_f1.csv, table_f1.md, f1_cost_vs_accuracy.png and f2_dimension_vs_accuracy.png,
plus table_contrast.csv / .md (report Table 11), table_pairs.csv (pairwise McNemar tests cited in the report),
table_disk.csv (disk space each run needs to predict) and f3_c_vs_macro_f1.png (report Figure 3). No model or GPU
needed, but disk sizes are measured from the files on this machine (Hugging Face cache, GloVe files, saved models).
Fine-tuned Ettin rows come from outputs/runs/<run>/metrics.json (src/train.py), labelled with where they ran.
Table contrast re-scores every arm's outputs/runs/<run>/val_predictions.csv on the validation reviews with and
without a contrast word (common.CONTRAST), and tests each arm against the reference model on both subsets
(exact McNemar). The reference for the (f) report is the final Ettin recipe retrained on the rented RTX 3090
(--reference-run, default ettin400m_final_3090), so every (f) number comes from one machine; table_f1.csv's
mcnemar_p is recomputed against it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.metrics import f1_score  # noqa: E402

from common import CONTRAST, LEDGER, REFERENCE, REPORT, ROOT, RUNS, TEXT, mcnemar_counts  # noqa: E402

COLUMNS = ["arm", "accuracy", "macro_f1", "neg_f1", "macro_f1_contrast", "chosen_C", "feature_dim",
           "trainable_params", "fit_seconds", "c_search_seconds", "predict_seconds_per_1k", "peak_rss_gb",
           "peak_gpu_gb", "model_size_mb", "disk_mb", "mcnemar_p", "gpu", "cpu"]
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
GLOVE_DIR = ROOT / "data" / "external" / "glove"
F8B_RUN = "ettin400m_final_3090"   # Owen's ettin400m_final recipe, retrained unchanged on the rented RTX 3090


def checkpoint_bytes(repo):
    """Bytes of a cached Hugging Face model's weight files (safetensors if present, else .bin)."""
    snaps = sorted((HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"), key=lambda q: q.stat().st_mtime)
    files = list(snaps[-1].iterdir())
    weights = [f for f in files if f.suffix == ".safetensors"] or [f for f in files if f.name == "pytorch_model.bin"]
    assert weights, f"no weights cached for {repo}"
    return sum(f.resolve().stat().st_size for f in weights)


def saved_model_bytes(run):
    """Bytes of the weights src/train.py saved with --save-model (fails on a Git LFS pointer)."""
    total = sum(f.stat().st_size for f in (RUNS / run / "model").glob("*.safetensors"))
    assert total > 10_000_000, f"{run}: weights missing or only a Git LFS pointer"
    return total


def disk_row(arm, run):
    """What a run must keep on disk to predict new reviews, and how many bytes that is."""
    m = json.loads((RUNS / run / "metrics.json").read_text())
    cfg = json.loads((RUNS / run / "config.json").read_text())
    if arm.startswith(("f1_", "f2_", "f3_")):
        return "saved vectoriser and classifier (pickle)", m["model_size_mb"] * 1e6, "measured"
    if arm.startswith("f4_"):
        dim = int(arm.split("glove")[1].split("_")[0])
        return f"GloVe {dim}-d text file plus classifier", (GLOVE_DIR / f"glove.6B.{dim}d.txt").stat().st_size, "measured"
    if arm.startswith("f5"):
        return "BiLSTM weights incl. vocabulary embeddings, 32-bit", m["model_size_mb"] * 1e6, "parameters x 4 bytes"
    if arm.startswith(("f6_", "f7_", "f10_")):
        return f"pretrained checkpoint {cfg['model']} plus classifier", checkpoint_bytes(cfg["model"]), "measured"
    if arm.startswith(("f8", "f9_")):
        source = F8B_RUN if arm.startswith(("f8b", "f9_ettin3090")) else "ettin400m_final"
        return f"fine-tuned weights saved in outputs/runs/{source}/model", saved_model_bytes(source), "measured"
    if arm.startswith("f11"):
        return "fine-tuned weights (not saved), 32-bit", m["params_m"] * 4e6, "parameters x 4 bytes"
    raise ValueError(arm)


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


def c_figure(path):
    """Report Figure 3: cross-validated macro-F1 at each C (saved by select_c), plus the default-C frozen Ettin run.

    The curves are 5-fold CV scores on the training split; the single point is the earlier validation run with
    C = 1 and features only L2-normalised (outputs/runs/f7_ettin400m_frozen_C1), which predicted every review positive.
    """
    def curve(run):
        m = json.loads((RUNS / run / "metrics.json").read_text())
        cs = sorted(m["cv_macro_f1_by_C"], key=float)
        return np.array([float(c) for c in cs]), np.array([m["cv_macro_f1_by_C"][c] for c in cs]), float(m["chosen_C"])
    series = [("f2_tfidf_12gram", "TF-IDF, unweighted", "#2a78d6", "--", "o"),
              ("f2_tfidf_12gram_balanced", "TF-IDF, balanced", "#2a78d6", "-", "o"),
              ("f7_ettin400m_frozen", "Frozen Ettin, standardised", "#1baf7a", "-", "D")]
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    old = json.loads((RUNS / "f7_ettin400m_frozen_C1" / "metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.axhline(old["macro_f1"], color="0.55", lw=1, ls=":", zorder=1)
    ax.text(5e3, old["macro_f1"] + 0.008, f"every review predicted positive ({old['macro_f1']:.3f})", fontsize=8,
            color="0.4", ha="right")
    ax.axvline(1, color="0.55", lw=1, ls=":", zorder=1)
    ax.text(1.12, 0.93, "scikit-learn default C = 1", fontsize=8, color="0.4", va="top")
    for run, label, color, ls, mk in series:
        c, f, chosen = curve(run)
        ax.plot(c, f, ls=ls, marker=mk, color=color, lw=2, ms=5, label=label, zorder=3)
        ax.scatter([chosen], [f[c == chosen][0]], s=150, facecolors="none", edgecolors=color, lw=1.5, zorder=4)
        below = run != "f2_tfidf_12gram"          # unweighted curve rises steeply right of C = 1: label it on the left
        ax.annotate(f"{f[c == 1][0]:.3f}", (1, f[c == 1][0]), xytext=(6, -12) if below else (-7, 0),
                    textcoords="offset points", ha="left" if below else "right", va="center", fontsize=8.5,
                    path_effects=halo, zorder=6)
    ax.plot([1], [old["macro_f1"]], marker="D", ms=7, mfc="white", mec="#1baf7a", mew=2, ls="none", zorder=5,
            label="Frozen Ettin, not standardised (validation)")
    ax.scatter([], [], s=150, facecolors="none", edgecolors="0.35", lw=1.5, label="C chosen by cross-validation")
    ax.set_xscale("log")
    ax.set_xlim(1.5e-4, 6e3)
    ax.set_ylim(0.42, 0.94)
    ax.set_xlabel("regularisation strength C (log scale; larger C = weaker penalty)")
    ax.set_ylabel("macro-F1, 5-fold cross-validation")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.36), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def pairs_for(ref_run, f9):
    """Pairwise tests cited in the (f) report; ref_run is the fine-tuned Ettin-400M reference, f9 its length prefix."""
    return [("f5_bilstm_glove300_frozen", "f5b_bilstm_scratch"),   # pretrained vs learned word vectors
            ("f7_ettin400m_frozen", ref_run),                      # frozen vs fine-tuned Ettin-400M
            ("ettin150m_lr2e-5", ref_run),                         # Ettin-150M vs Ettin-400M
            (f"{f9}_len256", f"{f9}_len512"),                      # input length, same weights, same GPU
            (f"{f9}_len128", f"{f9}_len512")]


def pair_rows(pairs):
    """Exact McNemar test and the number of changed predictions for each (run a, run b) on the 1,437 reviews."""
    rows = []
    for a, b in pairs:
        pa = pd.read_csv(RUNS / a / "val_predictions.csv").set_index(TEXT)
        pb = pd.read_csv(RUNS / b / "val_predictions.csv").set_index(TEXT).reindex(pa.index)
        assert pb["pred"].notna().all() and (pa["label"] == pb["label"]).all(), f"{a} / {b}: different reviews"
        y, xa, xb = (v.values.astype(int) for v in (pa["label"], pa["pred"], pb["pred"]))
        mc = mcnemar_counts(y, xa, xb)
        rows.append({"run_a": a, "run_b": b, "n": len(y), "macro_f1_a": round(f1_score(y, xa, average="macro"), 4),
                     "macro_f1_b": round(f1_score(y, xb, average="macro"), 4),
                     "predictions_changed": int((xa != xb).sum()), "only_a_wrong": mc["only_arm_wrong"],
                     "only_b_wrong": mc["only_ref_wrong"], "mcnemar_p": mc["mcnemar_p"]})
    return pd.DataFrame(rows)


def contrast_rows(runs, ref_run):
    """Macro-F1 on the validation reviews with and without a contrast word, for each {arm: run folder}."""
    ref = pd.read_csv(RUNS / ref_run / "val_predictions.csv").set_index(TEXT)
    has = np.asarray(ref.index.str.contains(CONTRAST), dtype=bool)
    y, base = ref["label"].values.astype(int), ref["pred"].values.astype(int)   # val_predictions.csv columns
    rows = []
    for arm, run in runs.items():
        p = pd.read_csv(RUNS / run / "val_predictions.csv").set_index(TEXT).reindex(ref.index)
        assert p["pred"].notna().all() and (p["label"].values == y).all(), f"{run}: not the 1,437 validation reviews"
        pred = p["pred"].values.astype(int)
        row = {"arm": arm, "run": run, "n_contrast": int(has.sum()), "n_contrast_neg": int((y[has] == 0).sum()),
               "n_no_contrast": int((~has).sum()), "n_no_contrast_neg": int((y[~has] == 0).sum()),
               "macro_f1": round(f1_score(y, pred, average="macro"), 4),
               "macro_f1_contrast": round(f1_score(y[has], pred[has], average="macro"), 4),
               "macro_f1_no_contrast": round(f1_score(y[~has], pred[~has], average="macro"), 4),
               "wrong_contrast": int((pred[has] != y[has]).sum()),
               "wrong_no_contrast": int((pred[~has] != y[~has]).sum())}
        if run != ref_run:
            row["mcnemar_p"] = mcnemar_counts(y, pred, base)["mcnemar_p"]
            row["contrast_mcnemar_p"] = mcnemar_counts(y[has], pred[has], base[has])["mcnemar_p"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f11-run", default="ettin150m_lr2e-5")
    p.add_argument("--f11-hardware", default="RTX 3090 24 GB (rented server)")
    p.add_argument("--reference-run", default=F8B_RUN, help="fine-tuned Ettin run every McNemar test compares with")
    a = p.parse_args()
    f9 = "f9_ettin3090" if a.reference_run == F8B_RUN else "f9_ettin_final"
    arms = pd.read_csv(LEDGER)
    f8, f11 = "f8_ettin400m_finetuned (submitted)", "f11_ettin150m_finetuned"
    f8b = "f8b_ettin400m_finetuned_3090"
    extra = ettin_rows({"ettin400m_final": (f8, "RTX 4060 Laptop 8 GB"), a.f11_run: (f11, a.f11_hardware),
                        F8B_RUN: (f8b, "RTX 3090 24 GB (rented server)")})
    table = pd.concat([arms, pd.DataFrame(extra)], ignore_index=True)
    table = table[[c for c in COLUMNS if c in table.columns]].sort_values("macro_f1").reset_index(drop=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    runs = {arm: arm for arm in arms["arm"]} | {f8: REFERENCE.parent.name, f11: a.f11_run}
    if (RUNS / F8B_RUN / "val_predictions.csv").exists():
        runs[f8b] = F8B_RUN
    disk = pd.DataFrame([(arm, run, *disk_row(arm, run)) for arm, run in runs.items()],
                        columns=["arm", "run", "stored", "bytes", "how"])
    disk["disk_mb"] = (disk["bytes"] / 1e6).round(2)
    disk.drop(columns="bytes").to_csv(REPORT / "table_disk.csv", index=False)
    table = table.merge(disk[["arm", "disk_mb"]], on="arm", how="left")
    table = table[[c for c in COLUMNS if c in table.columns]]
    contrast = contrast_rows(runs, a.reference_run).sort_values("macro_f1_contrast").reset_index(drop=True)
    contrast.to_csv(REPORT / "table_contrast.csv", index=False)
    (REPORT / "table_contrast.md").write_text(contrast.to_markdown(index=False))
    print(contrast.to_string(index=False))
    pairs = pair_rows(pairs_for(a.reference_run, f9) + [(F8B_RUN, "ettin400m_final")])
    pairs.to_csv(REPORT / "table_pairs.csv", index=False)
    print(pairs.to_string(index=False))
    by_arm = contrast.set_index("arm")
    for col in ("macro_f1", "macro_f1_contrast"):                # ledger rows must agree with the re-score
        known = table.set_index("arm")[col].dropna()
        known = known[known.index.isin(by_arm.index) & ~known.index.isin([f8, f11, f8b])]
        diff = (known - by_arm.loc[known.index, col]).abs().max()
        assert diff < 1e-3, f"{col}: ledger and val_predictions.csv disagree by {diff}"
    for arm in (f8, f11, f8b):                                     # src/train.py runs do not record the contrast score
        if arm in by_arm.index:
            table.loc[table["arm"] == arm, "macro_f1_contrast"] = by_arm.loc[arm, "macro_f1_contrast"]
    table["mcnemar_p"] = table["arm"].map(by_arm["mcnemar_p"])     # every arm against the reference run
    table["mcnemar_reference"] = a.reference_run
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

    fig, ax = plt.subplots(figsize=(6.5, 4.2))   # report Figure 2: values are labelled because print has no hover
    tf = table[table["arm"].str.startswith(("f3_", "f2_tfidf_12gram_balanced"))].sort_values("feature_dim")
    gl = table[table["arm"].str.startswith("f4_")].sort_values("feature_dim")
    ax.plot(tf["feature_dim"], tf["macro_f1"], "o-", color="#2a78d6", lw=2, ms=6,
            label="TF-IDF, words and word pairs, balanced: vocabulary size")
    ax.plot(gl["feature_dim"], gl["macro_f1"], "s-", color="#eb6834", lw=2, ms=6, label="Mean GloVe vector: vector size")
    for i, (x, y) in enumerate(zip(tf["feature_dim"], tf["macro_f1"])):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -15) if i == 0 else (0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5)
    full = tf.iloc[-1]
    ax.annotate(f"all {full['feature_dim']:,.0f}", (full["feature_dim"], full["macro_f1"]), xytext=(0, -15),
                textcoords="offset points", ha="center", fontsize=8.5, color="0.35")
    for x, y in zip(gl["feature_dim"], gl["macro_f1"]):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(25, 3e5)
    ax.set_xlabel("number of features (log scale)")
    ax.set_ylabel("validation macro-F1")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "f2_dimension_vs_accuracy.png", dpi=200)
    c_figure(figs / "f3_c_vs_macro_f1.png")
    print(f"wrote Table F1, the contrast table and three figures to {REPORT}")


if __name__ == "__main__":
    main()
