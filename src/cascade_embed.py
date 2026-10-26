"""Cheap specialists for Ettin's uncertain reviews, compared with the Qwen3-8B cascade (src/cascade.py).

Both use Qwen3-Embedding-0.6B vectors and no LLM:
  knn    - similarity-weighted vote of the k most similar training-split reviews (the review itself excluded)
  logreg - logistic regression on the embeddings, fitted on the training split (out-of-fold for training reviews)
Same protocol as src/cascade.py: route when Ettin's confidence < threshold, choose the threshold on the 5-fold
out-of-fold training predictions, report validation for every threshold, write test predictions.

  .venv/Scripts/python src/cascade_embed.py --run-name cascade_embed
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import THRESHOLDS, cascade, embed, metrics  # noqa: E402
from data import LABEL, ROOT, TEXT, load_splits  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--ettin-run", default="ettin400m_final")
    p.add_argument("--cv-run", default="ettin400m_5fold")
    p.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--C", type=float, default=1.0, help="logistic regression inverse regularisation")
    return p.parse_args()


def knn_vote(e_q, e_pool, y_pool, k, exclude=None):
    """P(positive) = similarity-weighted share of positive labels among the k nearest training reviews."""
    sims = e_q @ e_pool.T
    if exclude is not None:
        sims[np.arange(len(e_q)), exclude] = -np.inf
    nn = np.argsort(-sims, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(sims, nn, 1), 0, None)
    return (w * y_pool[nn]).sum(1) / np.maximum(w.sum(1), 1e-9)


def main(a):
    run_dir = ROOT / "outputs" / "runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2))

    tr, _ = load_splits()
    oof = pd.read_csv(ROOT / "outputs" / "cv" / a.cv_run / "oof_predictions.csv")
    va = pd.read_csv(ROOT / "outputs" / "runs" / a.ettin_run / "val_predictions.csv")
    te = pd.read_csv(ROOT / "outputs" / "runs" / a.ettin_run / "test_check_with_probs.csv")
    pool, y_pool = tr[TEXT].tolist(), tr[LABEL].values.astype(int)
    where = {t: j for j, t in enumerate(pool)}
    if not oof[TEXT].map(where.__contains__).all():
        sys.exit("OOF reviews must all be training-split reviews")

    doc = {"pool": embed(a.embedder, pool, False), "val": embed(a.embedder, va[TEXT].tolist(), False),
           "test": embed(a.embedder, te[TEXT].tolist(), False)}
    qry = {"oof": embed(a.embedder, oof[TEXT].tolist(), True), "val": embed(a.embedder, va[TEXT].tolist(), True),
           "test": embed(a.embedder, te[TEXT].tolist(), True)}

    p = {"knn": {"oof": knn_vote(qry["oof"], doc["pool"], y_pool, a.k, exclude=oof[TEXT].map(where).values),
                 "val": knn_vote(qry["val"], doc["pool"], y_pool, a.k),
                 "test": knn_vote(qry["test"], doc["pool"], y_pool, a.k)}}
    lr = LogisticRegression(C=a.C, max_iter=3000)
    oof_lr = cross_val_predict(lr, doc["pool"], y_pool, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               method="predict_proba")[:, 1]
    lr.fit(doc["pool"], y_pool)
    p["logreg"] = {"oof": oof_lr[oof[TEXT].map(where).values], "val": lr.predict_proba(doc["val"])[:, 1],
                   "test": lr.predict_proba(doc["test"])[:, 1]}

    y_oof, y_va = oof["label"].values, va["label"].values
    base = (va["p_pos"].values > 0.5).astype(int)
    report = {"val_ettin": metrics(y_va, base)}
    for name, pp in p.items():
        rows = []
        for t in THRESHOLDS:
            pred, routed = cascade(oof["p_pos"].values, pp["oof"], t)
            rows.append({"threshold": t, "routed": int(routed.sum()), **metrics(y_oof, pred)})
        best = pd.DataFrame(rows).pipe(lambda d: d.loc[d["macro_f1"].idxmax(), "threshold"])
        vrows = []
        for t in THRESHOLDS:
            pred, routed = cascade(va["p_pos"].values, pp["val"], t)
            fixed, broke = int(((base != y_va) & (pred == y_va)).sum()), int(((base == y_va) & (pred != y_va)).sum())
            vrows.append({"threshold": t, "routed": int(routed.sum()), **metrics(y_va, pred), "fixed": fixed,
                          "broke": broke, "mcnemar_p": round(binomtest(broke, fixed + broke, 0.5).pvalue, 3)
                          if fixed + broke else 1.0})
        order = np.argsort(-(1 - np.maximum(va["p_pos"], 1 - va["p_pos"]).values))[:100]
        pred_t, routed_t = cascade(te["p_pos"].values, pp["test"], best)
        pd.DataFrame({"id": te["id"], LABEL: pred_t}).to_csv(run_dir / f"submission_{name}.csv", index=False)
        report[name] = {"standalone_val": metrics(y_va, (pp["val"] > 0.5).astype(int)),
                        "acc_top100_uncertain": round(((pp["val"][order] > 0.5) == y_va[order]).mean(), 2),
                        "oof": rows, "chosen_threshold": float(best), "val": vrows,
                        "test": {"routed": int(routed_t.sum()),
                                 "changed_vs_ettin": int((pred_t != (te["p_pos"].values > 0.5)).sum())}}
        print(f"\n== {name} == standalone val {report[name]['standalone_val']}, "
              f"acc on 100 most uncertain {report[name]['acc_top100_uncertain']}")
        print("OOF:"); print(pd.DataFrame(rows).to_string(index=False))
        print(f"chosen threshold {best}; validation:"); print(pd.DataFrame(vrows).to_string(index=False))
        print("test:", report[name]["test"])
    print("\nValidation: Ettin", report["val_ettin"])
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
