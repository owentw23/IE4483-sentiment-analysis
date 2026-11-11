"""Shared helpers for the report (f)/(g)/(h) experiments. Reuses the repo's existing infrastructure:

* data: src/data.load_splits(), the same split as every Ettin run (deduplicated, stratified 80/20, seed 42;
  5,748 train / 1,437 validation). Nothing here reads or writes PreProcessingPipeline/.
* outputs: each arm gets outputs/runs/<arm>/{config.json, metrics.json, val_predictions.csv}, the layout
  src/train.py uses, so `python src/evaluate.py --run <arm>` works on any arm unchanged.
* ledger: one row per arm in outputs/results_f.csv. outputs/results.csv stays the transformer-sweep ledger
  written by src/train.py (its columns are training hyperparameters that most arms here don't have).

Run one arm per process (see src/fgh/run_f.sh) so the peak-memory numbers belong to that arm alone.
"""
import json
import platform
import re
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import LABEL, ROOT, TEXT, load_splits  # noqa: E402,F401

RUNS = ROOT / "outputs" / "runs"          # same per-run folders as src/train.py
LEDGER = ROOT / "outputs" / "results_f.csv"
REPORT = ROOT / "outputs" / "report_f"    # Table F1 and figures (src/fgh/report_f.py)
HOTEL = ROOT / "data" / "external" / "hotel"   # (g) hotel parts built by src/fgh/hotel_data.py (git-ignored)
LEDGER_G = ROOT / "outputs" / "results_g.csv"
REPORT_G = ROOT / "outputs" / "report_g"
REFERENCE = ROOT / "outputs" / "runs" / "ettin400m_final" / "val_predictions.csv"  # the submitted model
CONTRAST = re.compile(r"\b(?:but|however|although|though|unfortunately|except|yet)\b", re.IGNORECASE)

# Every logistic-regression head in (f) chooses its C the same way (decided 2026-11-08; sklearn's default C = 1.0
# made several arms predict almost everything positive).
C_FOLDS = StratifiedKFold(5, shuffle=True, random_state=42)
C_SELECTION = ("5-fold stratified CV on the training split (seed 42), metric macro-F1; every step before the "
               "classifier (vectoriser or scaler) is refitted inside each fold")
DENSE_C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)  # standardised dense features


def splits():
    """The fixed protocol split; fails loudly if it ever drifts."""
    tr, va = load_splits()
    assert (len(tr), len(va)) == (5748, 1437), (len(tr), len(va))
    return tr, va


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "neg_f1": round(f[0], 4), "neg_precision": round(p[0], 4), "neg_recall": round(r[0], 4)}


def hardware():
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "gpu": gpu, "os": platform.platform(), "python": platform.python_version()}


def peak_rss_gb():
    """Peak resident memory of this process (ru_maxrss is KB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)


class Timer:
    """with timer("fit"): ...  ->  timer.t["fit"] = seconds"""

    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        yield
        self.t[name] = round(time.perf_counter() - t0, 3)


def select_c(steps, x, y, grid, class_weight=None, max_iter=5000, n_jobs=3):
    """Fit Pipeline(*steps, LogisticRegression) with C chosen by C_FOLDS cross-validation on (x, y) only.

    The steps sit inside the pipeline, so they are refitted on each fold's training part and no held-out text or
    statistic leaks into them. The best pipeline is then refitted on all of (x, y). n_jobs=3: the server has
    6 CPUs and 13.6 GB RAM, and 6 workers made joblib lose a worker.
    """
    pipe = Pipeline([*steps, ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight))])
    search = GridSearchCV(pipe, {"clf__C": list(grid)}, scoring="f1_macro", cv=C_FOLDS, n_jobs=n_jobs, refit=True)
    search.fit(x, y)
    return search


def c_record(search, grid):
    """(config, extra) entries saying how C was chosen, incl. the CV curve and whether C hit the grid edge."""
    c = search.best_params_["clf__C"]
    config = {"C_selection": C_SELECTION, "C_grid": list(grid), "chosen_C": c}
    extra = {"chosen_C": c, "chosen_C_at_grid_edge": c in (grid[0], grid[-1]),
             "cv_macro_f1_best": round(float(search.best_score_), 4),
             "cv_macro_f1_by_C": {f"{g:g}": round(float(s), 4)
                                  for g, s in zip(grid, search.cv_results_["mean_test_score"])}}
    return config, extra


def mcnemar_vs_reference(va, pred):
    """Exact McNemar test against the submitted Ettin model on the same 1,437 reviews."""
    if not REFERENCE.exists():
        return {}
    ref = pd.read_csv(REFERENCE).set_index(TEXT)["pred"]
    base = va[TEXT].map(ref)
    if base.isna().any():
        return {}
    y, base = va[LABEL].values.astype(int), base.values.astype(int)
    only_arm = int(((base == y) & (pred != y)).sum())
    only_ettin = int(((base != y) & (pred == y)).sum())
    n = only_arm + only_ettin
    return {"only_arm_wrong": only_arm, "only_ettin_wrong": only_ettin,
            "mcnemar_p": round(binomtest(only_arm, n, 0.5).pvalue, 4) if n else 1.0}


def mcnemar_counts(y, pred_arm, pred_ref):
    """Exact McNemar test of two prediction vectors on the same reviews."""
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_arm, pred_ref))
    only_arm, only_ref = int(((b == y) & (a != y)).sum()), int(((b != y) & (a == y)).sum())
    n = only_arm + only_ref
    return {"only_arm_wrong": only_arm, "only_ref_wrong": only_ref,
            "mcnemar_p": round(binomtest(only_arm, n, 0.5).pvalue, 4) if n else 1.0}


# ---------------------------------------------------------------- (g) hotel project
def hotel_text(part):
    """Text of one hotel part (eval / stress3 / pool / fewshot). Methods only ever see these columns."""
    df = pd.read_csv(HOTEL / f"{part}_text.csv" if part != "fewshot" else HOTEL / "fewshot.csv",
                     keep_default_na=False)
    return df


def hotel_key(part):
    """The hidden answer key (stars and label). Read only by the scoring code in save_g_arm."""
    return pd.read_csv(HOTEL / f"{part}_key.csv")


def hotel_inputs(df, mode="text"):
    """Model input: review text only (default, products have no title) or the dataset's title + newline + text."""
    if mode == "title_text":
        return (df["title"].astype(str) + "\n" + df["text"].astype(str)).tolist()
    return df["text"].astype(str).tolist()


G1_ARM = "g1_ettin_final"  # the submitted model on hotels: the reference for hotel McNemar tests


def save_g_arm(arm, *, config, product_p, hotel_p, stress_p, pool_p=None, cost, extra=None):
    """Write one (g) arm and upsert its row in outputs/results_g.csv. Decision rule: P(positive) >= 0.5.

    product_p: P(positive) on the 1,437 product validation reviews (same order as splits()[1]).
    hotel_p / stress_p / pool_p: P(positive) on hotel_text("eval" / "stress3" / "pool"), in file order.
    Files: val_predictions.csv (product, src/train.py format), hotel_predictions.csv, stress3_predictions.csv,
    pool_predictions.csv (optional; teacher input for self-training), config.json, metrics.json.
    """
    d = RUNS / arm
    d.mkdir(parents=True, exist_ok=True)
    _, va = splits()
    y_va = va[LABEL].values.astype(int)
    product_p = np.asarray(product_p, dtype=float)
    pred_va = (product_p >= 0.5).astype(int)
    m_prod = metrics(y_va, pred_va)
    mask = va[TEXT].str.contains(CONTRAST).values
    m_prod["macro_f1_contrast"] = round(f1_score(y_va[mask], pred_va[mask], average="macro"), 4)
    m_prod.update(mcnemar_vs_reference(va, pred_va))

    ev_text, ev_key = hotel_text("eval"), hotel_key("eval")
    assert (ev_text["review_id"].values == ev_key["review_id"].values).all()
    y_h = ev_key["label"].values.astype(int)
    hotel_p = np.asarray(hotel_p, dtype=float)
    pred_h = (hotel_p >= 0.5).astype(int)
    m_hotel = metrics(y_h, pred_h)
    m_hotel["predicted_negative_share"] = round(float((pred_h == 0).mean()), 4)
    m_hotel["true_negative_share"] = round(float((y_h == 0).mean()), 4)
    ref = RUNS / G1_ARM / "hotel_predictions.csv"
    if arm != G1_ARM and ref.exists():
        r = pd.read_csv(ref)
        if (r["review_id"].values == ev_key["review_id"].values).all():
            m_hotel["vs_g1"] = mcnemar_counts(y_h, pred_h, r["pred"].values)

    st_key = hotel_key("stress3")
    stress_p = np.asarray(stress_p, dtype=float)
    m_stress = {"n": len(stress_p), "share_predicted_positive": round(float((stress_p >= 0.5).mean()), 4),
                "mean_p_pos": round(float(stress_p.mean()), 4)}

    m = {"product": m_prod, "hotel": m_hotel, "stress3": m_stress,
         "drop_macro_f1": round(m_prod["macro_f1"] - m_hotel["macro_f1"], 4),
         "drop_accuracy": round(m_prod["accuracy"] - m_hotel["accuracy"], 4), **cost, **(extra or {}),
         "hardware": hardware()}
    pd.DataFrame({TEXT: va[TEXT].values, "label": y_va, "pred": pred_va, "p_pos": np.round(product_p, 6)}) \
        .to_csv(d / "val_predictions.csv", index=False)
    pd.DataFrame({"review_id": ev_key["review_id"], "label": y_h, "pred": pred_h, "p_pos": np.round(hotel_p, 6)}) \
        .to_csv(d / "hotel_predictions.csv", index=False)
    pd.DataFrame({"review_id": st_key["review_id"], "pred": (stress_p >= 0.5).astype(int),
                  "p_pos": np.round(stress_p, 6)}).to_csv(d / "stress3_predictions.csv", index=False)
    if pool_p is not None:
        pool = hotel_text("pool")
        pd.DataFrame({"review_id": pool["review_id"], "p_pos": np.round(np.asarray(pool_p, dtype=float), 6)}) \
            .to_csv(d / "pool_predictions.csv", index=False)
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (d / "metrics.json").write_text(json.dumps(m, indent=2, default=str))

    row = {"arm": arm,
           **{f"product_{k}": v for k, v in m_prod.items() if not isinstance(v, (dict, list))},
           **{f"hotel_{k}": v for k, v in m_hotel.items() if not isinstance(v, (dict, list))},
           "hotel_mcnemar_p_vs_g1": m_hotel.get("vs_g1", {}).get("mcnemar_p"),
           "drop_macro_f1": m["drop_macro_f1"], "stress3_share_pos": m_stress["share_predicted_positive"],
           **{k: v for k, v in cost.items() if not isinstance(v, (dict, list))},
           "gpu": m["hardware"]["gpu"]}
    old = pd.read_csv(LEDGER_G) if LEDGER_G.exists() else pd.DataFrame(columns=["arm"])
    pd.concat([old[old["arm"] != arm], pd.DataFrame([row])], ignore_index=True).to_csv(LEDGER_G, index=False)
    print(json.dumps({"arm": arm, **{k: v for k, v in m.items() if k != "hardware"}}, indent=1, default=str))
    return m


def save_arm(arm, va, p_pos, *, config, cost, extra=None):
    """Write one arm's files and upsert its row in outputs/results_f.csv. Decision rule: P(positive) >= 0.5."""
    d = RUNS / arm
    d.mkdir(parents=True, exist_ok=True)
    y = va[LABEL].values.astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    pred = (p_pos >= 0.5).astype(int)
    m = metrics(y, pred)
    mask = va[TEXT].str.contains(CONTRAST).values
    m["macro_f1_contrast"] = round(f1_score(y[mask], pred[mask], average="macro"), 4)
    m["n_contrast"] = int(mask.sum())
    m.update(mcnemar_vs_reference(va, pred))
    m.update(cost)
    if extra:
        m.update(extra)
    m["hardware"] = hardware()
    pd.DataFrame({TEXT: va[TEXT].values, "label": y, "pred": pred, "p_pos": np.round(p_pos, 6)}) \
        .to_csv(d / "val_predictions.csv", index=False)
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (d / "metrics.json").write_text(json.dumps(m, indent=2, default=str))

    row = {"arm": arm, **{k: v for k, v in m.items() if not isinstance(v, (dict, list))},
           "cpu": m["hardware"]["cpu"], "gpu": m["hardware"]["gpu"]}
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=["arm"])
    pd.concat([old[old["arm"] != arm], pd.DataFrame([row])], ignore_index=True).to_csv(LEDGER, index=False)
    print(json.dumps({"arm": arm, **{k: v for k, v in m.items() if k != "hardware"}}, indent=1))
    return m
