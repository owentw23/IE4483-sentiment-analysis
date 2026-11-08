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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import LABEL, ROOT, TEXT, load_splits  # noqa: E402,F401

RUNS = ROOT / "outputs" / "runs"          # same per-run folders as src/train.py
LEDGER = ROOT / "outputs" / "results_f.csv"
REPORT = ROOT / "outputs" / "report_f"    # Table F1 and figures (src/fgh/report_f.py)
REFERENCE = ROOT / "outputs" / "runs" / "ettin400m_final" / "val_predictions.csv"  # the submitted model
CONTRAST = re.compile(r"\b(but|however|although|though|unfortunately|except|yet)\b", re.IGNORECASE)


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
