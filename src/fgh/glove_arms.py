"""(f) Static word vectors: average the GloVe vectors of a review's words, then logistic regression.

  python src/fgh/glove_arms.py --dim 50      # also 100, 300
First use downloads glove.6B.zip (822 MB; Wikipedia 2014 + Gigaword 5, 400k lower-case words) into
data/external/glove/ (git-ignored): from Stanford NLP's own Hugging Face mirror (stanfordnlp/glove), falling back
to nlp.stanford.edu, which some cloud hosts cannot reach. Words missing from GloVe are skipped; a review with no known word
gets the zero vector. Also reports how much of the training vocabulary GloVe covers.
Classifier: StandardScaler + LogisticRegression, C chosen by 5-fold CV on the training split (common.select_c),
the same rule as every other logistic-regression head in (f).
"""
import argparse
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DENSE_C_GRID, LABEL, ROOT, TEXT, Timer, c_record, peak_rss_gb, save_arm,  # noqa: E402
                    select_c, splits)

GLOVE_DIR = ROOT / "data" / "external" / "glove"
GLOVE_URLS = ("https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",  # Stanford NLP's HF mirror
              "https://nlp.stanford.edu/data/glove.6B.zip")
_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    """Lower-case word tokens, matching GloVe 6B's lower-cased vocabulary."""
    return _TOKEN.findall(text.lower())


def download_zip(urls, dest):
    """Try each URL in turn; write to a .part file and keep it only if it is a valid zip."""
    tmp = dest.with_suffix(".part")
    for url in urls:
        try:
            print(f"downloading {url} (822 MB) ...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if zipfile.is_zipfile(tmp):
                tmp.rename(dest)
                return
            print(f"{url} did not return a zip file", flush=True)
        except OSError as e:  # urllib.error.URLError is an OSError
            print(f"{url} failed: {e}", flush=True)
    sys.exit("could not download glove.6B.zip from any source")


def ensure_glove(dim):
    path = GLOVE_DIR / f"glove.6B.{dim}d.txt"
    if not path.exists():
        GLOVE_DIR.mkdir(parents=True, exist_ok=True)
        archive = GLOVE_DIR / "glove.6B.zip"
        if not archive.exists():
            download_zip(GLOVE_URLS, archive)
        with zipfile.ZipFile(archive) as z:
            z.extract(path.name, GLOVE_DIR)
    return path


def load_glove(dim, keep=None):
    """{word: vector}; with keep, only those words are loaded (saves memory)."""
    vecs = {}
    with open(ensure_glove(dim), encoding="utf8") as f:
        for line in f:
            word, rest = line.rstrip("\n").split(" ", 1)
            if keep is None or word in keep:
                vecs[word] = np.asarray(rest.split(" "), dtype=np.float32)
    return vecs


def mean_vectors(texts, vecs, dim):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        known = [vecs[w] for w in tokenize(t) if w in vecs]
        if known:
            out[i] = np.mean(known, axis=0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, choices=[50, 100, 200, 300], required=True)
    a = p.parse_args()
    tr, va = splits()
    counts = Counter(w for t in tr[TEXT] for w in tokenize(t))
    t = Timer()
    with t("load"):
        vecs = load_glove(a.dim, keep=set(counts) | {w for t_ in va[TEXT] for w in tokenize(t_)})
    covered_types = sum(1 for w in counts if w in vecs) / len(counts)
    covered_tokens = sum(c for w, c in counts.items() if w in vecs) / sum(counts.values())
    with t("features"):
        x_tr = mean_vectors(tr[TEXT].tolist(), vecs, a.dim)
    with t("search"):
        search = select_c([("scale", StandardScaler())], x_tr, tr[LABEL].astype(int), DENSE_C_GRID)
    best = search.best_estimator_
    with t("predict"):
        p_pos = best.predict_proba(mean_vectors(va[TEXT].tolist(), vecs, a.dim))[:, 1]
    c_config, c_extra = c_record(search, DENSE_C_GRID)
    save_arm(f"f4_glove{a.dim}_mean", va, p_pos,
             config={"family": "static word vectors", "vectors": f"glove.6B.{a.dim}d", "pooling": "mean",
                     "tokenizer": _TOKEN.pattern + " on lower-cased text", "classifier": repr(best),
                     "fitted_on": "training split only", **c_config},
             cost={"feature_dim": a.dim, "trainable_params": a.dim + 1,
                   "fit_seconds": round(t.t["features"] + search.refit_time_, 3),  # features + final fit once
                   "c_search_seconds": t.t["search"],
                   "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "glove_load_seconds": t.t["load"], "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": None,
                   "model_size_mb": round((a.dim + 1) * 8 / 1e6, 4),
                   "lookup_table_mb": round(400_000 * a.dim * 4 / 1e6, 1)},
             extra={"glove_covers_train_word_types": round(covered_types, 4),
                    "glove_covers_train_tokens": round(covered_tokens, 4), **c_extra})


if __name__ == "__main__":
    main()
