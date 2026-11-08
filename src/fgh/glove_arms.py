"""(f) Static word vectors: average the GloVe vectors of a review's words, then logistic regression.

  python src/fgh/glove_arms.py --dim 50      # also 100, 300
First use downloads glove.6B.zip (822 MB; Wikipedia 2014 + Gigaword 5, 400k lower-case words) from Stanford
NLP into data/external/glove/ (git-ignored). Words missing from GloVe are skipped; a review with no known word
gets the zero vector. Also reports how much of the training vocabulary GloVe covers.
"""
import argparse
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, ROOT, TEXT, Timer, peak_rss_gb, save_arm, splits  # noqa: E402

GLOVE_DIR = ROOT / "data" / "external" / "glove"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    """Lower-case word tokens, matching GloVe 6B's lower-cased vocabulary."""
    return _TOKEN.findall(text.lower())


def ensure_glove(dim):
    path = GLOVE_DIR / f"glove.6B.{dim}d.txt"
    if not path.exists():
        GLOVE_DIR.mkdir(parents=True, exist_ok=True)
        archive = GLOVE_DIR / "glove.6B.zip"
        if not archive.exists():
            print(f"downloading {GLOVE_URL} (822 MB) ...", flush=True)
            urllib.request.urlretrieve(GLOVE_URL, archive)
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
    with t("fit"):
        x_tr = mean_vectors(tr[TEXT].tolist(), vecs, a.dim)
        clf = LogisticRegression(max_iter=2000).fit(x_tr, tr[LABEL].astype(int))
    with t("predict"):
        p_pos = clf.predict_proba(mean_vectors(va[TEXT].tolist(), vecs, a.dim))[:, 1]
    save_arm(f"f4_glove{a.dim}_mean", va, p_pos,
             config={"family": "static word vectors", "vectors": f"glove.6B.{a.dim}d", "pooling": "mean",
                     "tokenizer": _TOKEN.pattern + " on lower-cased text", "classifier": repr(clf)},
             cost={"feature_dim": a.dim, "trainable_params": a.dim + 1, "fit_seconds": t.t["fit"],
                   "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "glove_load_seconds": t.t["load"], "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": None,
                   "model_size_mb": round((a.dim + 1) * 8 / 1e6, 4),
                   "lookup_table_mb": round(400_000 * a.dim * 4 / 1e6, 1)},
             extra={"glove_covers_train_word_types": round(covered_types, 4),
                    "glove_covers_train_tokens": round(covered_tokens, 4)})


if __name__ == "__main__":
    main()
