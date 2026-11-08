"""(f) Sparse lexical arms: binary bag-of-words, TF-IDF, and a TF-IDF vocabulary-size sweep (CPU only).

Also re-creates the supporting files for the draft's Table 10 TF-IDF rows (0.911 / 0.766 unweighted,
0.920 / 0.841 class-weighted), whose settings were never committed. Two candidate settings are run:
  * the course notebook's TfidfVectorizer() (single words, lower-cased), and
  * TfidfVectorizer(ngram_range=(1, 2)) (the "1–2 gram" label used in PROGRESS.md).
Whatever these runs give is what the report uses.

  python src/fgh/tfidf_arms.py --list          # arm names, one per line
  python src/fgh/tfidf_arms.py --arm f3        # every arm whose name starts with "f3"
  python src/fgh/tfidf_arms.py                 # all arms (peak memory then accumulates across arms)

Vectoriser and classifier are fitted on the training split only. LogisticRegression: lbfgs, C = 1.0.
"""
import argparse
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, TEXT, Timer, peak_rss_gb, save_arm, splits  # noqa: E402


def arms():
    a = {
        "f1_bow_binary_1gram": (lambda: CountVectorizer(binary=True), None),
        "f2_tfidf_1gram": (lambda: TfidfVectorizer(), None),
        "f2_tfidf_1gram_balanced": (lambda: TfidfVectorizer(), "balanced"),
        "f2_tfidf_12gram": (lambda: TfidfVectorizer(ngram_range=(1, 2)), None),
        "f2_tfidf_12gram_balanced": (lambda: TfidfVectorizer(ngram_range=(1, 2)), "balanced"),
    }
    for n in (500, 2000, 5000, 10000):
        a[f"f3_tfidf_12gram_balanced_max{n}"] = (
            lambda n=n: TfidfVectorizer(ngram_range=(1, 2), max_features=n), "balanced")
    return a


def run(name, make_vec, class_weight, tr, va):
    t = Timer()
    with t("fit"):
        vec = make_vec()
        x_tr = vec.fit_transform(tr[TEXT])
        clf = LogisticRegression(max_iter=2000, class_weight=class_weight).fit(x_tr, tr[LABEL].astype(int))
    with t("predict"):
        p_pos = clf.predict_proba(vec.transform(va[TEXT]))[:, 1]
    dim = len(vec.vocabulary_)
    save_arm(name, va, p_pos,
             config={"family": "sparse lexical", "vectorizer": repr(vec), "classifier": repr(clf),
                     "class_weight": class_weight, "fitted_on": "training split only"},
             cost={"feature_dim": dim, "trainable_params": dim + 1, "fit_seconds": t.t["fit"],
                   "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": None,
                   "model_size_mb": round(len(pickle.dumps((vec, clf))) / 1e6, 2)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="", help="run arms whose name starts with this (default: all)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    chosen = {k: v for k, v in arms().items() if k.startswith(a.arm)}
    if a.list:
        print("\n".join(chosen))
        return
    if not chosen:
        sys.exit(f"no arm matches {a.arm!r}")
    tr, va = splits()
    for name, (make_vec, cw) in chosen.items():
        run(name, make_vec, cw, tr, va)


if __name__ == "__main__":
    main()
