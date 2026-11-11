"""(f) Sparse lexical arms: binary bag-of-words, TF-IDF, and a TF-IDF vocabulary-size sweep (CPU only).

Also re-creates the supporting files for the draft's Table 10 TF-IDF rows (0.911 / 0.766 unweighted,
0.920 / 0.841 class-weighted), whose settings were never committed. Two candidate settings are run:
  * the course notebook's TfidfVectorizer() (single words, lower-cased), and
  * TfidfVectorizer(ngram_range=(1, 2)) (the "1–2 gram" label used in PROGRESS.md).
Whatever these runs give is what the report uses.

  python src/fgh/tfidf_arms.py --list                  # arm names, one per line
  python src/fgh/tfidf_arms.py --arm f2_tfidf_1gram    # exactly that arm
  python src/fgh/tfidf_arms.py --arm f3                # every arm whose name starts with "f3"
  python src/fgh/tfidf_arms.py                         # all arms (peak memory then accumulates across arms)

Vectoriser and classifier are fitted on the training split only. LogisticRegression (lbfgs); its C is chosen per
arm by 5-fold stratified cross-validation on the training split only (macro-F1, the project's selection metric),
with the vectoriser refitted inside every fold so no held-out text leaks into the vocabulary or IDF weights
(common.select_c). The chosen pipeline is then refitted on all 5,748 training reviews and scored once on validation.
Grid extended to 3000 on 2026-11-08 because three unweighted arms chose C = 100, the old upper edge.
"""
import argparse
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, TEXT, Timer, c_record, peak_rss_gb, save_arm, select_c, splits  # noqa: E402

C_GRID = (0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000)


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
    with t("search"):  # 5-fold CV over the C grid, then refit of the best pipeline on the whole training split
        search = select_c([("vec", make_vec())], tr[TEXT], tr[LABEL].astype(int), C_GRID, class_weight=class_weight)
    best = search.best_estimator_
    with t("predict"):
        p_pos = best.predict_proba(va[TEXT])[:, 1]
    vec, clf = best.named_steps["vec"], best.named_steps["clf"]
    dim = len(vec.vocabulary_)
    c_config, c_extra = c_record(search, C_GRID)
    save_arm(name, va, p_pos,
             config={"family": "sparse lexical", "vectorizer": repr(vec), "classifier": repr(clf),
                     "class_weight": class_weight, "fitted_on": "training split only", **c_config},
             cost={"feature_dim": dim, "trainable_params": dim + 1,
                   "fit_seconds": round(search.refit_time_, 3),  # training the final model once (vectoriser + LR)
                   "c_search_seconds": t.t["search"],             # whole CV search incl. refit (3 parallel workers)
                   "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "peak_rss_gb": peak_rss_gb(), "peak_gpu_gb": None,
                   "model_size_mb": round(len(pickle.dumps((vec, clf))) / 1e6, 2)},
             extra=c_extra)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="", help="an exact arm name, else every arm starting with this (default: all)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    all_arms = arms()
    # exact name first: "f2_tfidf_1gram" must not also run "f2_tfidf_1gram_balanced" in the same process
    chosen = ({a.arm: all_arms[a.arm]} if a.arm in all_arms
              else {k: v for k, v in all_arms.items() if k.startswith(a.arm)})
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
