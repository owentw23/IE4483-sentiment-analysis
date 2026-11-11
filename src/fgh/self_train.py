"""(g5, g5b) Self-training data: pseudo-label the unlabelled hotel pool with our product model (the teacher).

  python src/fgh/self_train.py --variant g5    # teacher confidence >= 0.99
  python src/fgh/self_train.py --variant g5b   # same, and the NLI model g2 must agree with the teacher

Needs outputs/runs/g1_ettin_final/pool_predictions.csv (and g2_finecat_nli_v2/pool_predictions.csv for g5b),
written by `hotel_arms.py --pool`. Rules fixed in the plan (7 Nov) before any hotel score was seen:
  1. teacher = ettin400m_final, the submitted product model; it labels all 20,000 pool reviews
  2. keep a review when the teacher's confidence in its predicted class is >= 0.99
     (g5b: and g2's prediction agrees with the teacher)
  3. cap the larger class at random (seed 42) so the kept set has the same negative share the teacher predicted
     over the whole pool, so positives do not drown out negatives
  4. a fresh Ettin-400M student (pretrained checkpoint, not the teacher) trains on product reviews + the kept hotel
     reviews with the submitted recipe; src/train.py --extra-train does exactly this (see run_g.sh, stage selftrain)
  5. the student's best epoch is chosen on product validation, the only labelled data the scenario has
For analysis only, the hidden stars give the pseudo-label error rate (3-star reviews counted separately), which
links (g) to (h): self-training feeds its own mistakes back as noisy labels.
Outputs: data/external/hotel/pseudo_<variant>.csv (reviews, sentiments; the train.py --extra-train format) and
outputs/report_g/pseudo_<variant>.json (counts and error rates, no text).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOTEL, LABEL, REPORT_G, RUNS, TEXT, hotel_key, hotel_text  # noqa: E402

THRESHOLD = 0.99
SEED = 42


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["g5", "g5b"], required=True)
    a = p.parse_args()
    pool = hotel_text("pool")
    teacher = pd.read_csv(RUNS / "g1_ettin_final" / "pool_predictions.csv")
    assert (teacher["review_id"].values == pool["review_id"].values).all()
    p_t = teacher["p_pos"].values
    pred = (p_t >= 0.5).astype(int)
    conf = np.maximum(p_t, 1 - p_t)
    keep = conf >= THRESHOLD
    stats = {"variant": a.variant, "pool": len(pool), "threshold": THRESHOLD,
             "teacher_predicted_negative_share": round(float((pred == 0).mean()), 4),
             "confident": {"negative": int((keep & (pred == 0)).sum()), "positive": int((keep & (pred == 1)).sum())}}
    if a.variant == "g5b":
        other = pd.read_csv(RUNS / "g2_finecat_nli_v2" / "pool_predictions.csv")
        assert (other["review_id"].values == pool["review_id"].values).all()
        agree = (other["p_pos"].values >= 0.5).astype(int) == pred
        stats["agreement_with_g2_whole_pool"] = round(float(agree.mean()), 4)
        keep &= agree
        stats["confident_and_agree"] = {"negative": int((keep & (pred == 0)).sum()),
                                        "positive": int((keep & (pred == 1)).sum())}

    # cap the larger class so the kept negative share equals the teacher's predicted share over the pool
    q = (pred == 0).mean()
    idx = {c: np.flatnonzero(keep & (pred == c)) for c in (0, 1)}
    n0, n1 = len(idx[0]), len(idx[1])
    if n0 / max(n0 + n1, 1) < q:
        n1 = min(n1, int(round(n0 * (1 - q) / q)))
    else:
        n0 = min(n0, int(round(n1 * q / (1 - q))))
    rng = np.random.default_rng(SEED)
    chosen = np.sort(np.r_[rng.choice(idx[0], n0, replace=False), rng.choice(idx[1], n1, replace=False)])
    stats["kept"] = {"negative": int(n0), "positive": int(n1), "total": int(n0 + n1),
                     "negative_share": round(n0 / max(n0 + n1, 1), 4)}

    out = pd.DataFrame({TEXT: pool["text"].values[chosen], LABEL: pred[chosen],
                        "review_id": pool["review_id"].values[chosen], "p_teacher": np.round(p_t[chosen], 6)})
    out.to_csv(HOTEL / f"pseudo_{a.variant}.csv", index=False)

    # analysis only: compare with the hidden stars
    key = hotel_key("pool").set_index("review_id").loc[out["review_id"]]
    stars, true = key["overall"].values, key["label"].values
    lab = stars != 3
    wrong = (out[LABEL].values != true) & lab
    stats["pseudo_label_check"] = {
        "share_3_star": round(float((~lab).mean()), 4),
        "error_rate_excl_3_star": round(float(wrong.sum() / lab.sum()), 4),
        "error_rate_negative_pseudo": round(float(wrong[out[LABEL].values == 0].sum()
                                                  / max(lab[out[LABEL].values == 0].sum(), 1)), 4),
        "error_rate_positive_pseudo": round(float(wrong[out[LABEL].values == 1].sum()
                                                  / max(lab[out[LABEL].values == 1].sum(), 1)), 4),
        "teacher_error_whole_pool_excl_3_star": round(float(
            (pred != hotel_key("pool")["label"].values)[hotel_key("pool")["overall"].values != 3].mean()), 4)}
    REPORT_G.mkdir(parents=True, exist_ok=True)
    (REPORT_G / f"pseudo_{a.variant}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
