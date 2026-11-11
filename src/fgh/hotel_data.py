"""(g) Hotel data: download, clean, hidden answer key and splits (plan items G1-G4). CPU only, about a minute.

  python src/fgh/hotel_data.py

Source: Hugging Face dataset jniimi/tripadvisor-review-rating (Apache-2.0; 201,295 English TripAdvisor reviews
from 2012, originally Li, Ott & Cardie 2013). Only its parquet file is downloaded, to data/external/tripadvisor/.
The hotel data simulates the brief's hypothetical hotel project for report (g)/(h). It never trains or tunes the
submitted model, and it stays out of git (data/external/ is git-ignored).

Steps and fixed rules (decided before any model is scored on hotels):
  G2 clean: English only (the dataset is already fastText-filtered), non-empty text, exact duplicate texts removed
     (keep the first). Model input is the review text without its title, because product reviews have no title;
     the title is kept in the files for a title + text variant.
  G3 answer key: overall 1-2 stars -> 0 (negative), 4-5 -> 1 (positive), 3 -> stress set only. This is the rule
     Blitzer et al. (2007) used to build our product data. Keys are written to separate *_key.csv files; methods
     read only *_text.csv, and only the scoring code reads the keys.
  G4 splits: the plan said "group by hotel", but the dataset's `hotel_id` column is unique on every row (values
     ~1.2e8-1.5e8, which look like TripAdvisor review ids), so it carries no hotel identity. Reviews are grouped
     by `user_id` instead, so no reviewer appears in two parts. Users are shuffled with seed 42 and cut into three
     disjoint blocks (10% / 10% / 80% of users); each part is then sampled from its own block:
       eval    2,000 labelled reviews (1-2 or 4-5 stars) at the natural class ratio   -> scored once per method
       stress3   300 three-star reviews                                               -> mixed-review stress set
       pool   20,000 reviews of any star rating, used as unlabelled text (g5, g6)     -> stars only for analysis
       fewshot  25 per class drawn from the pool's labelled reviews; ranks 1-8 / 1-16 / 1-25 give the nested
                8 / 16 / 25-per-class sets that stand in for hand labels in g7 (one draw, seed 42)
Outputs: data/external/hotel/{eval,stress3,pool}_{text,key}.csv, fewshot.csv, and a text-free summary with every
count and check in outputs/report_g/hotel_split.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOTEL, REPORT_G, ROOT, splits  # noqa: E402

REPO = "jniimi/tripadvisor-review-rating"
PARQUET = "data/train-00000-of-00001.parquet"
SRC = ROOT / "data" / "external" / "tripadvisor" / PARQUET
SEED = 42
SIZES = {"eval": 2000, "stress3": 300, "pool": 20000}
USER_BLOCKS = {"eval": 0.10, "stress3": 0.10, "pool": 0.80}  # share of shuffled users per block
FEWSHOT_PER_CLASS = 25
ASPECTS = ["cleanliness", "value", "location", "rooms", "sleep_quality"]


def download():
    if not SRC.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=SRC.parents[1])
    return SRC


def star_counts(s):
    return {str(int(k)): int(v) for k, v in s.value_counts().sort_index().items()}


def main():
    log = {"source": f"huggingface.co/datasets/{REPO} ({PARQUET})", "seed": SEED}
    df = pd.read_parquet(download(), columns=["hotel_id", "user_id", "title", "text", "overall", *ASPECTS, "lang"])
    log["raw_rows"] = len(df)
    log["hotel_id_unique_per_row"] = bool(df["hotel_id"].is_unique)
    df = df.rename(columns={"hotel_id": "review_id"})  # unique per row: a review id, not a hotel id (see docstring)

    steps = []
    df = df[df["lang"] == "__label__en"]
    steps.append(("english (fastText tag __label__en)", len(df)))
    df = df.assign(text=df["text"].fillna("").astype(str).str.strip(), title=df["title"].fillna("").astype(str))
    df = df[df["text"] != ""]
    steps.append(("non-empty text", len(df)))
    df = df.drop_duplicates("text").reset_index(drop=True)  # same rule as the product protocol: keep the first copy
    steps.append(("exact duplicate texts removed", len(df)))
    df = df[df["overall"].isin([1, 2, 3, 4, 5])].copy()
    steps.append(("overall rating in 1-5", len(df)))
    log["filter_steps"] = [{"step": s, "rows": n} for s, n in steps]

    df["overall"] = df["overall"].astype(int)
    df["label"] = np.select([df["overall"] <= 2, df["overall"] >= 4], [0, 1], default=-1)  # -1 = 3 stars
    df["user_id"] = df["user_id"].fillna("").astype(str)
    no_user = df["user_id"] == ""
    df.loc[no_user, "user_id"] = "review:" + df.loc[no_user, "review_id"].astype(str)  # own group if no user id
    log["rows_without_user_id"] = int(no_user.sum())
    labelled = df[df["label"] >= 0]
    log["corpus_stars"] = star_counts(df["overall"])
    log["corpus_negative_share_excl_3star"] = round(float((labelled["label"] == 0).mean()), 4)
    tr, _ = splits()
    log["product_train_negative_share"] = round(float((tr["sentiments"] == 0).mean()), 4)

    # G4: disjoint user blocks, then sample each part inside its own block
    users = df["user_id"].unique()
    rng = np.random.default_rng(SEED)
    users = users[rng.permutation(len(users))]
    cuts = np.cumsum([int(round(USER_BLOCKS[k] * len(users))) for k in ("eval", "stress3")])
    block_of = pd.Series(np.select([np.arange(len(users)) < cuts[0], np.arange(len(users)) < cuts[1]],
                                   ["eval", "stress3"], default="pool"), index=users)
    df["block"] = df["user_id"].map(block_of)
    parts = {
        "eval": df[(df["block"] == "eval") & (df["label"] >= 0)].sample(SIZES["eval"], random_state=SEED),
        "stress3": df[(df["block"] == "stress3") & (df["overall"] == 3)].sample(SIZES["stress3"], random_state=SEED),
        "pool": df[df["block"] == "pool"].sample(SIZES["pool"], random_state=SEED),
    }
    pool_lab = parts["pool"][parts["pool"]["label"] >= 0]
    few = pd.concat([pool_lab[pool_lab["label"] == c].sample(FEWSHOT_PER_CLASS, random_state=SEED)
                     .assign(rank=range(1, FEWSHOT_PER_CLASS + 1)) for c in (0, 1)])

    # checks: sizes, disjoint users and reviews, few-shot inside the pool, labels as specified
    for k, n in SIZES.items():
        assert len(parts[k]) == n, (k, len(parts[k]))
    names = list(parts)
    overlap = {f"{a}-{b}": int(len(set(parts[a]["user_id"]) & set(parts[b]["user_id"])))
               for i, a in enumerate(names) for b in names[i + 1:]}
    assert not any(overlap.values()), overlap
    assert few["review_id"].isin(parts["pool"]["review_id"]).all()
    assert set(parts["eval"]["label"]) == {0, 1} and (parts["stress3"]["overall"] == 3).all()
    log["user_overlap_between_parts"] = overlap
    log["users"] = {"total": int(len(users)), **{k: int((block_of == k).sum()) for k in USER_BLOCKS}}

    HOTEL.mkdir(parents=True, exist_ok=True)
    for k, d in parts.items():
        d[["review_id", "title", "text"]].to_csv(HOTEL / f"{k}_text.csv", index=False)
        d[["review_id", "label", "overall", *ASPECTS]].to_csv(HOTEL / f"{k}_key.csv", index=False)
    few[["review_id", "title", "text", "label", "rank"]].to_csv(HOTEL / "fewshot.csv", index=False)

    words = {k: d["text"].str.split().str.len() for k, d in parts.items()}
    log["parts"] = {k: {"rows": len(d), "stars": star_counts(d["overall"]),
                        "negative_share_excl_3star": (round(float((d.loc[d["label"] >= 0, "label"] == 0).mean()), 4)
                                                      if (d["label"] >= 0).any() else None),
                        "words_median": float(words[k].median()), "words_p95": float(words[k].quantile(0.95))}
                    for k, d in parts.items()}
    log["fewshot"] = {"per_class": FEWSHOT_PER_CLASS, "nested_sets": [8, 16, 25],
                      "stars": star_counts(few["overall"]) if "overall" in few else None}
    log["files"] = {k: [f"data/external/hotel/{k}_text.csv", f"data/external/hotel/{k}_key.csv"] for k in parts}
    log["files"]["fewshot"] = ["data/external/hotel/fewshot.csv"]
    REPORT_G.mkdir(parents=True, exist_ok=True)
    (REPORT_G / "hotel_split.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
