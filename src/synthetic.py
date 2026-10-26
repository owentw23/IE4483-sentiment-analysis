"""Parse, post-process and dedup the synthetic negative reviews in data/synthetic/batch_*.txt.

Post-processing copies the formatting quirks of the real training negatives, so the model cannot tell
synthetic from real by formatting alone (it should only learn from content):
  - final punctuation: 69% of real negatives end on a letter (the scrape strips it, sometimes with the last
    letter), 23% on ".", 4% on "!"
  - 46% use two spaces after sentence ends
Near-duplicates of any real train/val/test review (TF-IDF cosine >= --max-sim) are dropped.

Writes data/synthetic/synthetic_negatives.json (columns reviews, sentiments, persona, type).
  .venv/Scripts/python src/synthetic.py
"""
import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, ROOT, SEED, TEXT, load_test  # noqa: E402

SYN = ROOT / "data" / "synthetic"
HEADER = re.compile(r"^@@ p=(P\d+) t=(\w+)\s*$")


def parse(path):
    rows, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER.match(line)
        if m:
            cur = {"persona": m[1], "type": m[2], "lines": []}
            rows.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    return [{"persona": r["persona"], "type": r["type"], TEXT: "\n".join(r["lines"]).strip()} for r in rows]


def match_format(text, rng):
    """Mimic the real negatives' sentence spacing and endings."""
    if rng.random() < 0.46:
        text = re.sub(r"([.!?]) (?=[A-Z])", r"\1  ", text)
    body = text.rstrip(".!? ")
    r = rng.random()
    if r < 0.23:
        return body + "."
    if r < 0.27:
        return body + "!"
    if r < 0.35 and len(body) > 20 and body[-1].isalpha():
        return body[:-1]  # scrape artefact: last letter cut off
    return body


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-sim", type=float, default=0.6)
    p.add_argument("--n", type=int, default=800, help="keep this many reviews (random sample, seed 42)")
    a = p.parse_args(argv)

    rows = [r for f in sorted(SYN.glob("batch_*.txt")) for r in parse(f)]
    df = pd.DataFrame(rows)
    empty = df[TEXT].str.len() == 0
    if empty.any():
        sys.exit(f"{empty.sum()} empty reviews")
    print(f"parsed {len(df)} reviews; types {df.type.value_counts().to_dict()}")
    print("words p10/50/90:", df[TEXT].str.split().str.len().quantile([.1, .5, .9]).astype(int).tolist())

    dup = df[TEXT].duplicated()
    df = df[~dup].reset_index(drop=True)
    real = pd.concat([pd.read_json(ROOT / "train.json")[TEXT], load_test()[TEXT]]).astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(real + df[TEXT].tolist())
    sim = cosine_similarity(vec.transform(df[TEXT]), vec.transform(real)).max(1)
    near = sim >= a.max_sim
    print(f"exact internal duplicates dropped: {int(dup.sum())}; near-duplicates of real reviews dropped: "
          f"{int(near.sum())} (max cosine {sim.max():.2f}, median {pd.Series(sim).median():.2f})")
    df = df[~near].reset_index(drop=True)
    if len(df) > a.n:
        df = df.sample(a.n, random_state=SEED).sort_index().reset_index(drop=True)

    rng = random.Random(SEED)
    df[TEXT] = [match_format(t, rng) for t in df[TEXT]]
    df[LABEL] = 0
    out = SYN / "synthetic_negatives.json"
    df[[TEXT, LABEL, "persona", "type"]].to_json(out, orient="records", indent=1, force_ascii=False)
    print(f"wrote {len(df)} reviews to {out}")


if __name__ == "__main__":
    main()
