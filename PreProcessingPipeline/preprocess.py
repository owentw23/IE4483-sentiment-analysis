"""Data preprocessing pipeline for the IE4483 product-review sentiment project.

Feature format: pretrained language model (Ettin encoder). Each raw review is
lightly normalised, duplicate training reviews are removed, the training data is
split into train/validation, and the model's own tokenizer converts every review
into token IDs plus an attention mask.

How to run (e.g. in Google Colab, with train.json and test.json next to this file):

    pip install "transformers>=4.48" datasets scikit-learn pandas
    python preprocess.py              # prints the statistics used in the report
                                      # and writes the cleaned splits to data/
    python preprocess.py --no-tokenize   # same, without transformers/datasets

From a training notebook:

    from preprocess import prepare_data
    data = prepare_data()
    data.datasets        # DatasetDict(train, validation, test) for the HF Trainer
    data.class_weights   # [w_negative, w_positive] for a weighted loss
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = "jhu-clsp/ettin-encoder-150m"
MAX_LENGTH = 256  # truncates ~2% of training reviews; see token_length_stats()
VAL_SIZE = 0.1
SEED = 42
DATA_DIR = Path(__file__).resolve().parent

_WHITESPACE = re.compile(r"\s+")


@dataclass
class PreparedData:
    train: pd.DataFrame  # columns: reviews (original), clean_text, sentiments
    val: pd.DataFrame
    test: pd.DataFrame  # original order preserved; no sentiments column
    class_weights: list[float]
    stats: dict = field(default_factory=dict)
    tokenizer: object = None
    datasets: object = None  # datasets.DatasetDict when tokenised


# ---------------------------------------------------------------------------
# Step 1: load and validate
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_json(Path(data_dir) / "train.json")
    test_df = pd.read_json(Path(data_dir) / "test.json")
    _validate(train_df, has_labels=True, name="train.json")
    _validate(test_df, has_labels=False, name="test.json")
    return train_df, test_df


def _validate(df: pd.DataFrame, has_labels: bool, name: str) -> None:
    required = ["reviews", "sentiments"] if has_labels else ["reviews"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}")
    bad_text = ~df["reviews"].map(lambda r: isinstance(r, str) and r.strip() != "")
    if bad_text.any():
        raise ValueError(f"{name}: {int(bad_text.sum())} empty or non-string reviews")
    if has_labels and not df["sentiments"].isin([0, 1]).all():
        raise ValueError(f"{name}: sentiments must be 0 or 1")


# ---------------------------------------------------------------------------
# Step 2: light text normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalise a review without removing any sentiment cue.

    Deliberately no lowercasing, punctuation stripping, stop-word removal or
    stemming: the Ettin tokenizer is case-sensitive ("NOT" and "not" are
    different tokens), has tokens for punctuation runs such as "!!!", and
    negations like "not" appear in most stop-word lists.

    No NFKC Unicode folding either: in this data it only changes 9 reviews,
    mostly shoe sizes ("5½" -> "51⁄2", which reads as 51/2). The tokenizer
    already applies NFC normalisation itself.
    """
    return _WHITESPACE.sub(" ", text).strip()  # newlines, tabs, repeated spaces -> one space


# ---------------------------------------------------------------------------
# Step 3: remove duplicate training reviews
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop repeated reviews, compared after normalisation.

    Copies with the same label collapse to one row, so the same text cannot end
    up in both the train and validation split. A text that appears with both
    labels is dropped entirely because its true label is ambiguous.
    """
    conflicting = df.groupby("clean_text")["sentiments"].transform("nunique") > 1
    kept = df[~conflicting].drop_duplicates("clean_text", keep="first")
    stats = {
        "duplicate_rows_removed": int((~conflicting).sum() - len(kept)),
        "conflicting_rows_removed": int(conflicting.sum()),
    }
    return kept.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Step 4: stratified train/validation split and class weights
# ---------------------------------------------------------------------------

def split_train_val(df: pd.DataFrame, val_size: float = VAL_SIZE,
                    seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["sentiments"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def class_weights(labels: pd.Series) -> list[float]:
    """Inverse-frequency weights n / (k * n_c), for a weighted cross-entropy loss."""
    counts = np.bincount(labels, minlength=2)
    return (len(labels) / (len(counts) * counts)).round(4).tolist()


# ---------------------------------------------------------------------------
# Step 5: tokenisation (the actual text -> feature conversion)
# ---------------------------------------------------------------------------
# transformers/datasets are imported inside these functions so that steps 1-4
# also work for the TF-IDF / word-embedding baselines without installing them.

def load_tokenizer(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def token_length_stats(texts: pd.Series, tokenizer,
                       cutoffs: tuple[int, ...] = (128, 256, 512)) -> dict:
    """Token counts (incl. special tokens) used to justify MAX_LENGTH."""
    lengths = np.array([len(ids) for ids in tokenizer(texts.tolist())["input_ids"]])
    stats = {
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }
    stats["truncated"] = {c: float((lengths > c).mean()) for c in cutoffs}
    return stats


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   tokenizer, max_length: int = MAX_LENGTH):
    """Tokenise each split into input_ids + attention_mask (+ labels).

    No padding here: pad per batch at training time with DataCollatorWithPadding,
    which is much cheaper than padding every review to max_length.
    """
    from datasets import Dataset, DatasetDict

    def to_dataset(df: pd.DataFrame, with_labels: bool):
        columns = {"text": df["clean_text"].tolist()}
        if with_labels:
            columns["labels"] = df["sentiments"].astype(int).tolist()
        return Dataset.from_dict(columns).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["text"])

    return DatasetDict(
        train=to_dataset(train_df, with_labels=True),
        validation=to_dataset(val_df, with_labels=True),
        test=to_dataset(test_df, with_labels=False),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_data(data_dir: Path = DATA_DIR, model_name: str = MODEL_NAME,
                 max_length: int = MAX_LENGTH, val_size: float = VAL_SIZE,
                 seed: int = SEED, tokenize: bool = True) -> PreparedData:
    raw_train, test_df = load_data(data_dir)

    raw_train["clean_text"] = raw_train["reviews"].map(normalize_text)
    test_df["clean_text"] = test_df["reviews"].map(normalize_text)

    deduped, dedup_stats = drop_duplicates(raw_train)
    train_df, val_df = split_train_val(deduped, val_size, seed)

    stats = {
        "raw_train": len(raw_train),
        "raw_negative": int((raw_train["sentiments"] == 0).sum()),
        "normalisation_changed": int((raw_train["reviews"] != raw_train["clean_text"]).sum()),
        **dedup_stats,
        "after_dedup": len(deduped),
        "test": len(test_df),
        "test_seen_in_train": int(test_df["clean_text"].isin(deduped["clean_text"]).sum()),
        "train": len(train_df),
        "val": len(val_df),
        "train_negative_rate": float((train_df["sentiments"] == 0).mean()),
        "val_negative_rate": float((val_df["sentiments"] == 0).mean()),
    }
    data = PreparedData(train=train_df, val=val_df, test=test_df,
                        class_weights=class_weights(train_df["sentiments"]), stats=stats)

    if tokenize:
        data.tokenizer = load_tokenizer(model_name)
        stats["token_lengths"] = token_length_stats(train_df["clean_text"], data.tokenizer)
        data.datasets = build_datasets(train_df, val_df, test_df, data.tokenizer, max_length)
    return data


def print_report(data: PreparedData, max_length: int = MAX_LENGTH) -> None:
    s = data.stats
    print("== Preprocessing summary ==")
    print(f"Raw training reviews:            {s['raw_train']:,} "
          f"({s['raw_negative']:,} negative = {s['raw_negative'] / s['raw_train']:.1%})")
    print(f"Changed by normalisation:        {s['normalisation_changed']:,}")
    print(f"Duplicate rows removed:          {s['duplicate_rows_removed']:,} "
          f"(conflicting-label rows removed: {s['conflicting_rows_removed']})")
    print(f"Training reviews after dedup:    {s['after_dedup']:,}")
    print(f"Test reviews (kept, in order):   {s['test']:,} "
          f"({s['test_seen_in_train']} also appear in train)")
    print(f"Train / validation:              {s['train']:,} / {s['val']:,} "
          f"(negative {s['train_negative_rate']:.1%} / {s['val_negative_rate']:.1%})")
    print(f"Class weights [neg, pos]:        {data.class_weights}")
    if "token_lengths" in s:
        t = s["token_lengths"]
        print(f"Token lengths (train split):     median {t['median']}, p95 {t['p95']}, "
              f"p99 {t['p99']}, max {t['max']}")
        truncated = ", ".join(f"{c}: {r:.1%}" for c, r in t["truncated"].items())
        print(f"Reviews truncated at length:     {truncated}  (using {max_length})")
        print(f"Tokenised datasets:              {data.datasets}")


def save_splits(data: PreparedData, out_dir: Path) -> None:
    """Write the splits so every feature format (TF-IDF, GloVe, Ettin) uses the same data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data.train.to_csv(out_dir / "train_split.csv", index=False)
    data.val.to_csv(out_dir / "val_split.csv", index=False)
    data.test.to_csv(out_dir / "test_clean.csv", index=False)
    print(f"Saved train_split.csv, val_split.csv, test_clean.csv to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the product-review data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder containing train.json and test.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "data",
                        help="where to write the cleaned splits")
    parser.add_argument("--model", default=MODEL_NAME, help="tokenizer to use")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--no-tokenize", action="store_true",
                        help="skip tokenisation (no transformers/datasets needed)")
    args = parser.parse_args()

    data = prepare_data(args.data_dir, args.model, args.max_length,
                        tokenize=not args.no_tokenize)
    print_report(data, args.max_length)
    save_splits(data, args.out_dir)


if __name__ == "__main__":
    main()
