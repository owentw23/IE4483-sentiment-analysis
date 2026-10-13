"""Shared data loading and the frozen validation protocol (see CLAUDE.md).

Every model must use load_splits() so that all arms are compared on the identical split.
If a cleaned dataset is used, pass its path; the same file must then be used for every arm.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TEXT, LABEL = "reviews", "sentiments"
SEED = 42


def _read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path)
    if TEXT not in df.columns:
        raise ValueError(f"{path} has no '{TEXT}' column (columns: {list(df.columns)})")
    df[TEXT] = df[TEXT].astype(str).str.strip()
    return df


def load_splits(train_file="train.json", val_size=0.2, seed=SEED):
    """Deduplicate, then stratified 80/20 split. Returns (train_df, val_df)."""
    df = _read(train_file).drop_duplicates(TEXT).reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    tr, va = train_test_split(df, test_size=val_size, stratify=df[LABEL], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def load_test(test_file="test.json"):
    """Test reviews in their original order (no dedup, no shuffling)."""
    return _read(test_file).reset_index(drop=True)


def class_weights(labels, mode="balanced"):
    """Balanced weights w_c = N / (K * N_c), computed from the training split only."""
    labels = np.asarray(labels)
    k = len(np.unique(labels))
    if mode == "none":
        return [1.0] * k
    counts = np.bincount(labels, minlength=k)
    return (len(labels) / (k * counts)).tolist()


if __name__ == "__main__":
    tr, va = load_splits()
    te = load_test()
    print(f"train {len(tr)} (neg {int((tr[LABEL] == 0).sum())} / pos {int((tr[LABEL] == 1).sum())})")
    print(f"val   {len(va)} (neg {int((va[LABEL] == 0).sum())} / pos {int((va[LABEL] == 1).sum())})")
    print(f"test  {len(te)}")
    print("class weights (neg, pos):", [round(w, 3) for w in class_weights(tr[LABEL])])
