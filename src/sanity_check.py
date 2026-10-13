"""Karpathy-style sanity checks before real training (lecture L5):
1. Initial loss should be close to ln 2 = 0.693 for an untrained 2-class head.
2. The model must be able to overfit one small batch (loss -> ~0). If it can't, the pipeline is broken.

  .venv/Scripts/python src/sanity_check.py [--model jhu-clsp/ettin-encoder-400m]
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, TEXT, class_weights, load_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="jhu-clsp/ettin-encoder-400m")
p.add_argument("--n", type=int, default=16)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--lr", type=float, default=3e-5)
a = p.parse_args()

set_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr, _ = load_splits()
w = torch.tensor(class_weights(tr[LABEL]), device=dev)
# A balanced tiny batch: n/2 negatives + n/2 positives
batch_df = pd.concat([tr[tr[LABEL] == c].sample(a.n // 2, random_state=42) for c in (0, 1)])
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
batch = tok(list(batch_df[TEXT]), truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
y = torch.tensor(batch_df[LABEL].values, device=dev)

model.eval()
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
    loss0 = torch.nn.functional.cross_entropy(model(**batch).logits.float(), y, weight=w).item()
print(f"1) initial loss {loss0:.3f}  (expected ~{math.log(2):.3f})")

model.train()
model.gradient_checkpointing_enable()
# Plain fp32 AdamW needs ~16 bytes/param = 6.4 GB for 400M params and spills past 8 GB VRAM (lecture L6).
import bitsandbytes as bnb  # noqa: E402
opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)
for step in range(1, a.steps + 1):
    with torch.autocast(dev, dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), y, weight=w)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 10 == 0 or step == 1:
        print(f"   step {step:3d}  loss {loss.item():.4f}")
print("2) overfit one batch:", "PASS" if loss.item() < 0.05 else "CHECK - loss did not approach 0")
