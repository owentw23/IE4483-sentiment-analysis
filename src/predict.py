"""Question (d): predict test.json with a saved model and write submission.csv.

  .venv/Scripts/python src/predict.py --model-dir outputs/runs/<run-name>/model [--out submission.csv]

Writes one row per test review in the original order: id, sentiments (0/1).
Also writes <out>_with_probs.csv (review text + P(positive)) for error analysis (question e).
The final column format is still to be confirmed with the instructor; change COLUMNS below if needed.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, ROOT, TEXT, load_test  # noqa: E402

COLUMNS = ["id", LABEL]

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--test-file", default="test.json")
p.add_argument("--out", default="submission.csv")
p.add_argument("--max-length", type=int, default=512)
p.add_argument("--batch-size", type=int, default=32)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).to(dev).eval()
te = load_test(a.test_file)

probs = []
with torch.no_grad():
    for i in range(0, len(te), a.batch_size):
        enc = tok(list(te[TEXT][i:i + a.batch_size]), truncation=True, max_length=a.max_length,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu())
p_pos = torch.cat(probs).numpy()
pred = (p_pos >= 0.5).astype(int)  # argmax for 2 classes

out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
pd.DataFrame({"id": range(len(te)), LABEL: pred})[COLUMNS].to_csv(out, index=False)
pd.DataFrame({"id": range(len(te)), TEXT: te[TEXT], LABEL: pred, "p_pos": p_pos.round(4)}) \
    .to_csv(out.with_name(out.stem + "_with_probs.csv"), index=False)
print(f"wrote {out} ({len(te)} rows; predicted negative: {int((pred == 0).sum())}, positive: {int((pred == 1).sum())})")
