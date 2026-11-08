"""(f) Pretrained contextual models used without fine-tuning (GPU recommended).

  python src/fgh/pretrained_arms.py --arm f6_qwen3_embedding_frozen   # frozen Qwen3-Embedding-0.6B + LogReg
  python src/fgh/pretrained_arms.py --arm f7_ettin400m_frozen         # frozen Ettin-400M (not fine-tuned) + LogReg
  python src/fgh/pretrained_arms.py --arm f10_siebert                 # SiEBERT used as-is

f6 repeats the embedding of src/cascade.py exactly (bf16, left padding, last-token vector, L2-normalised,
max length 512, no query prompt) with LogisticRegression(C=1.0, max_iter=3000), so it re-creates the 0.908
macro-F1 recorded in outputs/runs/cascade_embed/metrics.json, this time with timing and a clean log.
f7 is the same backbone as our submitted model with nothing fine-tuned: masked mean pooling, L2-normalised.
f10 re-creates the draft's SiEBERT row (0.958 / 0.921). SiEBERT was fine-tuned on 15 labelled sentiment
datasets, including four domains of Blitzer et al. 2007 (our data's corpus), so it is not label-free.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, TEXT, Timer, peak_rss_gb, save_arm, splits  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODELS = {"f6_qwen3_embedding_frozen": "Qwen/Qwen3-Embedding-0.6B",
          "f7_ettin400m_frozen": "jhu-clsp/ettin-encoder-400m",
          "f10_siebert": "siebert/sentiment-roberta-large-english"}


def load(name, cls=AutoModel, **tok_kwargs):
    tok = AutoTokenizer.from_pretrained(name, **tok_kwargs)
    model = cls.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    return tok, model


@torch.no_grad()
def embed(tok, model, texts, pooling, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEVICE)
        h = model(**b).last_hidden_state.float()
        if pooling == "last":  # left padding, so the last position is the last real token
            v = h[:, -1]
        else:
            mask = b["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(F.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def classify(tok, model, texts, bs=32):
    pos = next(i for i, lab in model.config.id2label.items() if str(lab).upper().startswith("POS"))
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEVICE)
        out.append(torch.softmax(model(**b).logits.float(), -1)[:, pos].cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=list(MODELS), required=True)
    a = p.parse_args()
    name = MODELS[a.arm]
    tr, va = splits()
    t = Timer()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if a.arm == "f10_siebert":
        with t("load"):
            tok, model = load(name, AutoModelForSequenceClassification)
        with t("predict"):
            p_pos = classify(tok, model, va[TEXT].tolist())
        fit_seconds, trainable, dim, cfg = 0.0, 0, None, {"use": "as-is, no training on our data",
                                                           "labelled_training_data": "15 sentiment datasets"}
    else:
        pooling = "last" if a.arm.startswith("f6") else "mean"
        with t("load"):
            tok, model = load(name, padding_side="left") if pooling == "last" else load(name)
        with t("fit"):
            x_tr = embed(tok, model, tr[TEXT].tolist(), pooling)
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(x_tr, tr[LABEL].astype(int))
        with t("predict"):
            p_pos = clf.predict_proba(embed(tok, model, va[TEXT].tolist(), pooling))[:, 1]
        fit_seconds, dim = t.t["fit"], x_tr.shape[1]
        trainable = dim + 1
        cfg = {"pooling": pooling, "normalise": "L2", "max_length": 512, "classifier": repr(clf),
               "fitted_on": "training split only"}

    n_params = sum(q.numel() for q in model.parameters())
    save_arm(a.arm, va, p_pos,
             config={"family": "pretrained contextual, frozen", "model": name, "dtype": str(DTYPE), **cfg},
             cost={"feature_dim": dim, "trainable_params": trainable, "frozen_params": n_params,
                   "fit_seconds": fit_seconds, "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "model_load_seconds": t.t["load"], "peak_rss_gb": peak_rss_gb(),
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEVICE == "cuda" else None,
                   "model_size_mb": round(n_params * (2 if DTYPE == torch.bfloat16 else 4) / 1e6, 1)})


if __name__ == "__main__":
    main()
