"""(f) Pretrained contextual models used without fine-tuning (GPU recommended).

  python src/fgh/pretrained_arms.py --arm f6_qwen3_embedding_frozen   # frozen Qwen3-Embedding-0.6B + LogReg
  python src/fgh/pretrained_arms.py --arm f7_ettin400m_frozen         # frozen Ettin-400M (not fine-tuned) + LogReg
  python src/fgh/pretrained_arms.py --arm f10_siebert                 # SiEBERT used as-is

f6 repeats the embedding of src/cascade.py exactly (bf16, left padding, last-token vector, L2-normalised,
max length 512, no query prompt). With cascade.py's LogisticRegression(C=1.0) it re-created the 0.908 macro-F1 in
outputs/runs/cascade_embed/metrics.json (0.906; kept as outputs/runs/f6_qwen3_embedding_frozen_C1).
f7 is the same backbone as our submitted model with nothing fine-tuned: masked mean pooling, L2-normalised. With
C = 1.0 it predicted every review positive (kept as outputs/runs/f7_ettin400m_frozen_C1).
Since 2026-11-08 both use StandardScaler + LogisticRegression with C chosen by 5-fold CV on the training split
(common.select_c), the same rule as every other logistic-regression head in (f).
f10 re-creates the draft's SiEBERT row (0.958 / 0.921). SiEBERT was fine-tuned on 15 labelled sentiment
datasets, including four domains of Blitzer et al. 2007 (our data's corpus), so it is not label-free.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DENSE_C_GRID, LABEL, TEXT, Timer, c_record, peak_rss_gb, save_arm,  # noqa: E402
                    select_c, splits)

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
        c_extra, c_search_seconds = {}, None
    else:
        pooling = "last" if a.arm.startswith("f6") else "mean"
        with t("load"):
            tok, model = load(name, padding_side="left") if pooling == "last" else load(name)
        with t("features"):
            x_tr = embed(tok, model, tr[TEXT].tolist(), pooling)
        with t("search"):
            search = select_c([("scale", StandardScaler())], x_tr, tr[LABEL].astype(int), DENSE_C_GRID)
        best = search.best_estimator_
        with t("predict"):
            p_pos = best.predict_proba(embed(tok, model, va[TEXT].tolist(), pooling))[:, 1]
        fit_seconds = round(t.t["features"] + search.refit_time_, 3)  # embed train split + final fit once
        dim = x_tr.shape[1]
        trainable = dim + 1
        c_config, c_extra = c_record(search, DENSE_C_GRID)
        c_search_seconds = t.t["search"]
        cfg = {"pooling": pooling, "normalise": "L2", "max_length": 512, "classifier": repr(best),
               "fitted_on": "training split only", **c_config}

    n_params = sum(q.numel() for q in model.parameters())
    save_arm(a.arm, va, p_pos,
             config={"family": "pretrained contextual, frozen", "model": name, "dtype": str(DTYPE), **cfg},
             cost={"feature_dim": dim, "trainable_params": trainable, "frozen_params": n_params,
                   "fit_seconds": fit_seconds, "c_search_seconds": c_search_seconds,
                   "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "model_load_seconds": t.t["load"], "peak_rss_gb": peak_rss_gb(),
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEVICE == "cuda" else None,
                   "model_size_mb": round(n_params * (2 if DTYPE == torch.bfloat16 else 4) / 1e6, 1)},
             extra=c_extra)


if __name__ == "__main__":
    main()
