"""(g7) A few hand-labelled hotel reviews: 8, 16 or 25 per class (16-50 labels; Blitzer et al. used 50).

  python src/fgh/fewshot.py --method ettin  --k 8 16 25   # g7a: keep training our product Ettin
  python src/fgh/fewshot.py --method setfit --k 8 16 25   # g7b: SetFit on gte-modernbert-base (needs `setfit`)

The labels are the stars of data/external/hotel/fewshot.csv: 25 random pool reviews per class (seed 42); the k-set
is ranks 1..k, so the sets are nested. One draw; few-shot scores swing with the draw, and the report says so.
Settings are fixed in advance because 16-50 labels leave nothing to validate on; nothing is tuned on hotel stars.

g7a, "Ettin again": start from ettin400m_final (the submitted model) and continue training on the k-per-class hotel
    reviews mixed 1:1 with k-per-class product reviews from our training split (seed 42), so it does not forget
    products. Learning rate 1e-5 (half the submitted 2e-5), 10 epochs, final weights (no early stopping), effective
    batch 16 (8 x 2), 10% warm-up, weight decay 0.01, 8-bit AdamW, bf16, max length 512, seed 42.
g7b, "something else": SetFit (Tunstall et al. 2022) with the setfit library's defaults: contrastive pairs from the
    hotel labels only (20 pair iterations per review), CosineSimilarityLoss, 1 epoch, batch 16, body learning rate
    2e-5, then a logistic-regression head; body Alibaba-NLP/gte-modernbert-base (149M, ModernBERT-base), max length
    512, seed 42.
Each arm is scored like every other (g) arm: product validation, hotel evaluation set, 3-star stress set.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, ROOT, TEXT, hotel_inputs, hotel_text, save_g_arm, splits  # noqa: E402

SEED = 42
FINAL = ROOT / "outputs" / "runs" / "ettin400m_final" / "model"
SETFIT_BODY = "Alibaba-NLP/gte-modernbert-base"
G7A = {"lr": 1e-5, "epochs": 10, "batch_size": 8, "grad_accum": 2, "warmup_ratio": 0.1, "weight_decay": 0.01,
       "optim": "adamw_bnb_8bit", "bf16": True, "max_length": 512, "seed": SEED, "product_mix": "1:1, k per class"}
G7B = {"body": SETFIT_BODY, "num_iterations": 20, "num_epochs": 1, "batch_size": 16, "body_learning_rate": 2e-5,
       "head": "LogisticRegression (setfit default)", "max_length": 512, "seed": SEED, "product_reviews_used": 0,
       "precision": "fp32", "gradient_checkpointing": True}


def few(k):
    f = hotel_text("fewshot")
    f = f[f["rank"] <= k]
    assert (f["label"].value_counts() == k).all(), f["label"].value_counts()
    return f


def eval_sets():
    _, va = splits()
    return {"product": va[TEXT].tolist(), "hotel": hotel_inputs(hotel_text("eval")),
            "stress": hotel_inputs(hotel_text("stress3"))}


def run_ettin(k):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)
    from hotel_arms import by_length
    set_seed(SEED)
    tr, _ = splits()
    f = few(k)
    prod = pd.concat([tr[tr[LABEL] == c].sample(k, random_state=SEED) for c in (0, 1)])
    data = pd.concat([pd.DataFrame({TEXT: f["text"], LABEL: f["label"].astype(int)}), prod[[TEXT, LABEL]]]) \
        .sample(frac=1, random_state=SEED).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(FINAL)
    model = AutoModelForSequenceClassification.from_pretrained(FINAL)
    enc = tok(data[TEXT].tolist(), truncation=True, max_length=G7A["max_length"])
    ds = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(y)}
          for i, y in enumerate(data[LABEL])]
    args = TrainingArguments(
        output_dir=str(ROOT / "data" / "external" / "models" / f"g7a_k{k}_tmp"), learning_rate=G7A["lr"],
        num_train_epochs=G7A["epochs"], per_device_train_batch_size=G7A["batch_size"],
        gradient_accumulation_steps=G7A["grad_accum"], warmup_steps=G7A["warmup_ratio"],
        weight_decay=G7A["weight_decay"], lr_scheduler_type="linear", optim=G7A["optim"], bf16=True,
        gradient_checkpointing=True, save_strategy="no", logging_steps=5, report_to="none", seed=SEED,
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    fit = time.time() - t0
    model = trainer.model.to(torch.bfloat16).eval()

    @torch.no_grad()
    def batch(texts):
        b = tok(texts, truncation=True, max_length=G7A["max_length"], padding=True, return_tensors="pt").to("cuda")
        return torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy()

    p = {name: by_length(texts, batch, 32) for name, texts in eval_sets().items()}
    save_g_arm(f"g7a_ettin_k{k}", product_p=p["product"], hotel_p=p["hotel"], stress_p=p["stress"],
               config={"method": "continue training ettin400m_final on few hotel labels + product reviews",
                       "start": str(FINAL.relative_to(ROOT)), "hotel_labels_used": 2 * k, "k_per_class": k,
                       "n_train": len(data), **G7A,
                       "train_loss_log": [h for h in trainer.state.log_history if "loss" in h]},
               cost={"params": int(sum(q.numel() for q in model.parameters())), "fit_seconds": round(fit, 1),
                     "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})


def run_setfit(k):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    f = few(k)
    model = SetFitModel.from_pretrained(SETFIT_BODY)
    # transformers 5 loads the checkpoint's stored dtype (fp16 here); SetFit trains in fp32 (no AMP by default),
    # and an fp16 body fails in the loss backward ("Found dtype Float but expected Half").
    model.model_body = model.model_body.float()
    # fp32 at 512 tokens x 16 pairs needs >24 GB of activations; checkpointing recomputes them in the backward pass
    # (same maths, same settings, slower) instead of shortening the input or the batch.
    model.model_body.gradient_checkpointing_enable()
    args = TrainingArguments(batch_size=G7B["batch_size"], num_epochs=G7B["num_epochs"],
                             num_iterations=G7B["num_iterations"], body_learning_rate=G7B["body_learning_rate"],
                             max_length=G7B["max_length"], seed=SEED)
    trainer = Trainer(model=model, args=args,
                      train_dataset=Dataset.from_dict({"text": f["text"].tolist(), "label": f["label"].tolist()}))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    fit = time.time() - t0
    pos = list(model.model_head.classes_).index(1)
    p = {name: np.asarray(model.predict_proba(texts, batch_size=32))[:, pos] for name, texts in eval_sets().items()}
    save_g_arm(f"g7b_setfit_k{k}", product_p=p["product"], hotel_p=p["hotel"], stress_p=p["stress"],
               config={"method": "SetFit", "hotel_labels_used": 2 * k, "k_per_class": k, **G7B},
               cost={"params": int(sum(q.numel() for q in model.model_body.parameters())), "fit_seconds": round(fit, 1),
                     "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["ettin", "setfit"], required=True)
    p.add_argument("--k", type=int, nargs="+", default=[8, 16, 25])
    a = p.parse_args()
    for k in a.k:
        (run_ettin if a.method == "ettin" else run_setfit)(k)


if __name__ == "__main__":
    main()
