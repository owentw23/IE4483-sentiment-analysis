"""(g6) Domain-adaptive pretraining: continue Ettin-400M's masked-language-model training on unlabelled hotel text.

  python src/fgh/dapt.py      # -> data/external/models/g6_dapt_mlm (then src/train.py fine-tunes it; run_g.sh)

Settings written down before the run (plan: "a separate masked-LM step whose settings are written down before it
runs"); after Gururangan et al. (2020), "Don't stop pretraining", scaled to one GPU:
  start          jhu-clsp/ettin-encoder-400m (the same checkpoint the submitted model started from)
  text           the 20,000 unlabelled hotel pool reviews, minus 1,000 held out to measure MLM loss; no labels,
                 no stars, no evaluation-set text
  masking        30% of tokens (Ettin and ModernBERT pretrain with 30%), standard 80/10/10 replacement
  length         512 tokens, truncated, no packing
  optimisation   2 epochs, learning rate 5e-5, linear decay, 10% warm-up, weight decay 0.01, effective batch 16
                 (8 x 2), 8-bit AdamW, bf16, gradient checkpointing, seed 42
  then           src/train.py fine-tunes this checkpoint on the product reviews with the submitted recipe, unchanged
Reported: MLM loss on 1,000 held-out pool reviews and on the 1,437 product validation reviews, before and after
(same masks for both, seed 42), so the report can show the model adapted to hotels without forgetting products.
Outputs: the checkpoint (git-ignored) and outputs/report_g/dapt.json.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORT_G, ROOT, TEXT, hotel_text, splits  # noqa: E402

BASE = "jhu-clsp/ettin-encoder-400m"
OUT = ROOT / "data" / "external" / "models" / "g6_dapt_mlm"
SEED = 42
CFG = {"base": BASE, "mlm_probability": 0.30, "max_length": 512, "epochs": 2, "lr": 5e-5, "warmup_ratio": 0.1,
       "weight_decay": 0.01, "batch_size": 8, "grad_accum": 2, "optim": "adamw_bnb_8bit", "bf16": True,
       "gradient_checkpointing": True, "seed": SEED, "held_out_pool_reviews": 1000}


class Enc(torch.utils.data.Dataset):
    def __init__(self, texts, tok):
        self.enc = tok(list(texts), truncation=True, max_length=CFG["max_length"])

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


@torch.no_grad()
def mlm_loss(model, ds, collator, bs=16):
    """Mean masked-token loss with a fixed masking seed, so before/after use identical masks."""
    model.eval()
    torch.manual_seed(SEED)  # the collator draws its masks from torch's global generator
    tot, n = 0.0, 0
    for i in range(0, len(ds), bs):
        b = collator([ds[j] for j in range(i, min(i + bs, len(ds)))])
        b = {k: v.to(model.device) for k, v in b.items()}
        out = model(**b)
        k = int((b["labels"] != -100).sum())
        tot, n = tot + float(out.loss) * k, n + k
    return round(tot / n, 4)


def main():
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMaskedLM.from_pretrained(BASE).to("cuda")
    pool = hotel_text("pool").sample(frac=1, random_state=SEED)["text"].tolist()
    held, train_texts = pool[:CFG["held_out_pool_reviews"]], pool[CFG["held_out_pool_reviews"]:]
    _, va = splits()
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=CFG["mlm_probability"])
    ds_train, ds_held, ds_prod = Enc(train_texts, tok), Enc(held, tok), Enc(va[TEXT], tok)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        before = {"hotel_held_out": mlm_loss(model, ds_held, collator), "product_val": mlm_loss(model, ds_prod, collator)}
    print("MLM loss before:", before, flush=True)

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"), learning_rate=CFG["lr"], num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["batch_size"], gradient_accumulation_steps=CFG["grad_accum"],
        warmup_steps=CFG["warmup_ratio"], weight_decay=CFG["weight_decay"], lr_scheduler_type="linear",
        optim=CFG["optim"], bf16=True, gradient_checkpointing=True, save_strategy="no", logging_steps=50,
        report_to="none", seed=SEED, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, data_collator=collator)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = round((time.time() - t0) / 60, 2)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        after = {"hotel_held_out": mlm_loss(model, ds_held, collator), "product_val": mlm_loss(model, ds_prod, collator)}
    print("MLM loss after:", after, flush=True)
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    report = {"config": CFG, "n_train_reviews": len(ds_train), "train_minutes": minutes,
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
              "mlm_loss_before": before, "mlm_loss_after": after, "checkpoint": str(OUT.relative_to(ROOT)),
              "train_loss_log": [h for h in trainer.state.log_history if "loss" in h]}
    REPORT_G.mkdir(parents=True, exist_ok=True)
    (REPORT_G / "dapt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "train_loss_log"}, indent=1))


if __name__ == "__main__":
    main()
