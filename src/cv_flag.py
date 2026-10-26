"""Out-of-fold predictions on the training split, to flag likely mislabelled reviews (confident learning).

The training split (same 80/20 protocol as train.py) is cut into K stratified folds. For each fold a fresh
model trains on the other K-1 folds for a fixed number of epochs and predicts the held-out fold, so every
training review gets a prediction from a model that never saw it. The validation split is never touched.

Writes outputs/cv/<name>/oof_predictions.csv: reviews, label, p_pos, fold.
  .venv/Scripts/python src/cv_flag.py --name ettin400m_5fold
Smoke test:
  .venv/Scripts/python src/cv_flag.py --name smoke --limit 64 --epochs 1
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, SEED, TEXT, load_splits  # noqa: E402
from train import OUT, TextDataset, WeightedTrainer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-400m")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="subsample N training rows for smoke tests")
    return p.parse_args(argv)


def main(args):
    out_dir = OUT / "cv" / args.name
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists; choose another --name")
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    tr, _ = load_splits()
    if args.limit:
        tr = tr.sample(min(args.limit, len(tr)), random_state=SEED).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    p_pos = np.full(len(tr), np.nan)
    fold_of = np.full(len(tr), -1)

    for k, (fit_idx, held_idx) in enumerate(
            StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(tr, tr[LABEL])):
        t0 = time.time()
        set_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        fit, held = tr.iloc[fit_idx], tr.iloc[held_idx]
        targs = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"), learning_rate=args.lr, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 4,
            gradient_accumulation_steps=args.grad_accum, warmup_steps=0.1, weight_decay=0.01,
            optim="adamw_bnb_8bit", bf16=torch.cuda.is_available(), gradient_checkpointing=True,
            save_strategy="no", eval_strategy="no", logging_steps=50, report_to="none", seed=SEED,
            dataloader_num_workers=0,
        )
        trainer = WeightedTrainer(model=model, args=targs, data_collator=DataCollatorWithPadding(tok),
                                  train_dataset=TextDataset(fit[TEXT], fit[LABEL], tok, args.max_length),
                                  loss_weights=[1.0, 1.0])
        trainer.train()
        logits = trainer.predict(TextDataset(held[TEXT], None, tok, args.max_length)).predictions
        p_pos[held_idx] = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1)[:, 1].numpy()
        fold_of[held_idx] = k
        acc = ((p_pos[held_idx] > 0.5) == held[LABEL].values).mean()
        print(f"fold {k}: held-out acc {acc:.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
        del trainer, model
        torch.cuda.empty_cache()

    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    pd.DataFrame({TEXT: tr[TEXT], "label": tr[LABEL], "p_pos": p_pos, "fold": fold_of}) \
        .to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"wrote {out_dir / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main(parse_args())
