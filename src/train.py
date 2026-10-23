"""Fine-tune a pretrained transformer encoder for binary sentiment classification.

One call = one training run. Every run writes to outputs/runs/<run-name>/:
  config.json, metrics.json, history.json (per-step/epoch logs), val_predictions.csv
and appends one row to outputs/results.csv (the sweep ledger).
Optional: --save-model keeps the best checkpoint, --predict-test writes submission.csv.

Example (main model, defaults = the planned configuration):
  .venv/Scripts/python src/train.py --run-name ettin400m_lr2e-5
Smoke test (tiny subset, 1 epoch):
  .venv/Scripts/python src/train.py --run-name smoke --limit 64 --epochs 1
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, ROOT, TEXT, class_weights, load_splits, load_test  # noqa: E402

OUT = ROOT / "outputs"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-400m")
    # Hyperparameters (question c). Defaults = the planned configuration.
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="per-step micro-batch")
    p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--scheduler", default="linear", choices=["linear", "cosine", "constant_with_warmup"])
    p.add_argument("--max-length", type=int, default=512)
    # Default "none": on Ettin-400M, balanced weights scored 0.926 macro-F1 vs 0.947 without (2026-10-13 sweep).
    p.add_argument("--class-weights", default="none", choices=["balanced", "none"])
    p.add_argument("--neg-weight", type=float, default=None,
                   help="custom loss weight for the negative class (positive = 1.0); overrides --class-weights")
    p.add_argument("--seed", type=int, default=42)
    # Memory / hardware settings
    p.add_argument("--optim", default="adamw_bnb_8bit", help="adamw_bnb_8bit | adamw_torch | adamw_torch_fused")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--fp16", action="store_true", help="fp16 mixed precision (for GPUs without bf16, e.g. Colab T4)")
    p.add_argument("--no-grad-ckpt", action="store_true")
    # Data and outputs
    p.add_argument("--train-file", default="train.json")
    p.add_argument("--test-file", default="test.json")
    p.add_argument("--val-size", type=float, default=0.2, help="validation fraction (0.2 = 80/20, 0.1 = 90/10)")
    p.add_argument("--normalize", action="store_true",
                   help="groupmate preprocessing: whitespace normalisation + dedup on normalised text")
    p.add_argument("--normalize-text-only", action="store_true",
                   help="keep the original split, normalise whitespace afterwards (isolates preprocessing)")
    p.add_argument("--limit", type=int, default=0, help="subsample N train/val rows for smoke tests")
    p.add_argument("--save-model", action="store_true", help="keep the best checkpoint in run dir/model")
    p.add_argument("--predict-test", action="store_true", help="write run dir/submission.csv")
    return p.parse_args(argv)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok, max_length):
        self.enc = tok(list(texts), truncation=True, max_length=max_length)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy: L = -w_y log p_y."""

    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.loss_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss


def metrics_from_logits(logits, labels):
    preds = np.asarray(logits).argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro"),
            "neg_f1": f[0], "neg_precision": p[0], "neg_recall": r[0], "pos_f1": f[1]}


def run(args):
    run_dir = OUT / "runs" / args.run_name
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; choose another --run-name")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    tr, va = load_splits(args.train_file, val_size=args.val_size, normalize=args.normalize,
                         normalize_text_only=args.normalize_text_only)
    if args.limit:
        tr = tr.sample(min(args.limit, len(tr)), random_state=args.seed).reset_index(drop=True)
        va = va.sample(min(args.limit, len(va)), random_state=args.seed).reset_index(drop=True)
    weights = [args.neg_weight, 1.0] if args.neg_weight else class_weights(tr[LABEL], args.class_weights)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    train_ds = TextDataset(tr[TEXT], tr[LABEL], tok, args.max_length)
    val_ds = TextDataset(va[TEXT], va[LABEL], tok, args.max_length)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio of total steps
        weight_decay=args.weight_decay, lr_scheduler_type=args.scheduler,
        optim=args.optim, bf16=not args.no_bf16 and not args.fp16 and torch.cuda.is_available(),
        fp16=args.fp16,
        gradient_checkpointing=not args.no_grad_ckpt,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=25, report_to="none", seed=args.seed, dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda ev: metrics_from_logits(ev.predictions, ev.label_ids),
        loss_weights=weights,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    # Best checkpoint is loaded (load_best_model_at_end). Evaluate and save validation predictions.
    t1 = time.time()
    val_out = trainer.predict(val_ds)
    val_seconds = time.time() - t1
    probs = torch.softmax(torch.tensor(val_out.predictions, dtype=torch.float32), -1).numpy()
    pd.DataFrame({TEXT: va[TEXT], "label": va[LABEL], "pred": probs.argmax(-1), "p_pos": probs[:, 1]}) \
        .to_csv(run_dir / "val_predictions.csv", index=False)
    m = metrics_from_logits(val_out.predictions, va[LABEL].values)

    best_epoch = None
    evals = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    if evals:
        best_epoch = max(evals, key=lambda h: h["eval_macro_f1"])["epoch"]
    metrics = {**{k: round(float(v), 4) for k, v in m.items()},
               "best_epoch": best_epoch, "train_minutes": round(train_seconds / 60, 2),
               "val_seconds": round(val_seconds, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
               "n_train": len(tr), "n_val": len(va), "class_weights": [round(w, 3) for w in weights],
               "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    if args.predict_test:
        te = load_test(args.test_file, normalize=args.normalize or args.normalize_text_only)
        test_out = trainer.predict(TextDataset(te[TEXT], None, tok, args.max_length))
        pd.DataFrame({"id": range(len(te)), LABEL: test_out.predictions.argmax(-1)}) \
            .to_csv(run_dir / "submission.csv", index=False)
    if args.save_model:
        trainer.save_model(str(run_dir / "model"))
        tok.save_pretrained(str(run_dir / "model"))
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    ledger = OUT / "results.csv"
    row = {"run": args.run_name, "model": args.model, "lr": args.lr, "epochs": args.epochs,
           "eff_batch": args.batch_size * args.grad_accum, "warmup": args.warmup_ratio,
           "wd": args.weight_decay, "sched": args.scheduler, "max_len": args.max_length,
           "cw": f"neg{args.neg_weight}" if args.neg_weight else args.class_weights, "seed": args.seed, "limit": args.limit,
           **{k: metrics[k] for k in ["accuracy", "macro_f1", "neg_f1", "neg_precision", "neg_recall",
                                       "best_epoch", "train_minutes", "peak_gpu_gb"]}}
    new = not ledger.exists()
    with ledger.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run(parse_args())
