"""Grid search over hyperparameters. Each combination runs src/train.py in a fresh process
(so GPU memory is fully released between runs) and lands in outputs/results.csv.

Examples:
  # Stage 1 of tuning: learning-rate sweep for the main model
  .venv/Scripts/python src/sweep.py --tag lr --lr 1e-5 2e-5 3e-5
  # Ablation: class weights on vs off at the chosen learning rate
  .venv/Scripts/python src/sweep.py --tag cw --lr 2e-5 --class-weights balanced none
  # Baseline model with the same protocol
  .venv/Scripts/python src/sweep.py --tag deberta --model microsoft/deberta-v3-large --lr 1e-5 2e-5
  # Print the ledger sorted by macro-F1
  .venv/Scripts/python src/sweep.py --show
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "outputs" / "results.csv"
GRID = {  # flag -> type; every flag accepts several values
    "model": str, "lr": float, "epochs": int, "batch-size": int, "grad-accum": int,
    "warmup-ratio": float, "weight-decay": float, "scheduler": str, "max-length": int,
    "class-weights": str, "neg-weight": float, "seed": int,
}
SHORT = {"model": "", "lr": "lr", "epochs": "ep", "batch-size": "bs", "grad-accum": "ga",
         "warmup-ratio": "wu", "weight-decay": "wd", "scheduler": "", "max-length": "len",
         "class-weights": "cw", "neg-weight": "negw", "seed": "s"}


def show():
    if not LEDGER.exists():
        sys.exit("no results yet")
    df = pd.read_csv(LEDGER)
    cols = ["run", "lr", "epochs", "eff_batch", "warmup", "wd", "max_len", "cw", "accuracy", "macro_f1",
            "neg_f1", "neg_recall", "best_epoch", "train_minutes", "peak_gpu_gb"]
    print(df.sort_values("macro_f1", ascending=False)[cols].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="sweep", help="prefix for run names")
    p.add_argument("--show", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    for flag, typ in GRID.items():
        p.add_argument(f"--{flag}", type=typ, nargs="+")
    a, extra = p.parse_known_args()  # extra flags (e.g. --train-file X, --limit 64) pass through unchanged
    if a.show:
        return show()
    chosen = {f: getattr(a, f.replace("-", "_")) for f in GRID if getattr(a, f.replace("-", "_"))}
    combos = [dict(zip(chosen, vals)) for vals in itertools.product(*chosen.values())] or [{}]
    print(f"{len(combos)} run(s)")
    for combo in combos:
        varying = [k for k in chosen if len(chosen[k]) > 1] or list(chosen)
        name = a.tag + "".join(
            f"_{SHORT[k]}{str(combo[k]).split('/')[-1]}" for k in varying)
        cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--run-name", name]
        for k, v in combo.items():
            cmd += [f"--{k}", str(v)]
        cmd += extra
        print(">>", " ".join(cmd[1:]))
        if not a.dry_run:
            subprocess.run(cmd, check=False)
    if not a.dry_run and LEDGER.exists():
        show()


if __name__ == "__main__":
    main()
