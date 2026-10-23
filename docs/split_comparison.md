# Split comparison: 80/20 vs 90/10 (with groupmate preprocessing)

**Goal:** decide which validation split the team uses. Both runs apply Richie's preprocessing
(`--normalize`: whitespace normalisation, dedup on normalised text, drop conflicting labels) and the
current best settings: Ettin-400M, no class weights, lr 2e-5, 3 epochs, max length 512.

| Run | Split | Train / val | Expected time (RTX 4060, 8 GB) |
|---|---|---|---|
| `ettin400m_8020_norm` | 80/20 | 5,745 / 1,437 | ~12 min |
| `ettin400m_9010_norm` | 90/10 | 6,463 / 719 | ~13–14 min |

Reference (original protocol, no normalisation, 80/20): `ettin400m_lr2e-5_noweights` →
acc 0.973, macro-F1 0.947, neg-F1 0.909.

## Steps (Windows laptop, inside the repo folder)

1. Check there are no local changes (should print nothing besides untracked files):
   ```bash
   git status --short
   ```
2. Get the branch:
   ```bash
   git fetch
   ```
   ```bash
   git switch exp/split-comparison
   ```
3. Run 80/20:
   ```bash
   .venv/Scripts/python src/train.py --run-name ettin400m_8020_norm --normalize
   ```
4. Run 90/10:
   ```bash
   .venv/Scripts/python src/train.py --run-name ettin400m_9010_norm --normalize --val-size 0.1
   ```
5. Check both rows were added:
   ```bash
   .venv/Scripts/python src/sweep.py --show
   ```
6. Push the results back so they can be analysed from any machine:
   ```bash
   git add outputs/results.csv outputs/runs/ettin400m_8020_norm outputs/runs/ettin400m_9010_norm
   ```
   ```bash
   git commit -m "exp: 80/20 vs 90/10 split comparison results"
   ```
   ```bash
   git push
   ```

If a run name already exists, the script stops; pick a new `--run-name` (e.g. add `_v2`).
If the GPU runs out of memory, add `--batch-size 4 --grad-accum 4` (same effective batch of 16).

## How to read the results

- Each run is scored on its **own** validation set (1,437 vs 719 reviews), so the numbers are not
  directly comparable. Noise is about ±1 point (80/20) and ±1.3 points (90/10).
- A gap of ≤ 1 point does not show either split is better.
- Trade-off: 90/10 trains on 718 more reviews; 80/20 gives a more reliable validation score for the report.
- Also compare `outputs/runs/<run>/metrics.json` → `neg_precision`, `neg_recall`, `best_epoch`.

## Decision to record afterwards

Write the chosen split in `CLAUDE.md` (Validation protocol) and `PROGRESS.md`, then merge
`exp/split-comparison` into `main`.
