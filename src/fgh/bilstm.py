"""(f) BiLSTM over word sequences: f5 = frozen GloVe 300-d vectors, f5b = embeddings learned from scratch.

  python src/fgh/bilstm.py --arm f5_bilstm_glove300_frozen
  python src/fgh/bilstm.py --arm f5b_bilstm_scratch
Fills the draft's empty "GloVe plus BiLSTM" row. Settings are fixed in advance, not tuned:
  vocabulary = every word in the training split (+ <pad>, <unk>); reviews cut to 256 words;
  embedding 300-d -> 1-layer BiLSTM, 128 units per direction -> max-pool over time -> dropout 0.3 -> linear 256->2;
  unweighted cross-entropy (as the final Ettin model), Adam lr 2e-3, batch 32, gradient clipping 1.0,
  up to 10 epochs, keeping the epoch with the best validation macro-F1 (same selection rule as Ettin); seed 42.
"""
import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, TEXT, Timer, peak_rss_gb, save_arm, splits  # noqa: E402
from glove_arms import load_glove, tokenize  # noqa: E402

MAX_WORDS, EMB_DIM, HIDDEN = 256, 300, 128
ARMS = {"f5_bilstm_glove300_frozen": True, "f5b_bilstm_scratch": False}  # arm -> use frozen GloVe


class BiLSTM(nn.Module):
    def __init__(self, emb, freeze, hidden=HIDDEN, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=-1e4, total_length=ids.shape[1])
        return self.out(self.drop(h.max(dim=1).values))  # padding (-1e4) never wins the max


def encode(texts, stoi):
    return [[stoi.get(w, 1) for w in tokenize(t)][:MAX_WORDS] or [1] for t in texts]


def batch(seqs, idx, device):
    rows = [seqs[i] for i in idx]
    lengths = torch.tensor([len(r) for r in rows])
    ids = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long)
    for j, r in enumerate(rows):
        ids[j, :len(r)] = torch.tensor(r)
    return ids.to(device), lengths


@torch.no_grad()
def predict(model, seqs, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(seqs), bs):
        ids, lengths = batch(seqs, range(i, min(i + bs, len(seqs))), device)
        out.append(torch.softmax(model(ids, lengths).float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    use_glove = ARMS[a.arm]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = splits()

    counts = Counter(w for t in tr[TEXT] for w in tokenize(t))
    itos = ["<pad>", "<unk>"] + sorted(counts)
    stoi = {w: i for i, w in enumerate(itos)}
    emb = rng.normal(0, 0.1, (len(itos), EMB_DIM)).astype(np.float32)
    emb[0] = 0
    glove_hits = None
    if use_glove:
        vecs = load_glove(EMB_DIM, keep=set(itos))
        glove_hits = sum(1 for w in itos[2:] if w in vecs)
        for w, i in stoi.items():
            if w in vecs:
                emb[i] = vecs[w]
    model = BiLSTM(torch.tensor(emb), freeze=use_glove).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()

    x_tr, x_va = encode(tr[TEXT], stoi), encode(va[TEXT], stoi)
    y_tr = torch.tensor(tr[LABEL].values.astype(int))
    y_va = va[LABEL].values.astype(int)
    gen = torch.Generator().manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = Timer()
    history, best_f1, best_epoch, best_state = [], -1.0, 0, None
    with t("fit"):
        for epoch in range(1, a.epochs + 1):
            model.train()
            perm = torch.randperm(len(x_tr), generator=gen).tolist()
            total = 0.0
            for i in range(0, len(perm), a.batch_size):
                idx = perm[i:i + a.batch_size]
                ids, lengths = batch(x_tr, idx, device)
                loss = loss_fn(model(ids, lengths), y_tr[idx].to(device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += loss.item() * len(idx)
            f1 = f1_score(y_va, predict(model, x_va, device) >= 0.5, average="macro")
            history.append({"epoch": epoch, "train_loss": round(total / len(perm), 4), "val_macro_f1": round(f1, 4)})
            print(history[-1], flush=True)
            if f1 > best_f1:
                best_f1, best_epoch, best_state = f1, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with t("predict"):
        p_pos = predict(model, x_va, device)
    n_trainable = sum(q.numel() for q in params)
    n_total = sum(q.numel() for q in model.parameters())
    save_arm(a.arm, va, p_pos,
             config={"family": "word sequence + RNN", "embeddings": "glove.6B.300d, frozen" if use_glove
                     else "random N(0, 0.1), trained", "vocab_size": len(itos), "max_words": MAX_WORDS,
                     "hidden_per_direction": HIDDEN, "pooling": "max over time", "dropout": 0.3,
                     "loss": "cross-entropy, no class weights", "optimizer": "Adam", "lr": a.lr,
                     "batch_size": a.batch_size, "epochs": a.epochs, "seed": a.seed, "device": device},
             cost={"feature_dim": EMB_DIM, "trainable_params": n_trainable, "total_params": n_total,
                   "fit_seconds": t.t["fit"], "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                   "peak_rss_gb": peak_rss_gb(),
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
                   "model_size_mb": round(n_total * 4 / 1e6, 1)},
             extra={"best_epoch": best_epoch, "history": history,
                    "glove_covers_vocab": round(glove_hits / (len(itos) - 2), 4) if glove_hits is not None else None})


if __name__ == "__main__":
    main()
