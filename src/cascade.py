"""Cascade: Ettin-400M encoder for confident reviews, Qwen3-8B (4-bit) with nearest-neighbour examples for the rest.

A review goes to the LLM when Ettin's confidence max(p, 1-p) is below a threshold. The LLM sees the 8 most similar
training-split reviews with their labels (retrieved with Qwen3-Embedding-0.6B, the review itself excluded) and is
scored by its next-token probability of "positive" vs "negative" (thinking disabled, nothing is generated).

The threshold is chosen on the 5-fold out-of-fold training predictions (src/cv_flag.py), never on validation.
Inputs: outputs/cv/<cv-run>/oof_predictions.csv, outputs/runs/<ettin-run>/val_predictions.csv and the test
probabilities from src/predict.py (<ettin-run>/test_check_with_probs.csv). LLM probabilities are cached per review.

  .venv/Scripts/python src/cascade.py --run-name cascade_qwen3-8b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABEL, ROOT, TEXT, load_splits  # noqa: E402

THRESHOLDS = [0.9, 0.99, 0.995, 0.999]
SYSTEM = ("You classify Amazon product reviews. Each label comes from the star rating the customer gave: "
          "'positive' = high rating, 'negative' = low rating. Mixed reviews are labelled by the customer's overall rating.")
QUERY = "Instruct: Given a product review, retrieve reviews with the same overall customer rating\nQuery:"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--ettin-run", default="ettin400m_final")
    p.add_argument("--cv-run", default="ettin400m_5fold")
    p.add_argument("--llm", default="Qwen/Qwen3-8B")
    p.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=1200, help="characters kept per review in the prompt")
    return p.parse_args()


@torch.no_grad()
def embed(name, texts, query):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    m = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    texts = [QUERY + t for t in texts] if query else list(texts)
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        h = m(**b).last_hidden_state[:, -1].float()  # left padding: last position = last token
        out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def metrics(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "neg_f1": round(f1_score(y, pred, pos_label=0), 4)}


def cascade(p_ettin, p_llm, t):
    """Ettin prediction unless its confidence is below t, then the LLM's."""
    routed = np.maximum(p_ettin, 1 - p_ettin) < t
    return np.where(routed, p_llm > 0.5, p_ettin > 0.5).astype(int), routed


def main(a):
    run_dir = ROOT / "outputs" / "runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2))

    tr, _ = load_splits()
    oof = pd.read_csv(ROOT / "outputs" / "cv" / a.cv_run / "oof_predictions.csv")
    va = pd.read_csv(ROOT / "outputs" / "runs" / a.ettin_run / "val_predictions.csv")
    te = pd.read_csv(ROOT / "outputs" / "runs" / a.ettin_run / "test_check_with_probs.csv")
    sets = {"oof": (oof[TEXT].values, oof["p_pos"].values), "val": (va[TEXT].values, va["p_pos"].values),
            "test": (te[TEXT].values, te["p_pos"].values)}

    # Reviews any candidate threshold could route (the largest threshold routes a superset of the others).
    need = sorted({t for texts, p in sets.values() for t in texts[np.maximum(p, 1 - p) < max(THRESHOLDS)]})
    cache_path = run_dir / "llm_cache.csv"
    cache = pd.read_csv(cache_path).set_index(TEXT)["p_llm"].to_dict() if cache_path.exists() else {}
    todo = [t for t in need if t not in cache]
    print(f"routed candidates: {len(need)} ({len(todo)} not cached)", flush=True)

    if todo:
        pool, pool_y = tr[TEXT].tolist(), tr[LABEL].values.astype(int)
        e_pool, e_q = embed(a.embedder, pool, False), embed(a.embedder, todo, True)
        sims = e_q @ e_pool.T
        where = {t: j for j, t in enumerate(pool)}
        for i, t in enumerate(todo):  # never show a review its own label
            if t in where:
                sims[i, where[t]] = -np.inf

        tok = AutoTokenizer.from_pretrained(a.llm)
        llm = AutoModelForCausalLM.from_pretrained(a.llm, device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)).eval()
        ids_lab = [tok.encode("negative")[0], tok.encode("positive")[0]]
        clip = lambda t: t[:a.max_chars] + (" ..." if len(t) > a.max_chars else "")
        name = {0: "negative", 1: "positive"}
        with torch.no_grad():
            for n, (text, s) in enumerate(zip(todo, sims)):
                shots = np.argsort(-s)[:a.shots][::-1]  # most similar example placed last
                ex = "".join(f"Review: {clip(pool[j])}\nLabel: {name[pool_y[j]]}\n\n" for j in shots)
                user = f"Labelled examples:\n\n{ex}Review: {clip(text)}\nLabel? Answer with one word: positive or negative."
                enc = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_tensors="pt", return_dict=True).to("cuda")
                logits = llm(**enc).logits[0, -1, ids_lab].float()
                cache[text] = torch.softmax(logits, -1)[1].item()
                if (n + 1) % 50 == 0 or n + 1 == len(todo):
                    pd.DataFrame({TEXT: list(cache), "p_llm": list(cache.values())}).to_csv(cache_path, index=False)
                    print(f"  LLM {n + 1}/{len(todo)}", flush=True)

    p_llm = {k: np.array([cache.get(t, np.nan) for t in texts]) for k, (texts, _) in sets.items()}

    # 1) choose the threshold on out-of-fold training predictions
    y_oof = oof["label"].values
    rows, report = [], {"oof_ettin": metrics(y_oof, (sets["oof"][1] > 0.5).astype(int))}
    for t in THRESHOLDS:
        pred, routed = cascade(sets["oof"][1], p_llm["oof"], t)
        rows.append({"threshold": t, "routed": int(routed.sum()), **metrics(y_oof, pred)})
    oof_tab = pd.DataFrame(rows)
    best = oof_tab.loc[oof_tab["macro_f1"].idxmax(), "threshold"]  # idxmax keeps the first (smallest) on ties
    report["oof"] = rows
    report["chosen_threshold"] = float(best)
    print("\nOOF (training, 5-fold): Ettin", report["oof_ettin"]); print(oof_tab.to_string(index=False))

    # 2) validation: report every threshold, the chosen one is the result
    y_va, base = va["label"].values, (va["p_pos"].values > 0.5).astype(int)
    rows = []
    for t in THRESHOLDS:
        pred, routed = cascade(sets["val"][1], p_llm["val"], t)
        fixed, broke = int(((base != y_va) & (pred == y_va)).sum()), int(((base == y_va) & (pred != y_va)).sum())
        p_mc = binomtest(broke, fixed + broke, 0.5).pvalue if fixed + broke else 1.0
        rows.append({"threshold": t, "routed": int(routed.sum()), **metrics(y_va, pred),
                     "fixed": fixed, "broke": broke, "mcnemar_p": round(p_mc, 3)})
        if t == best:
            pd.DataFrame({TEXT: va[TEXT], "label": y_va, "ettin_p": va["p_pos"], "llm_p": p_llm["val"],
                          "routed": routed, "pred": pred}).to_csv(run_dir / "val_predictions.csv", index=False)
    val_tab = pd.DataFrame(rows)
    report["val_ettin"] = metrics(y_va, base)
    report["val"] = rows
    print("\nValidation: Ettin", report["val_ettin"]); print(val_tab.to_string(index=False))

    # 3) test predictions with the chosen threshold
    pred, routed = cascade(sets["test"][1], p_llm["test"], best)
    pd.DataFrame({"id": te["id"], LABEL: pred}).to_csv(run_dir / "submission.csv", index=False)
    changed = int((pred != (sets["test"][1] > 0.5)).sum())
    report["test"] = {"routed": int(routed.sum()), "changed_vs_ettin": changed, "pred_negative": int((pred == 0).sum())}
    print(f"\nTest: threshold {best}, routed {int(routed.sum())} of {len(pred)}, changed vs Ettin {changed}, "
          f"predicted negative {int((pred == 0).sum())}")
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
