"""(g) Methods that need no hotel labels and no training: score products and hotels with the same fixed rule.

  python src/fgh/hotel_arms.py --list
  python src/fgh/hotel_arms.py --arm g1_ettin_final --pool     # also label the 20k unlabelled pool (teacher, g5)
  python src/fgh/hotel_arms.py --arm g2s_ettin68m_nli --check  # G6 check: load, label 10 product reviews, no files

Every arm scores the same three sets: the 1,437 product validation reviews, the 2,000-review hotel evaluation set
and the 300 three-star hotel reviews (data/external/hotel/, built by hotel_data.py). Methods see review text only;
the answer key is read by common.save_g_arm when scoring. Results: outputs/runs/<arm>/ and outputs/results_g.csv.

Rules fixed before any hotel score was seen (plan, 7 Nov):
  * NLI arms ask whether the review entails "This review is positive." or "This review is negative." and take
    whichever entailment logit is higher (softmax over the two = the standard zero-shot pipeline, multi_label off).
  * GLiClass and Decision-Kai get the labels "positive" / "negative"; the higher score wins.
  * Max length 512 tokens for every model (the submitted model's setting); NLI truncates the review, never the
    hypothesis. Model input is the review text without its title (g1t is the one title + text variant).
  * Decision rule P(positive) >= 0.5 everywhere. Nothing is tuned on hotel stars.

What each model learned from (checked on the model cards / dataset files, 8 Nov):
  g0  TF-IDF 1-2-gram + LogReg (balanced, C by 5-fold CV), trained on our 5,748 product reviews
  g1  ettin400m_final, the submitted model, trained on our product reviews
  g2  dleemiller/finecat-nli-v2-l: Ettin-400M fine-tuned on FineCat-NLI-Reason (2.05M pairs, distilled from a
      27B teacher). Its `zs_train` source (111,777 pairs) contains labelled review-sentiment pairs ("The sentiment
      in this example amazon product review is positive", IMDb, Yelp, app reviews), so it is NOT label-free; it is
      a reference like SiEBERT.
  g2s AdamCodd/ettin-nli-classifier: Ettin-68M fine-tuned on SNLI + MultiNLI only, the label-free Ettin arm (no
      NLI-only Ettin-400M exists on the Hub as of 8 Nov).
  g3  MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c: Mixtral-written synthetic task data + MNLI + FEVER-NLI; no
      human-labelled review data
  g4  knowledgator/gliclass-modern-large-v3.0: ModernBERT-large, synthetic and logic data (needs `gliclass`)
  g4b siebert/sentiment-roberta-large-english: 15 labelled sentiment sets incl. Blitzer 2007 and Yelp (reference)
  g4c vllm-sr/Decision-2.0-Kai-0.6B: Qwen3-0.6B decision model, training data not documented
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LABEL, ROOT, TEXT, Timer, c_record, hotel_inputs, hotel_text, peak_rss_gb,  # noqa: E402
                    save_g_arm, select_c, splits)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_LEN = 512
HYP = ("This review is positive.", "This review is negative.")
LABELS = ("positive", "negative")
FINAL = ROOT / "outputs" / "runs" / "ettin400m_final" / "model"
MODELS = ROOT / "data" / "external" / "models"
TFIDF_C_GRID = (0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000)  # same grid as tfidf_arms.py

ARMS = {
    "g0_tfidf_products": ("tfidf", None, "our 5,748 product reviews"),
    "g1_ettin_final": ("ettin_final", str(FINAL), "our 5,748 product reviews"),
    "g1t_ettin_final_title": ("ettin_final", str(FINAL), "our 5,748 product reviews"),
    "g2_finecat_nli_v2": ("nli", "dleemiller/finecat-nli-v2-l",
                          "NLI + zero-shot pairs incl. labelled Amazon/IMDb/Yelp/app review sentiment"),
    "g2s_ettin68m_nli": ("nli", "AdamCodd/ettin-nli-classifier", "SNLI + MultiNLI only (no sentiment labels)"),
    "g3_deberta_zeroshot_c": ("nli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c",
                              "synthetic task data (Mixtral) + MNLI + FEVER-NLI; no human-labelled reviews"),
    "g4_gliclass_modern_large": ("gliclass", "knowledgator/gliclass-modern-large-v3.0", "synthetic + logic data"),
    "g4b_siebert": ("siebert", "siebert/sentiment-roberta-large-english",
                    "15 labelled sentiment datasets incl. Blitzer 2007 and Yelp"),
    "g4c_decision_kai": ("decision", "vllm-sr/Decision-2.0-Kai-0.6B", "not documented"),
    # students trained by run_g.sh (src/train.py, submitted recipe); checkpoints in data/external/models/ (git-ignored)
    "g5_selftrain": ("ettin_final", str(MODELS / "g5_selftrain"),
                     "our product reviews + hotel pool pseudo-labelled by our model (confidence >= 0.99)"),
    "g5b_selftrain_agree": ("ettin_final", str(MODELS / "g5b_selftrain_agree"),
                            "as g5, keeping only pseudo-labels that g2 agrees with"),
    "g6_dapt": ("ettin_final", str(MODELS / "g6_dapt"),
                "our product reviews, after masked-LM pretraining on unlabelled hotel pool text"),
}


def by_length(texts, fn, bs):
    """Run fn on batches of similar length (less padding), return results in the original order."""
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty(len(texts), dtype=float)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        out[idx] = fn([texts[j] for j in idx])
    return out


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ------------------------------------------------------------------------------------------- scorers
def tfidf_scorer(_):
    tr, _ = splits()
    search = select_c([("vec", TfidfVectorizer(ngram_range=(1, 2)))], tr[TEXT], tr[LABEL].astype(int), TFIDF_C_GRID,
                      class_weight="balanced")
    best = search.best_estimator_
    c_config, _ = c_record(search, TFIDF_C_GRID)
    info = {"classifier": repr(best), **c_config, "feature_dim": len(best.named_steps["vec"].vocabulary_)}
    return (lambda texts: best.predict_proba(texts)[:, 1]), info


def seqcls_scorer(name, pos_index=None):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    if pos_index is None:
        pos_index = next(int(i) for i, lab in model.config.id2label.items() if str(lab).upper().startswith("POS"))

    @torch.no_grad()
    def batch(texts):
        b = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        return torch.softmax(model(**b).logits.float(), -1)[:, pos_index].cpu().numpy()

    return (lambda texts: by_length(texts, batch, 32)), {"params": n_params(model), "positive_index": pos_index}


def nli_scorer(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    labels = {int(i): str(lab).lower() for i, lab in model.config.id2label.items()}
    ent = next(i for i, lab in labels.items() if "entail" in lab and "not" not in lab)

    @torch.no_grad()
    def batch(texts):
        n = len(texts)
        b = tok(texts + texts, [HYP[0]] * n + [HYP[1]] * n, truncation="only_first", max_length=MAX_LEN,
                padding=True, return_tensors="pt").to(DEVICE)
        e = model(**b).logits.float()[:, ent]
        return torch.softmax(torch.stack([e[:n], e[n:]], 1), 1)[:, 0].cpu().numpy()

    return (lambda texts: by_length(texts, batch, 16)), {"params": n_params(model), "id2label": labels,
                                                         "entailment_index": ent, "hypotheses": HYP}


def gliclass_scorer(name):
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    model = GLiClassModel.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label", max_length=MAX_LEN,
                                          device="cuda:0" if DEVICE == "cuda" else "cpu", progress_bar=False)

    def batch(texts):
        res = pipe(texts, list(LABELS), threshold=0.0, batch_size=len(texts))
        out = []
        for r in res:
            s = {x["label"]: float(x["score"]) for x in r}
            pos, neg = s.get(LABELS[0], 0.0), s.get(LABELS[1], 0.0)
            out.append(pos / (pos + neg) if pos + neg > 0 else 0.5)
        return np.array(out)

    return (lambda texts: by_length(texts, batch, 16)), {"params": n_params(model), "labels": LABELS,
                                                         "score": "sigmoid scores, normalised pos / (pos + neg)"}


def decision_scorer(name):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    question = {"sentiment": {"type": "choice", "instructions": "What is the sentiment of this review?",
                              "criteria": {LABELS[0]: HYP[0], LABELS[1]: HYP[1]}}}

    def one(text):
        text = tok.decode(tok(text, add_special_tokens=False)["input_ids"][:MAX_LEN])  # same 512-token budget
        ans = model.system_one(state=text, questions=question)["answers"]["sentiment"]
        probs = ans.get("probabilities") or ans.get("probs") or ans
        return float(probs[LABELS[0]]) / (float(probs[LABELS[0]]) + float(probs[LABELS[1]]))

    @torch.no_grad()
    def batch(texts):
        return np.array([one(t) for t in texts])

    return batch, {"params": n_params(model), "question": question}


def build(kind, name, arm):
    if kind == "tfidf":
        return tfidf_scorer(name)
    if kind in ("ettin_final", "siebert"):
        if kind == "ettin_final":
            w = Path(name) / "model.safetensors"
            if not w.exists() or w.stat().st_size < 10_000_000:
                sys.exit(f"{w} is missing or a Git LFS pointer; run `git lfs pull`")
        return seqcls_scorer(name, pos_index=1 if kind == "ettin_final" else None)
    if kind == "nli":
        return nli_scorer(name)
    if kind == "gliclass":
        return gliclass_scorer(name)
    if kind == "decision":
        return decision_scorer(name)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=list(ARMS))
    p.add_argument("--list", action="store_true")
    p.add_argument("--pool", action="store_true", help="also score the 20k unlabelled pool (pool_predictions.csv)")
    p.add_argument("--check", action="store_true", help="G6: load and label 10 product reviews, write nothing")
    a = p.parse_args()
    if a.list or not a.arm:
        print("\n".join(ARMS))
        return
    kind, name, seen = ARMS[a.arm]
    mode = "title_text" if a.arm.startswith("g1t") else "text"
    t = Timer()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with t("load"):
        score, info = build(kind, name, a.arm)
    _, va = splits()

    if a.check:
        s = pd.concat([va[va[LABEL] == c].sample(5, random_state=42) for c in (0, 1)])
        pp = score(s[TEXT].tolist())
        y = s[LABEL].values.astype(int)
        print(json.dumps({"arm": a.arm, "check_correct": int(((pp >= 0.5).astype(int) == y).sum()), "of": len(y),
                          "p_pos": [round(float(v), 3) for v in pp], "labels": y.tolist(),
                          "load_seconds": t.t["load"], "info": info}, indent=1, default=str))
        return

    sets = {"product": va[TEXT].tolist(), "hotel": hotel_inputs(hotel_text("eval"), mode),
            "stress": hotel_inputs(hotel_text("stress3"), mode)}
    if a.pool:
        sets["pool"] = hotel_inputs(hotel_text("pool"), mode)
    out = {}
    for k, texts in sets.items():
        with t(k):
            out[k] = score(texts)
    gpu = round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEVICE == "cuda" else None
    save_g_arm(a.arm, product_p=out["product"], hotel_p=out["hotel"], stress_p=out["stress"], pool_p=out.get("pool"),
               config={"method": kind, "model": name, "labelled_data_seen": seen, "hotel_labels_used": 0,
                       "input": mode, "max_length": MAX_LEN, "dtype": str(DTYPE), "decision_rule": "P(pos) >= 0.5",
                       **{k: v for k, v in info.items() if k != "params"}},
               cost={"params": info.get("params"), "model_load_seconds": t.t["load"],
                     "predict_seconds_per_1k_hotel": round(t.t["hotel"] / len(sets["hotel"]) * 1000, 3),
                     "predict_seconds_per_1k_product": round(t.t["product"] / len(sets["product"]) * 1000, 3),
                     "pool_seconds": t.t.get("pool"), "peak_gpu_gb": gpu, "peak_rss_gb": peak_rss_gb()})


if __name__ == "__main__":
    main()
