"""(f9) The submitted model (ettin400m_final) predicting validation at max length 128, 256 and 512 tokens.

  python src/fgh/ettin_maxlen.py                                                    # submitted model (f9_ettin_final_len*)
  python src/fgh/ettin_maxlen.py --run ettin400m_final_3090 --arm-prefix f9_ettin3090   # its 3090 retrain (report Table 12)
Needs the real weights (Git LFS): `git lfs install && git lfs pull`. Inference only: shows how much accuracy
and time the truncation length costs. 256 was the original setting in PreProcessingPipeline/preprocess.py.
Token-length coverage is measured with the model's own tokenizer.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, TEXT, Timer, peak_rss_gb, save_arm, splits  # noqa: E402

RUNS_DIR = ROOT / "outputs" / "runs"


@torch.no_grad()
def predict(tok, model, texts, max_length, device, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=max_length, padding=True,
                return_tensors="pt").to(device)
        out.append(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--run", default="ettin400m_final", help="src/train.py run whose saved model/ is re-scored")
    p.add_argument("--arm-prefix", default="f9_ettin_final", help="arm names are <prefix>_len<L>")
    a = p.parse_args()
    MODEL_DIR = RUNS_DIR / a.run / "model"
    weights = MODEL_DIR / "model.safetensors"
    if not weights.exists() or weights.stat().st_size < 10_000_000:
        sys.exit(f"{weights} is missing or still a Git LFS pointer; run `git lfs install && git lfs pull` first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
    tr, va = splits()
    lengths_tr = np.array([len(ids) for ids in tok(tr[TEXT].tolist())["input_ids"]])
    lengths_va = np.array([len(ids) for ids in tok(va[TEXT].tolist())["input_ids"]])
    n_params = sum(q.numel() for q in model.parameters())
    for L in a.lengths:
        t = Timer()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with t("predict"):
            p_pos = predict(tok, model, va[TEXT].tolist(), L, device)
        save_arm(f"{a.arm_prefix}_len{L}", va, p_pos,
                 config={"family": "fine-tuned contextual (inference only)", "model": str(MODEL_DIR),
                         "max_length": L, "dtype": str(dtype)},
                 cost={"feature_dim": 1024, "trainable_params": n_params, "fit_seconds": None,
                       "predict_seconds_per_1k": round(t.t["predict"] / len(va) * 1000, 3),
                       "peak_rss_gb": peak_rss_gb(),
                       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
                       "model_size_mb": round(n_params * 4 / 1e6, 1)},
                 extra={"train_reviews_within_length": round(float((lengths_tr <= L).mean()), 4),
                        "val_reviews_within_length": round(float((lengths_va <= L).mean()), 4)})


if __name__ == "__main__":
    main()
