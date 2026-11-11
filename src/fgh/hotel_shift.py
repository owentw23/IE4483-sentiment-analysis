"""(g) Domain-shift diagnostics between our product reviews and the hotel reviews (plan item G5, Table G2).

  python src/fgh/hotel_shift.py       # after hotel_data.py; CPU + tokenizer only, about a minute

Measures, all on review text only (the hotel stars are used just for the class-balance line):
  * class balance: negative share in product training data vs the hotel corpus and the hotel evaluation set
  * length: words and Ettin tokens per review (the submitted model's tokenizer), share longer than 512 tokens,
    i.e. the share the submitted model sees cut short
  * vocabulary: share of word occurrences (and of distinct words) in the hotel evaluation set that never occur in
    the product training split, with product validation as the in-domain baseline (CountVectorizer defaults, the
    same tokens the TF-IDF arms use)
  * typical words: weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), product
    reviews (all 7,185) against an equal-size random sample of the hotel pool
  * proxy A-distance (Ben-David et al. 2007; used by Blitzer et al. 2007): train a linear classifier to tell the
    two domains apart; PAD = 2 (1 - 2 err) with err the 5-fold cross-validated error. 0 = indistinguishable,
    2 = perfectly separable. Reference: product training vs product validation reviews (same domain).
Output: outputs/report_g/shift.json (no review text).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, REPORT_G, TEXT, hotel_key, hotel_text, splits  # noqa: E402

SEED = 42
TOKENIZER = "jhu-clsp/ettin-encoder-400m"  # same tokenizer as ettin400m_final


def lengths(tok, texts):
    n_tok = np.array([len(ids) for ids in tok(list(texts), add_special_tokens=True)["input_ids"]])
    n_words = pd.Series(list(texts)).str.split().str.len().values
    return {"n": len(n_tok), "words_median": float(np.median(n_words)), "tokens_median": float(np.median(n_tok)),
            "tokens_p95": float(np.percentile(n_tok, 95)), "share_over_512_tokens": round(float((n_tok > 512).mean()), 4)}


def oov(vocab, analyzer, texts):
    toks = [w for t in texts for w in analyzer(t)]
    types = set(toks)
    unseen_tokens = sum(w not in vocab for w in toks)
    unseen_types = sum(w not in vocab for w in types)
    # counts kept as well as shares, so tables can round from the exact fraction rather than a rounded share
    return {"token_oov_share": round(unseen_tokens / len(toks), 4), "type_oov_share": round(unseen_types / len(types), 4),
            "n_tokens": len(toks), "n_types": len(types), "n_unseen_tokens": unseen_tokens, "n_unseen_types": unseen_types}


def log_odds(a_texts, b_texts, top=15, a0=1000.0):
    """Monroe et al. (2008) weighted log-odds, informative Dirichlet prior from the pooled counts."""
    cv = CountVectorizer(min_df=5)
    x = cv.fit_transform(list(a_texts) + list(b_texts))
    ya = np.asarray(x[:len(a_texts)].sum(0)).ravel().astype(float)
    yb = np.asarray(x[len(a_texts):].sum(0)).ravel().astype(float)
    na, nb = ya.sum(), yb.sum()
    alpha = a0 * (ya + yb) / (na + nb)
    delta = (np.log((ya + alpha) / (na + a0 - ya - alpha)) - np.log((yb + alpha) / (nb + a0 - yb - alpha)))
    z = delta / np.sqrt(1 / (ya + alpha) + 1 / (yb + alpha))
    words = np.array(cv.get_feature_names_out())
    o = np.argsort(z)
    return ([{"word": words[i], "z": round(float(z[i]), 1)} for i in o[::-1][:top]],
            [{"word": words[i], "z": round(float(z[i]), 1)} for i in o[:top]])


def proxy_a_distance(a_texts, b_texts):
    x = list(a_texts) + list(b_texts)
    y = np.r_[np.zeros(len(a_texts)), np.ones(len(b_texts))]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=5000))
    acc = cross_val_score(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy").mean()
    err = 1 - acc
    return {"n_per_domain": len(a_texts), "cv_error": round(float(err), 4), "pad": round(float(2 * (1 - 2 * err)), 3)}


def main():
    tr, va = splits()
    ev, ev_key, pool = hotel_text("eval"), hotel_key("eval"), hotel_text("pool")
    split = json.loads((REPORT_G / "hotel_split.json").read_text())
    out = {"class_balance": {
        "product_train_negative_share": round(float((tr[LABEL] == 0).mean()), 4),
        "hotel_corpus_negative_share_excl_3star": split["corpus_negative_share_excl_3star"],
        "hotel_eval_negative_share": round(float((ev_key["label"] == 0).mean()), 4),
        "hotel_corpus_3star_share": round(split["corpus_stars"]["3"] / sum(split["corpus_stars"].values()), 4)}}

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    out["length"] = {"product_train": lengths(tok, tr[TEXT]), "product_val": lengths(tok, va[TEXT]),
                     "hotel_eval": lengths(tok, ev["text"]), "hotel_pool": lengths(tok, pool["text"])}

    cv = CountVectorizer().fit(tr[TEXT])
    vocab, analyzer = cv.vocabulary_, cv.build_analyzer()
    out["vocabulary"] = {"product_train_vocab_size": len(vocab),
                         "product_val_vs_train": oov(vocab, analyzer, va[TEXT]),
                         "hotel_eval_vs_product_train": oov(vocab, analyzer, ev["text"])}

    product_all = pd.concat([tr[TEXT], va[TEXT]]).tolist()
    hotel_sample = pool["text"].sample(len(product_all), random_state=SEED).tolist()
    top_product, top_hotel = log_odds(product_all, hotel_sample)
    out["typical_words"] = {"product": top_product, "hotel": top_hotel}

    out["proxy_a_distance"] = {
        "product_vs_hotel": proxy_a_distance(product_all, hotel_sample),
        "reference_product_train_vs_val": proxy_a_distance(tr[TEXT].sample(len(va), random_state=SEED).tolist(),
                                                           va[TEXT].tolist())}
    REPORT_G.mkdir(parents=True, exist_ok=True)
    (REPORT_G / "shift.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
