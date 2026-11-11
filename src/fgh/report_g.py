"""(g) Build Table G1, Table G2, the example reviews and Figure G1 from outputs/results_g.csv and outputs/report_g/.

  python src/fgh/report_g.py      # CPU only, seconds
Writes outputs/report_g/table_g1.csv, table_g1.md, table_g2.md and g1_product_vs_hotel.png (all text-free), and the
example reviews to data/external/hotel/examples_g.md (git-ignored, because they quote hotel review text).
Example reviews are picked by a fixed rule, not by hand: the shortest hotel evaluation review in each category.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOTEL, LEDGER_G, REPORT_G, RUNS, hotel_key, hotel_text  # noqa: E402

NAMES = {  # arm -> (short name, hotel labels used, what the model learned sentiment from)
    "g0_tfidf_products": ("TF-IDF + LogReg (ours)", 0, "our product reviews"),
    "g1_ettin_final": ("Ettin-400M, submitted model", 0, "our product reviews"),
    "g1t_ettin_final_title": ("Same, title + text input", 0, "our product reviews"),
    "g2_finecat_nli_v2": ("Ettin-400M NLI (finecat-v2)", 0, "NLI + other labelled reviews"),
    "g2s_ettin68m_nli": ("Ettin-68M NLI (SNLI+MNLI)", 0, "none (NLI only)"),
    "g3_deberta_zeroshot_c": ("DeBERTa-v3-large zero-shot -c", 0, "none (synthetic + NLI)"),
    "g4_gliclass_modern_large": ("GLiClass ModernBERT-large", 0, "none (synthetic + logic)"),
    "g4b_siebert": ("SiEBERT (RoBERTa-large)", 0, "15 labelled review sets"),
    "g4c_decision_kai": ("Decision-2.0-Kai-0.6B", 0, "not documented"),
    "g5_selftrain": ("Self-training (Ettin student)", 0, "ours + pseudo-labelled hotels"),
    "g5b_selftrain_agree": ("Self-training, agreement filter", 0, "ours + pseudo-labelled hotels"),
    "g6_dapt": ("Domain-adaptive pretraining + Ettin", 0, "ours (after hotel MLM)"),
}
for k in (8, 16, 25):
    NAMES[f"g7a_ettin_k{k}"] = (f"Ettin again, {k}/class", 2 * k, "ours + few hotel labels")
    NAMES[f"g7b_setfit_k{k}"] = (f"SetFit, {k}/class", 2 * k, "few hotel labels only")
COLS = ["arm", "method", "hotel_labels", "learned_from", "product_macro_f1", "hotel_accuracy", "hotel_macro_f1",
        "hotel_neg_f1", "hotel_neg_precision", "hotel_neg_recall", "drop_macro_f1", "hotel_predicted_negative_share",
        "hotel_mcnemar_p_vs_g1", "stress3_share_pos", "predict_seconds_per_1k_hotel"]


def table_g1():
    t = pd.read_csv(LEDGER_G)
    t["method"] = t["arm"].map(lambda a: NAMES.get(a, (a,))[0])
    t["hotel_labels"] = t["arm"].map(lambda a: NAMES.get(a, (a, None))[1])
    t["learned_from"] = t["arm"].map(lambda a: NAMES.get(a, (a, None, ""))[2])
    t = t[[c for c in COLS if c in t.columns]].sort_values("hotel_macro_f1", ascending=False).reset_index(drop=True)
    t.to_csv(REPORT_G / "table_g1.csv", index=False)
    (REPORT_G / "table_g1.md").write_text(t.to_markdown(index=False))
    return t


def table_g2():
    s = json.loads((REPORT_G / "shift.json").read_text())
    L, V, P = s["length"], s["vocabulary"], s["proxy_a_distance"]
    rows = [("Negative share (1-2 stars among 1-2 and 4-5)", f"{s['class_balance']['product_train_negative_share']:.1%}",
             f"{s['class_balance']['hotel_eval_negative_share']:.1%}"),
            ("3-star share of all reviews (dropped from both)", "dropped at source",
             f"{s['class_balance']['hotel_corpus_3star_share']:.1%}"),
            ("Median length, words", f"{L['product_val']['words_median']:.0f}", f"{L['hotel_eval']['words_median']:.0f}"),
            ("Median length, Ettin tokens", f"{L['product_val']['tokens_median']:.0f}",
             f"{L['hotel_eval']['tokens_median']:.0f}"),
            ("95th percentile, Ettin tokens", f"{L['product_val']['tokens_p95']:.0f}", f"{L['hotel_eval']['tokens_p95']:.0f}"),
            ("Reviews cut at 512 tokens", f"{L['product_val']['share_over_512_tokens']:.2%}",
             f"{L['hotel_eval']['share_over_512_tokens']:.2%}"),
            ("Word occurrences unseen in product training", f"{V['product_val_vs_train']['token_oov_share']:.1%}",
             f"{V['hotel_eval_vs_product_train']['token_oov_share']:.1%}"),
            ("Distinct words unseen in product training", f"{V['product_val_vs_train']['type_oov_share']:.1%}",
             f"{V['hotel_eval_vs_product_train']['type_oov_share']:.1%}"),
            ("Proxy A-distance (0 same, 2 separable)", f"{P['reference_product_train_vs_val']['pad']:.2f} (train vs val)",
             f"{P['product_vs_hotel']['pad']:.2f}")]
    md = "| Measure | Products (validation) | Hotels (evaluation set) |\n|---|---|---|\n"
    md += "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    md += ("\n\nMost typical words (weighted log-odds): products "
           + ", ".join(w["word"] for w in s["typical_words"]["product"][:10]) + "; hotels "
           + ", ".join(w["word"] for w in s["typical_words"]["hotel"][:10]) + ".\n")
    (REPORT_G / "table_g2.md").write_text(md)
    return md


def examples():
    ev = hotel_text("eval").merge(hotel_key("eval"), on="review_id")
    g1 = pd.read_csv(RUNS / "g1_ettin_final" / "hotel_predictions.csv").set_index("review_id")
    g0 = pd.read_csv(RUNS / "g0_tfidf_products" / "hotel_predictions.csv").set_index("review_id")
    ev["g1"], ev["g0"] = ev["review_id"].map(g1["p_pos"]), ev["review_id"].map(g0["p_pos"])
    ev["len"] = ev["text"].str.len()
    neg = ev[ev["label"] == 0]
    cases = {"Transfer works: negative review, our Ettin right, our TF-IDF wrong": neg[(neg.g1 < 0.5) & (neg.g0 >= 0.5)],
             "Our Ettin misses a negative review (called positive)": neg[neg.g1 >= 0.5],
             "Our Ettin calls a positive review negative": ev[(ev["label"] == 1) & (ev.g1 < 0.5)]}
    md = "Rule: the shortest hotel evaluation review in each category (no hand picking).\n"
    for title, d in cases.items():
        md += f"\n### {title} ({len(d)} reviews in this category)\n"
        if len(d):
            r = d.sort_values("len").iloc[0]
            md += (f"\nreview_id {r.review_id}, {int(r.overall)} stars; P(positive): Ettin {r.g1:.3f}, "
                   f"TF-IDF {r.g0:.3f}\n\n> {r.text}\n")
    # Review text is hotel data: it goes next to the hotel files (git-ignored), not into the tracked outputs/.
    (HOTEL / "examples_g.md").write_text(md)


FIG_GROUPS = ["Product models, as they are", "Ready-made models", "Unlabelled hotel text", "A few labelled hotel reviews"]
FIG_ROWS = [  # (arm, label, group): every arm, grouped by how many hotel labels a method needs
    ("g1_ettin_final", "Submitted Ettin-400M", 0), ("g1t_ettin_final_title", "Same, title + text", 0),
    ("g0_tfidf_products", "TF-IDF + logistic regression", 0),
    ("g4b_siebert", "SiEBERT", 1), ("g4c_decision_kai", "Decision-2.0-Kai-0.6B", 1),
    ("g2_finecat_nli_v2", "Ettin-400M NLI (finecat-v2)", 1), ("g4_gliclass_modern_large", "GLiClass, zero-shot", 1),
    ("g3_deberta_zeroshot_c", "DeBERTa-v3 NLI, zero-shot", 1), ("g2s_ettin68m_nli", "Ettin-68M NLI, zero-shot", 1),
    ("g5_selftrain", "Self-training", 2), ("g5b_selftrain_agree", "Self-training, agreement filter", 2),
    ("g6_dapt", "Domain-adaptive pretraining", 2),
] + [(f"g7a_ettin_k{k}", f"Submitted Ettin trained further, {k}/class", 3) for k in (8, 16, 25)] \
  + [(f"g7b_setfit_k{k}", f"SetFit, {k}/class", 3) for k in (8, 16, 25)]


REPORT_GROUPS = ["Product models, as they are", "Ready-made models", "Unlabelled hotel text",
                 "50 labelled hotel reviews"]
REPORT_ROWS = [  # the methods in report Table 13, same order (Figure 4 shows nothing the table does not)
    ("g1_ettin_final", "Fine-tuned Ettin final encoder", 0), ("g0_tfidf_products", "TF-IDF + balanced logistic regression", 0),
    ("g4b_siebert", "SiEBERT", 1), ("g4_gliclass_modern_large", "GLiClass, zero-shot", 1),
    ("g3_deberta_zeroshot_c", "DeBERTa-v3 NLI, zero-shot", 1),
    ("g6_dapt", "Domain-adaptive pretraining", 2), ("g5_selftrain", "Self-training", 2),
    ("g7a_ettin_k25", "Ettin final encoder trained further", 3), ("g7b_setfit_k25", "SetFit", 3),
]


def figure(t, rows_spec=None, groups=None, name="g1_product_vs_hotel.png", width=7.6):
    # Dumbbell: product vs hotel macro-F1 per arm on one shared axis, rows grouped as in report Table 12.
    # Colours follow the artifact's validated categorical order (slot 1 blue = products, slot 2 orange = hotels);
    # shapes differ too (circle vs square) so identity never rests on colour alone. The product circle is drawn on
    # top of the hotel square, so both stay visible when the two scores coincide.
    t = t.set_index("arm")
    rows = [r for r in (rows_spec or FIG_ROWS) if r[0] in t.index]
    groups = groups or FIG_GROUPS
    y, labels, heads, pos = [], [], [], 0.0
    for i, (arm, label, g) in enumerate(rows):
        if i == 0 or g != rows[i - 1][2]:
            pos += 0.4 if i else 0.0
            heads.append((pos, groups[g]))
            pos += 0.9
        y.append(pos)
        labels.append(label)
        pos += 1
    prod = [t.loc[r[0], "product_macro_f1"] for r in rows]
    hotel = [t.loc[r[0], "hotel_macro_f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(width, 0.36 * pos + 1.3))
    ref = t.loc["g1_ettin_final", "hotel_macro_f1"] if "g1_ettin_final" in t.index else None
    if ref is not None:
        ax.axvline(ref, color="#586473", lw=1, ls=(0, (4, 3)), alpha=0.8, zorder=0, label="Submitted model on hotels")
    ax.hlines(y, prod, hotel, color="#c3cbd4", lw=2, zorder=1)
    ax.scatter(hotel, y, s=44, marker="s", color="#eb6834", edgecolor="white", linewidth=1.2, zorder=2,
               label="Hotel evaluation set (2,000 reviews)")
    ax.scatter(prod, y, s=40, marker="o", color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=3,
               label="Product validation (1,437 reviews)")
    ax.set_yticks(y, labels, fontsize=8.5)
    for yy, head in heads:
        ax.annotate(head, xy=(0, yy), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, fontweight="bold", color="#16212c")
    ax.set_ylim(pos - 0.4, -0.6)
    ax.set_xlim(0.775, 0.975)
    ax.set_xlabel("macro-F1")
    ax.grid(axis="x", color="#e6eaef")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_G / name, dpi=200, bbox_inches="tight")  # the legend sits above the axes


def main():
    REPORT_G.mkdir(parents=True, exist_ok=True)
    t = table_g1()
    print(t.to_string(index=False))
    print(table_g2())
    examples()
    figure(t)
    figure(pd.read_csv(LEDGER_G), REPORT_ROWS, REPORT_GROUPS, "g4_report_product_vs_hotel.png", width=6.6)
    print(f"wrote Table G1, Table G2, examples and Figure G1 to {REPORT_G}")


if __name__ == "__main__":
    main()
