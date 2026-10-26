# Synthetic negative reviews

800 negative reviews written by Claude (Opus 5.5) on 2026-10-26 for the synthetic-negatives ablation
(instructor approved). Only training-split negatives (80/20 protocol, seed 42) were read to learn the style;
no validation or test review was used as a model. Used only as extra **training** data, never for validation.

Files: `batch_01.txt` … `batch_09.txt` (802 reviews; batch 9 is 30 long reviews added to match the real length
tail). Format: a header line `@@ p=<persona> t=<type>` followed by the review text (may span several lines).
`src/synthetic.py` parses, post-processes, dedups and samples 800 into `synthetic_negatives.json`
(330 mixed, 107 implicit, 363 plain; max TF-IDF cosine to any real train/test review 0.34, so no near-copies).

## Style profile of the real training negatives (839 reviews)

- Products: Amazon apparel and accessories, mid-2000s: shoes/boots/slippers (~22%), clothing incl. bras,
  underwear, socks (~32%), watches/jewellery (~15%), bags/wallets, iPod/Zen/DS cases, FM transmitters.
- Complaints: sizing (runs small/large, wrong size chart; 36% mention size/fit), cheap quality (thin, shrinks,
  falls apart, stitching, zippers), not as described (not leather/cotton, wrong colour, not as pictured),
  seller/shipping (never arrived, wrong item, return hassle, restocking fees; 16% mention "return"), discomfort.
- Many are mixed ("looks great but..."), some sarcastic, some are one-liners, some long rants with pros/cons.
- Length (words): p10 20, p25 31, median 50, p75 77, p90 116.
- Formatting: 69% end without final punctuation (the scrape strips it; sometimes the last letter too),
  46% use double spaces after sentences, 22% contain line breaks, 1.3% are all caps, 7% start lowercase,
  frequent typos and ellipses.

## Personas

| id | persona |
|---|---|
| P01 | Practical mum buying kids' clothes and shoes; mentions ages, budget, school |
| P02 | Retired brand loyalist; decades of buying the same item, quality decline, formal tone, sometimes signs off |
| P03 | College student; lowercase, slang, abbreviations, typos |
| P04 | Plus-size woman; sizing and fit frustration, bras, dresses |
| P05 | New or nursing mum; nursing bras, baby clothes, diaper bags |
| P06 | All-caps ranter; seller, shipping and customer-service failures |
| P07 | Gadget buyer; iPod/Zen/PSP cases, FM transmitters, watches, headphones; technical details |
| P08 | Overseas or non-native English buyer; grammar quirks, customs, international shipping |
| P09 | Gift buyer; spouse, girlfriend, parent, niece; embarrassed by the gift |
| P10 | Athlete; running, hiking, cycling, swimming, gym gear that fails in use |
| P11 | Sarcastic humorist |
| P12 | Methodical reviewer; pros/cons lists, numbered points, measurements |
| P13 | Bride or formal-event shopper; dresses, shoes, hosiery, deadlines |
| P14 | Workwear buyer; nurse scrubs, work boots, server shoes, office wear |
| P15 | Fashion-focused young woman; "cute but..." |
| P16 | Terse one-liner |

## Types

- `mixed`: praises something but the overall verdict is clearly negative (the hard case the model misses).
- `implicit`: negative without obvious negative words (understatement, sarcasm, facts that imply failure).
- `plain`: ordinary negative review.
