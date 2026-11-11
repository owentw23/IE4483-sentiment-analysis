| Measure | Products (validation) | Hotels (evaluation set) |
|---|---|---|
| Negative share (1-2 stars among 1-2 and 4-5) | 14.6% | 10.9% |
| 3-star share of all reviews (dropped from both) | dropped at source | 14.0% |
| Median length, words | 43 | 112 |
| Median length, Ettin tokens | 55 | 138 |
| 95th percentile, Ettin tokens | 177 | 458 |
| Reviews cut at 512 tokens | 0.14% | 3.95% |
| Word occurrences unseen in product training | 2.0% | 6.9% |
| Distinct words unseen in product training | 21.7% | 51.8% |
| Proxy A-distance (0 same, 2 separable) | -0.04 (train vs val) | 1.98 |

Most typical words (weighted log-odds): products them, these, they, it, size, my, are, love, fit, quality; hotels was, we, the, room, our, were, stay, there, at, us.
