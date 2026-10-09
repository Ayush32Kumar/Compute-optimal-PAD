# Step 2: Fixed-interval scoring (P-Soups, n=50, 4-bit, greedy, rm_weight=0.8)

## Main table
| Setting | RM calls/resp | Helpful | Harmless | Humor | Tokens/s | s/response | Speedup vs k=1 | Peak GPU mem (MB) |
|---|---|---|---|---|---|---|---|---|
| No RM (base) | 0 | +0.640 | +0.966 | -1.069 | 3.606 | 35.5 | 5.39x | 11079 |
| k=8 | 16 | +1.003 | +0.789 | -0.904 | 2.349 | 54.5 | 3.51x | 11228 |
| k=4 | 32 | +0.829 | +1.023 | -0.930 | 1.730 | 74.0 | 2.58x | 11234 |
| k=2 | 64 | +1.047 | +0.882 | -0.532 | 1.132 | 113.1 | 1.69x | 11237 |
| k=1 | 128 | +0.738 | +0.925 | +0.704 | 0.669 | 191.2 | 1.00x | 11238 |

## Reward detail: mean ± standard error
| Setting | Helpful | Harmless | Humor |
|---|---|---|---|
| No RM (base) | +0.640 ± 0.202 | +0.966 ± 0.107 | -1.069 ± 0.191 |
| k=8 | +1.003 ± 0.187 | +0.789 ± 0.114 | -0.904 ± 0.190 |
| k=4 | +0.829 ± 0.201 | +1.023 ± 0.118 | -0.930 ± 0.177 |
| k=2 | +1.047 ± 0.184 | +0.882 ± 0.107 | -0.532 ± 0.201 |
| k=1 | +0.738 ± 0.187 | +0.925 ± 0.100 | +0.704 ± 0.222 |

## Win-rate vs unaligned base (RM score higher on the same prompt)
| Setting | Helpful | Harmless | Humor |
|---|---|---|---|
| k=8 | 54% | 44% | 64% |
| k=4 | 60% | 42% | 62% |
| k=2 | 66% | 46% | 72% |
| k=1 | 54% | 48% | 82% |

## Notes
- Tokens/s = 128 / mean seconds per response (generation always runs the full 128 steps, prefill included).
- Peak GPU memory is cuda:0 with the 4-bit LLM and 4-bit RM co-located (the RM-device bug in PAD_4bit.py), so it is combined LLM+RM memory.
- "No RM" is Step 1's rm_weight=0 run: the RM is loaded but never called. Its memory therefore still includes the RM weights.
- Paper reference, P-Soups 3-dim, RM column: PAD helpful 0.96 / harmless 0.85 / humor 0.75; base 1.06 / 0.83 / -0.93.
- Check: k=1 outputs identical to Step 1 PAD outputs on 50/50 prompts.
- Differences smaller than about 2x the standard error are within noise at n=50.
