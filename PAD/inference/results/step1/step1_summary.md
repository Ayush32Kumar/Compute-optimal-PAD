# Step 1: Baseline Reproduction Summary

**Config:** 3-dim P-Soups (rm_weight=0.8) vs. unaligned baseline (rm_weight=0.0)
**Dataset:** P-Soups, n=50
**Quantization:** 4-bit, both LLM and PAD reward model

## Reward scores (RM column, our models: Ray2333 helpful/harmless, mohameddhiab humor)
| Dim | PAD | Baseline | Paper PAD (RM) | Paper Base (RM) |
|---|---|---|---|---|
| helpful | +0.738 | +0.640 | +0.96 | +1.06 |
| harmless | +0.925 | +0.966 | +0.85 | +0.83 |
| humor | +0.704 | -1.069 | +0.75 | -0.93 |

## Win-rate (PAD score > baseline score, per prompt)
- helpful: 54.0% (n=50)
- harmless: 48.0% (n=50)
- humor: 82.0% (n=50)

## Timing
- PAD: 0.666 tok/s avg, 192.9s/response avg
- Baseline: 3.620 tok/s avg, 35.5s/response avg
- PAD is ~0.18x the baseline's speed (i.e. baseline runs 5x faster than PAD)

## Memory
- Avg combined peak (LLM+RM, both on cuda:0): 11238 MB (PAD) / 11079 MB (baseline)

