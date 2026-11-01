# Compute-Optimal Decoding for Personalized LLM Alignment (COPAD)

## Overview

This repository provides an experimental framework for reproducing **Personalized Alignment at Decoding-time (PAD)** and investigating compute-optimal strategies for test-time personalized text generation. 

It builds upon the official PAD implementation ([Chen et al., ICLR 2025](https://arxiv.org/abs/2408.03314)), which steers a base language model's generation using a personalized reward model (PRM) at decoding time without modifying the base model's weights. While decoding-time steering eliminates the need to train a separate policy per user preference profile, standard PAD evaluates the 8B-parameter PRM at **every single generated token**. This roughly doubles memory consumption and slows down generation by over 5 times.

As identified in the original paper's limitations, not all tokens require personalized steering. Some tokens (for example, high-confidence tokens) are generated identically regardless of user preference. 

This repository investigates whether personalized alignment quality can be maintained while reducing reward-model computation, for example by selectively applying steering at tokens where it is most useful. This project explores two compute-optimal decoding strategies:

1. **Fixed-interval scoring:** Invoking the PRM once every $k$ tokens ($k \in \{1, 2, 4, 8\}$) and decoding the intervening tokens with the base model alone.
2. **Confidence-gated scoring:** Invoking the PRM only when the base model's next-token distribution is uncertain (measured by a small margin between the logits of the top-2 candidate tokens), otherwise decoding with base model.

Both methods operate as modifications to PAD's decoding loop without retraining either the base LLM or the PRM. Both models are loaded in 4-bit precision via `bitsandbytes`, enabling execution within **~11.2 GB peak VRAM** on a single 16 GB GPU (such as a Kaggle/Colab T4).

---

## Repository Structure

```text
COPAD/
├── PAD/
│   ├── inference/
│   │   ├── configs/                      # Run configs for baseline, k1-k8, and conf_p12_5-p50
│   │   ├── results/                      # Step 1, fixed-interval, and confidence-gated results
│   │   │   ├── step1/                    # Baseline reproduction logs and summary tables
│   │   │   ├── fixed-interval/           # Sweeps for k ∈ {1, 2, 4, 8}
│   │   │   └── confidence-gated/         # Sweeps for p ∈ {12.5%, 25%, 50%} margin percentiles
│   │   ├── collect_model_outs_4bit.py    # Baseline reproduction runner
│   │   ├── collect_model_outs_strided.py # Flexible runner for fixed-interval and confidence-gated decoding
│   │   ├── PAD_4bit.py                   # 4-bit baseline PAD decoding loop
│   │   ├── PAD_4bit_strided.py           # Core decoding loop with stride and margin gating support
│   │   └── measure_reward.py             # Reward scoring across helpfulness, harmlessness, and humor
├── requirements.txt
└── README.md
```

---

## 1. Clone the Repository

Clone the repository and navigate into the `PAD/` directory:

```bash
git clone https://github.com/Ayush32Kumar/Compute-optimal-PAD.git
cd Compute-optimal-PAD/PAD/
```

## 2. Create a Python Environment

---

```bash
python3.10 -m venv pad_env
source pad_env/bin/activate
```

Upgrade `pip` and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r ../requirements.txt
```

Set the required environment and Python search paths:

```bash
export PYTHONPATH="..:$PWD:$PWD/LLaMAFactory/src"
export MPLBACKEND=Agg
```

---

## 3. Running The Experiments

### Step 1: Base Model vs. Full PAD Baseline (k=1)

Compare the unaligned base model against full token-level PAD on P-Soups:

```bash
python ./inference/collect_model_outs_4bit.py \
    --run_percent 100.0 \
    --config="configs/psoups_baseline_compare.config" \
    --out_file="results/step1/psoups_step1" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --max_new_token=128 \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. [Principles] harmless and helpfulness and humor [Instruction] "
```

*Outputs will be saved with a `_0.jsonl` suffix (e.g., `results/step1/psoups_step1_0.jsonl`).*

---

### Step 2: Compute-Efficient Decoding Strategies

#### Strategy A: Fixed-Interval Scoring ($k \in \{1, 2, 4, 8\}$)
Evaluate scoring every $k$-th token (e.g., $k=4$, which invokes the PRM once every 4 tokens):

```bash
# Substitute k4 with k1, k2, or k8 in the config and output path as needed
python ./inference/collect_model_outs_strided.py \
    --run_percent 100.0 \
    --config="configs/fixed_interval_k4.config" \
    --out_file="results/fixed-interval/k4" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --max_new_token=128 \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. [Principles] harmless and helpfulness and humor [Instruction] "
```

#### Strategy B: Confidence-Gated Scoring ($p \in \{12.5\%, 25\%, 50\%\}$)
Call the PRM only when the top-2 candidate token logit margin is below a calibrated threshold $\tau$:

* `p12.5` ($\tau = 0.28125$): targets ~16 RM calls per 128 tokens
* `p25` ($\tau = 0.71484$): targets ~32 RM calls per 128 tokens
* `p50` ($\tau = 1.78125$): targets ~64 RM calls per 128 tokens

```bash
# Run confidence-gated decoding with p25 threshold
python ./inference/collect_model_outs_strided.py \
    --run_percent 100.0 \
    --config="configs/conf_p25.config" \
    --out_file="results/confidence-gated/conf_p25" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --max_new_token=128 \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. [Principles] harmless and helpfulness and humor [Instruction] "
```

---

## 4. Evaluation & Reward Measurement

To score generated responses across Helpfulness, Harmlessness, and Humor, pass the `.jsonl` output file directly to `measure_reward.py`. Reward values are saved as `.npy` arrays inside `results/<subfolder>/scores/`.

```bash
TARGET_FILE="results/fixed-interval/k4_0.jsonl"

# 1. Helpfulness
python ./inference/measure_reward.py \
    --out_file="$TARGET_FILE" \
    --tokenizer="Ray2333/gpt2-large-helpful-reward_model" \
    --rm="Ray2333/gpt2-large-helpful-reward_model" \
    --rm_gpu="cuda:0"

# 2. Harmlessness
python ./inference/measure_reward.py \
    --out_file="$TARGET_FILE" \
    --tokenizer="Ray2333/gpt2-large-harmless-reward_model" \
    --rm="Ray2333/gpt2-large-harmless-reward_model" \
    --rm_gpu="cuda:0"

# 3. Humor
python ./inference/measure_reward.py \
    --out_file="$TARGET_FILE" \
    --tokenizer="mohameddhiab/humor-no-humor" \
    --rm="mohameddhiab/humor-no-humor" \
    --rm_gpu="cuda:0"
```

---

## 5. Experimental Results

Experiments were evaluated on the 3-dimensional **P-Soups** benchmark ($n=50$, max 128 new tokens, greedy decoding, rm_weight=0.8 using 4-bit quantization with both models co-located on a single GPU (`cuda:0`).

### Main Performance & Efficiency Comparison

| Strategy / Setting | RM Calls / Resp (min–max) | Helpful | Harmless | Humor | Humor Win-Rate vs Base | Tokens/s | Speedup vs Full PAD ($k=1$) | Peak GPU VRAM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Unaligned Base** | 0.0 (0–0) | +0.640 | +0.966 | -1.069 | — | 3.606 | **5.39x** | 11,079 MB |
| **Gated p12.5** | 12.9 (6–24) | +0.836 | +0.841 | -0.849 | 70% | 2.789 | **4.17x** | 11,222 MB |
| **Fixed $k=8$** | 16.0 (16–16) | +1.003 | +0.789 | -0.904 | 64% | 2.349 | **3.51x** | 11,228 MB |
| **Fixed $k=4$** | 32.0 (32–32) | +0.829 | +1.023 | -0.930 | 62% | 1.730 | **2.58x** | 11,234 MB |
| **Gated p25** | 32.0 (18–48) | +0.802 | +0.980 | **-0.254** | **82%** | 1.669 | **2.49x** | 11,233 MB |
| **Fixed $k=2$** | 64.0 (64–64) | +1.047 | +0.882 | -0.532 | 72% | 1.132 | **1.69x** | 11,237 MB |
| **Gated p50** | 65.1 (43–86) | +0.786 | +0.858 | **+0.252** | **84%** | 1.080 | **1.61x** | 11,236 MB |
| **Full PAD ($k=1$)** | 128.0 (128–128) | +0.738 | +0.925 | +0.704 | 82% | 0.669 | **1.00x** | 11,238 MB |

### Key Findings & Compute-Quality Trade-Offs

1. **Inference Bottleneck:** Full PAD ($k=1$) slows inference by **5.4 times** compared to the unaligned base model (0.67 tok/s vs 3.61 tok/s, 191.2s vs 35.5s per response).
2. **Confidence-Gating Outperforms Uniform Striding at Matched Compute:**
   * At an identical 32-call budget (25% compute), **Confidence-Gated (`p25`) vastly outperforms Fixed-Interval ($k=4$) on Humor** ($-0.254$ vs $-0.930$, $p = 0.007$, achieving an **82% vs 62% win-rate** against the base model).
   * At a 64-call budget, Confidence-Gated (`p50`) yields $+0.252$ humor vs $-0.532$ for $k=2$ ($p = 0.033$).
3. **Adaptive Temporal Allocation:** Confidence gating naturally front-loads compute to early decision steps where steering establishes context:
   * **Steps 0–7:** Gated `p25` allocates **35%** of its steering calls (and `p50` allocates **67%**), whereas uniform striding allocates a fixed 25% and 50% regardless of token difficulty.
4. **Memory Footprint:** Because both the 4-bit LLM and PRM remain resident on GPU during inference, peak memory is consistently ~11.2 GB across all settings.
