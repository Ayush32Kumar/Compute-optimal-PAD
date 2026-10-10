# Compute-Optimal Decoding for Personalized LLM Alignment

## Overview

This repository provides an experimental framework for reproducing Personalized Alignment at Decoding-time (PAD) and 
investigating more compute-efficient personalized text generation. It builds on the official PAD implementation, which 
steers a language model's generation using a personalized reward model (PRM) at decoding time, without modifying the base 
model's weights. This avoids training a separate policy per preference profile.
The current setup focuses on running PAD's released P-Soups configuration with the Llama-3-Base-8B-SFT language model and 
the RuizheChen/PAD reward-model checkpoint. It includes a 4-bit quantized inference implementation to reduce GPU memory requirements 
and scripts for collecting generated responses and inference timing.
A central motivation for this repository is the computational cost of applying personalized steering at every decoding step. 
Since the reward model is invoked repeatedly during generation, inference becomes slower and more memory-intensive. 
This repository investigates whether personalized alignment quality can be maintained while reducing 
reward-model computation, for example by selectively applying steering at tokens where it is most useful.

i. Fixed-interval scoring:

Call the PRM once every k tokens and decode the intervening tokens with the base model alone.

ii. Confidence-gated scoring:

Call the PRM only when the base model is somewhat uncertain (small margin between logits of its top-2 candidate next tokens),
and fall back to plain greedy/sampled decoding otherwise.

## 1. Clone the Repository

Clone PAD and navigate into the repository:

```bash
git clone https://github.com/Ayush32Kumar/Compute-optimal-PAD.git
cd Compute-Optimal-PAD/PAD/
```

## 2. Create a Python Environment

Make sure Python 3.10 and the `venv` module are installed.
Create and activate a virtual environment:

```bash
python3.10 -m venv pad_env
source pad_env/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 3. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Set the required Python import paths:

```bash
export PYTHONPATH="..:$PWD:$PWD/LLaMAFactory/src"
export MPLBACKEND=Agg
```

## 4. Run the Experiment

To compare base lm vs PAD:

```bash
python ./inference/collect_model_outs_4bit.py \
    --run_percent 100.0 \
    --config="configs/psoups_baseline_compare.config" \
    --out_file="results/psoups_step1" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --max_new_token=128 \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. [Principles] harmless and helpfulness and humor [Instruction] "
```
## 5. To run the evaluations

```bash
cd inference

#helpfulness

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="Ray2333/gpt2-large-helpful-reward_model" \
    --rm="Ray2333/gpt2-large-helpful-reward_model" \
    --rm_gpu="cuda:0"

#harmless

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="Ray2333/gpt2-large-harmless-reward_model" \
    --rm="Ray2333/gpt2-large-harmless-reward_model" \
    --rm_gpu="cuda:0"

#humor

python measure_reward.py \
  --out_file="<your_results>.jsonl" \
  --tokenizer="mohameddhiab/humor-no-humor" \
  --rm="mohameddhiab/humor-no-humor" \
  --rm_gpu="cuda:0"
```

## 6. Compute-efficient decoding strategies

i. Fixed-interval Scoring for k=4 (i.e. PRM called once every 4 tokens)

```bash
#replace k4 with k1, k2, k8 for other configurations

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

* Using Python 3.10 and the project environment to avoid dependency conflicts.
* The custom 4-bit implementation reduces GPU memory usage.
* Ensure the required model checkpoints are accessible and sufficient GPU memory is available.
