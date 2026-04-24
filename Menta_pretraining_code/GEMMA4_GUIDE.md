# Gemma 4B Fine-tuning Guide

This document summarises the changes made to add Gemma 4B support to the Menta training pipeline and provides step-by-step instructions for smoke testing, full pre-training, and evaluation.

---

## What Was Changed

Gemma 4B support was added **without modifying any existing files**. The Qwen3-4B pipeline continues to work exactly as before. Three new files were created:

| File | Purpose |
|------|---------|
| `gemma4_lora_trainer.py` | Subclass of `Qwen3LoRAMultiTaskTrainer` with one Gemma-specific fix |
| `config_gemma4.yaml` | Hyperparameter config targeting `google/gemma-4-E4B-it` |
| `Menta_gemma4_lora_config1_logprob.py` | Entry point for Config-1 + log-prob training and evaluation |

### Why only one code change was needed

Gemma 4B and Qwen3-4B share the same LoRA-compatible layer names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), the same causal language modelling architecture, and compatible plain-text prompt formats. The only behavioural difference is that Gemma's tokenizer defaults to **left-padding**, which breaks fine-tuning. `gemma4_lora_trainer.py` overrides `_setup_model_and_tokenizer` to set `padding_side = 'right'` immediately after the parent class loads the tokenizer.

---

## Prerequisites

### 1. Python environment

```bash
cd Menta_pretraining_code
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.8+, CUDA 12.4+, and at least 16 GB GPU VRAM for the 4B model with 8-bit quantisation.

### 2. Data files

Place the four CSV datasets in `data/`:

```
data/
├── dreaddit_StressAnalysis - Sheet1.csv
├── Reddit_depression_dataset.csv
├── SDCNL.csv
└── 500_Reddit_users_posts_labels.csv
```

### 3. HuggingFace access

Gemma 4B is a gated model. Accept the licence at [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), then authenticate:

```bash
huggingface-cli login
```

### 4. Output directories

```bash
mkdir -p cache/huggingface cache/nltk_data cache/wandb gemma4_outputs
```

---

## Stage 1 — Smoke Test: Tokenizer Check (no GPU required)

Verifies that the model ID is correct, Gemma's tokeniser is accessible, and the digit labels used by the log-prob evaluator (`"0"`–`"5"`) each encode as a single token.

```bash
python -c "
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('google/gemma-4-E4B-it')
print('padding_side (before fix):', tok.padding_side)   # expect: left

tok.padding_side = 'right'
print('padding_side (after fix):', tok.padding_side)    # expect: right
print('pad_token:               ', tok.pad_token)       # expect: <pad>

print()
print('Label token check (all must be single-token):')
for t in ['0', '1', '2', '3', '4', '5']:
    ids = tok.encode(t, add_special_tokens=False)
    status = 'OK' if len(ids) == 1 else 'MULTI-TOKEN — logprob fallback will apply'
    print(f'  {t!r} -> token IDs {ids}  [{status}]')
"
```

**Expected output:**

```
padding_side (before fix): left
padding_side (after fix):  right
pad_token:                 <pad>

Label token check (all must be single-token):
  '0' -> token IDs [235276]  [OK]
  '1' -> token IDs [235274]  [OK]
  ...
```

---

## Stage 2 — Smoke Test: Short Training Run (GPU required)

Runs one epoch over all six tasks with a batch size of 1 and a reduced sequence length. The goal is to confirm that the full forward/backward pass, LoRA attachment, and tokeniser fix are all working before committing to a multi-hour training job.

Create `smoke_test_gemma4.py` in `Menta_pretraining_code/`:

```python
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemma4_lora_trainer import Gemma4LoRAMultiTaskTrainerWithLogProb, GEMMA4_MODEL_ID
from Menta_lora_multitask_weighted_optimized import MultiTaskConfig, create_optimized_task_configs

tasks = create_optimized_task_configs()

config = MultiTaskConfig(
    model_name=GEMMA4_MODEL_ID,
    use_weighted_loss=True,
    output_dir="./gemma4_smoke_test_output",
    num_epochs=1,
    batch_size=1,
    max_length=128,
    use_8bit=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

lora_config = {
    "r": config.lora_r,
    "alpha": config.lora_alpha,
    "dropout": config.lora_dropout,
    "target_modules": config.target_modules
}

trainer = Gemma4LoRAMultiTaskTrainerWithLogProb(
    config, tasks, lora_config,
    alpha=5.0,
    beta=0.3
)
trainer.train()
print("Smoke test passed.")
```

```bash
python smoke_test_gemma4.py
```

**What to look for:**

| Sign | Meaning |
|------|---------|
| PEFT prints `trainable params` | LoRA attached to correct layers |
| Loss values appear without `nan` | Forward + backward pass working |
| No `CUDA out of memory` in first batch | Memory config is viable |
| `Smoke test passed.` at the end | Full epoch completed successfully |

**If you hit OOM**, reduce `max_length` to `64` or reduce `batch_size`. With 8-bit quantisation and `batch_size=1`, the model requires approximately 8–10 GB VRAM.

---

## Full Pre-training

Once both smoke tests pass, run the full Config-1 training with log-probability evaluation:

```bash
python Menta_gemma4_lora_config1_logprob.py
```

This runs 3 epochs over all six tasks using the LoRA Config-1 settings (r=8, alpha=16) and the BACC surrogate loss. Training time depends on dataset size and hardware; expect several hours on a single A100/H100.

### Configuration

All hyperparameters are in `config_gemma4.yaml`. Key settings:

```yaml
model:
  name: "google/gemma-4-E4B-it"
  max_length: 512
  use_8bit: true

training:
  batch_size: 4
  learning_rate: 5e-4
  num_epochs: 3

lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

logprob:
  alpha: 5.0   # sigmoid sharpness for BACC loss
  beta: 0.3    # weight of BACC loss vs cross-entropy
```

Task weights can be adjusted under `task_weights` — suicide risk tasks are weighted higher (1.5) than stress/depression tasks (1.0–1.2) by default.

### Outputs

| Path | Contents |
|------|----------|
| `gemma4_trained_model_config1/` | LoRA adapter weights (PEFT checkpoint) |
| `gemma4_config1_training_results.json` | Per-task evaluation metrics |
| `gemma4_outputs/` | Trainer checkpoints and logs |

---

## Evaluation

Evaluation runs automatically at the end of `Menta_gemma4_lora_config1_logprob.py` using the log-probability scoring method on the held-out 20% test split of each dataset. Results are written to `gemma4_config1_training_results.json`.

### Reading the results

```python
import json

with open("gemma4_config1_training_results.json") as f:
    results = json.load(f)

task_results = results["config_1_logprob_gemma4"]["task_results"]

for task_name, metrics in task_results.items():
    m = metrics["improved_logprob"]
    if "error" in m:
        print(f"{task_name}: ERROR — {m['error']}")
    else:
        print(f"{task_name}:")
        print(f"  Accuracy:          {m.get('accuracy', 'N/A'):.4f}")
        print(f"  Balanced Accuracy: {m.get('balanced_accuracy', 'N/A'):.4f}")
        print(f"  F1 Macro:          {m.get('f1_macro', 'N/A'):.4f}")
        print(f"  F1 Weighted:       {m.get('f1_weighted', 'N/A'):.4f}")
```

### Metrics explained

| Metric | Description |
|--------|-------------|
| `accuracy` | Standard classification accuracy |
| `balanced_accuracy` | Per-class accuracy averaged equally — preferred for imbalanced datasets |
| `f1_macro` | F1 averaged equally across all classes |
| `f1_weighted` | F1 weighted by class support |

### Running evaluation independently on a saved checkpoint

If you want to re-evaluate a saved checkpoint without retraining:

```python
import sys, os
sys.path.append(".")

from gemma4_lora_trainer import Gemma4LoRAMultiTaskTrainerWithLogProb, GEMMA4_MODEL_ID
from Menta_lora_multitask_weighted_optimized import MultiTaskConfig, create_optimized_task_configs
import pandas as pd
from sklearn.model_selection import train_test_split

tasks = create_optimized_task_configs()

config = MultiTaskConfig(
    model_name="./gemma4_trained_model_config1",  # path to saved checkpoint
    output_dir="./gemma4_eval_only",
    use_8bit=True,
)

lora_config = {
    "r": 8, "alpha": 16, "dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

trainer = Gemma4LoRAMultiTaskTrainerWithLogProb(config, tasks, lora_config)

for task in tasks:
    df = pd.read_csv(task.dataset_path)
    _, test_df = train_test_split(df, test_size=0.20, random_state=42)
    metrics = trainer.evaluate_task_with_logprob(task, test_df)
    print(task.name, metrics)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` to 1–2, or `max_length` to 256 |
| `401 Unauthorized` when loading model | Run `huggingface-cli login` and accept the Gemma licence |
| `Multi-token label detected` warning | Informational only — the evaluator uses the first token as fallback |
| `ModuleNotFoundError: Menta_lora_multitask_weighted_optimized` | Run scripts from inside `Menta_pretraining_code/` |
| NaN loss from the first step | Lower `learning_rate` in `config_gemma4.yaml` (try `1e-4`) |
