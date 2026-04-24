# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Menta is a privacy-preserving mental health monitoring system with two distinct components:

1. **`Menta_pretraining_code/`** — Python training pipeline for fine-tuning a Qwen3-4B-Instruct model on 6 mental health classification tasks using LoRA.
2. **`Menta_deployment/`** — iOS SwiftUI app that runs GGUF-quantized models on-device via llama.cpp for real-time inference benchmarking.

The trained model is exported as `Menta.gguf` and downloaded separately from [HuggingFace](https://huggingface.co/mHealthAI/Menta).

## Training (Python)

```bash
cd Menta_pretraining_code

# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p data cache/huggingface cache/nltk_data cache/wandb outputs

# Place CSVs in data/:
# - dreaddit_StressAnalysis - Sheet1.csv
# - Reddit_depression_dataset.csv
# - SDCNL.csv
# - 500_Reddit_users_posts_labels.csv

# Standard multi-task training
python Menta_lora_multitask_weighted_optimized.py

# Training with log-probability evaluation
python Menta_lora_config1_logprob.py

# See example_usage.py for programmatic API
```

Training requires CUDA 12.4+ and at least 16 GB GPU VRAM. Use `config.yaml` to adjust hyperparameters (LoRA rank/alpha, batch size, learning rate, task weights, logprob alpha/beta).

## iOS Deployment

```bash
cd Menta_deployment

# First-time: build the llama.cpp XCFramework (requires Xcode CLI tools)
cd llamacpp-framework && ./build-xcframework.sh && cd ..

# Download model files into Menta/ directory:
# - Menta.gguf (~2.3 GB)
# - Phi-4-mini-instruct-Q4_K_M.gguf (~2.3 GB)
# - qwen3-4b_Q4_K_M.gguf (~2.3 GB)

open Menta.xcodeproj   # Then Cmd+R to build and run
```

Requirements: Xcode 15+, iOS 16+, iPhone with A14 Bionic or later.

The `llamacpp-framework/` directory is a git submodule pointing to llama.cpp. Clone with `git clone --recursive` or run `git submodule update --init --recursive` after cloning.

## Architecture

### Training Pipeline (`Menta_pretraining_code/`)

- `Menta_lora_multitask_weighted_optimized.py` — main trainer; defines `MultiTaskConfig`, `TaskConfig`, and `Qwen3LoRAMultiTaskTrainer`. Implements BACC surrogate loss: `L = L_CE + β * L_BACC`.
- `improved_logprob_implementation.py` — mixin/wrapper that replaces standard classification with log-probability scoring (compares `log p("0"|prompt)` vs `log p("1"|prompt)`).
- `Menta_lora_config1_logprob.py` — entry point that combines both approaches using `create_improved_logprob_trainer()`.
- `config.yaml` — all hyperparameters; task weights are higher for suicide tasks (1.5) than stress/depression tasks (1.0–1.2).

### iOS App (`Menta_deployment/Menta/`)

The app is a benchmark harness, not a production screening tool. It loads GGUF models and evaluates them on bundled datasets.

- `LlamaState.swift` — `ObservableObject` driving the UI; owns the `LlamaContext` and orchestrates task evaluation, metrics collection (TTFT, ITPS, OTPS, OOM tracking).
- `Tasks.swift` — `TaskType` enum + `TaskConfig` structs defining all 6 tasks (dataset file paths, label/text columns, class names, prompt templates, class weights).
- `PromptGenerator.swift` — generates ChatML-formatted prompts (`<|im_start|>system...`) with randomized variants per task to improve generalization.
- `PredictionParser.swift` — multi-layer keyword + regex fallback parser that converts raw model text output to integer labels.
- `BatchProcessor.swift` — manages memory between batches; triggers cleanup to avoid OOM on device.
- `DatasetLoader.swift` — loads CSV datasets from the app bundle at runtime.
- `LibLlama.swift` — thin Swift wrapper around the llama.cpp C API.

### Task Definitions (both components share the same 6 tasks)

| Task | Type | Dataset | Classes |
|------|------|---------|---------|
| Stress Detection | Binary | Dreaddit | 0/1 |
| Depression Binary | Binary | Reddit Depression | 0/1 |
| Depression Severity | 4-class | Reddit Depression | 0–3 |
| Suicide Ideation | Binary | SDCNL | 0/1 |
| Suicide Risk Binary | Binary | 500 Reddit Users | 0/1 |
| Suicide Risk Severity | 5-class | 500 Reddit Users | 1–5 |

## Key Conventions

- The base model is `Qwen/Qwen3-4B-Instruct-2507`; prompts use Qwen3's ChatML format (`<|im_start|>` / `<|im_end|>`).
- Phi-4-mini uses a different prompt format — `PromptGenerator.swift` branches on model name.
- Model files (`.gguf`) are never committed; they are downloaded separately and placed in `Menta_deployment/Menta/`.
- Training outputs go to `outputs/` and `qwen3_trained_model/` (both gitignored).
- WandB runs in offline mode by default (`wandb_mode: "offline"` in config).
