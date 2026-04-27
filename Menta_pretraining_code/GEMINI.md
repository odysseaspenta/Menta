# Menta Pretraining Project Overview

This project is a multi-task fine-tuning framework for mental health classification using Large Language Models (LLMs). It primarily targets **Qwen3-4B-Instruct-2507** and **Gemma-4-E4B-it** (Gemma 4B) models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

The framework supports 6 mental health classification tasks across 4 datasets:
1. **Stress Detection**: (Dreaddit) Binary classification of stress in Reddit posts.
2. **Depression Binary**: (Reddit Depression) Detection of depression signs.
3. **Depression Severity**: (Reddit Depression) 4-class severity level classification.
4. **Suicide Ideation**: (SDCNL) Binary classification of suicidal ideation.
5. **Suicide Risk Binary**: (500 Reddit Users) User-level suicide risk detection.
6. **Suicide Risk Severity**: (500 Reddit Users) 5-class risk level classification.

## Core Technologies
- **Models**: Qwen3-4B-Instruct-2507, Gemma-4-E4B-it.
- **PEFT/LoRA**: Efficient fine-tuning of model subsets.
- **Log-Probability Evaluation**: A novel evaluation method comparing log probabilities of label tokens.
- **Quantization**: 8-bit quantization via `bitsandbytes` for memory efficiency.
- **Multi-task Learning**: Single model handles multiple classification objectives with weighted loss and BACC (Balanced Accuracy) surrogate loss.

## Project Structure
- `Menta_lora_multitask_weighted_optimized.py`: Core multi-task training logic.
- `improved_logprob_implementation.py`: Implementation of log-prob scoring and BACC surrogate loss.
- `gemma4_lora_trainer.py`: Gemma-specific trainer subclass with tokenizer and layer patching.
- `Menta_lora_config1_logprob.py`: Entry point for Qwen3 training (Config 1).
- `Menta_gemma4_lora_config1_logprob.py`: Entry point for Gemma 4B training (Config 1).
- `config.yaml` / `config_gemma4.yaml`: Configuration files for training hyperparameters.
- `dataset/`: Contains CSV files for the 4 target datasets.

## Building and Running

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data cache/huggingface cache/nltk_data cache/wandb outputs
```

### Training Commands
- **Qwen3 Training (Config 1)**:
  ```bash
  python Menta_lora_config1_logprob.py
  ```
- **Gemma 4B Training (Config 1)**:
  ```bash
  python Menta_gemma4_lora_config1_logprob.py
  ```
- **Custom Training (using setup.py entry points)**:
  ```bash
  qwen3-train --config config.yaml
  ```

### Data Setup
Ensure the following files are in the `dataset/` or `data/` directory (depending on config):
- `dreaddit_StressAnalysis - Sheet1.csv`
- `Reddit_depression_dataset.csv`
- `SDCNL.csv`
- `500_Reddit_users_posts_labels.csv`

## Development Conventions
- **Model Loading**: Models are loaded with 8-bit quantization by default to fit in ~16GB VRAM.
- **Tokenization**: Gemma requires `padding_side='right'` for training.
- **Evaluation**: The project uses both standard classification metrics (Accuracy, F1, BACC) and specialized Log-Probability scoring.
- **LoRA Targets**: Default targets are `["q_proj", "k_proj", "v_proj", "o_proj"]`. For Gemma, these are patched to `.linear` children.
- **Loss Function**: Uses a combination of Cross-Entropy and BACC surrogate loss controlled by `beta`.

## Key Parameters
- `alpha`: Sigmoid sharpness for BACC surrogate (Default: 5.0).
- `beta`: CE vs BACC trade-off weight (Default: 0.3).
- `use_weighted_loss`: Enables task-specific weights defined in config.
