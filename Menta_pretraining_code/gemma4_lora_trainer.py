#!/usr/bin/env python3
"""
Gemma 4B LoRA trainer — extends the Qwen3 base trainer with Gemma-specific tokenizer setup.

The only behavioral difference vs. Qwen3: Gemma's tokenizer defaults to left-padding,
which breaks fine-tuning. This subclass forces padding_side='right' after tokenizer load.
Everything else (LoRA target modules, loss, log-prob evaluation) is inherited unchanged.

Model ID: google/gemma-4-e4b-it
Verify the exact casing at huggingface.co/google before first use.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Menta_lora_multitask_weighted_optimized import Qwen3LoRAMultiTaskTrainer
from improved_logprob_implementation import create_improved_logprob_trainer

GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"


class Gemma4LoRAMultiTaskTrainer(Qwen3LoRAMultiTaskTrainer):
    """LoRA multi-task trainer configured for Gemma 4B instruct."""

    def _setup_model_and_tokenizer(self):
        super()._setup_model_and_tokenizer()
        # Gemma tokenizer defaults to left-padding; right-padding is required for causal LM training.
        self.tokenizer.padding_side = "right"


Gemma4LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Gemma4LoRAMultiTaskTrainer)
