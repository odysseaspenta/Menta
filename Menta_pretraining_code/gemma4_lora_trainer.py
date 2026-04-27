#!/usr/bin/env python3
"""
Gemma 4B LoRA trainer — extends the Qwen3 base trainer with Gemma-specific fixes.

Two differences from the Qwen3 trainer:
1. Tokenizer padding_side is forced to 'right' (Gemma defaults to left-padding).
2. LoRA target modules are rewritten to target the inner .linear child of each
   Gemma4ClippableLinear wrapper (e.g. "q_proj" → "q_proj.linear"), because PEFT
   does not support custom wrapper classes directly.

Model ID: google/gemma-4-E4B-it
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

    def _setup_lora(self):
        # With 8-bit quantisation, frozen weights produce outputs with no grad_fn.
        # prepare_model_for_kbit_training hooks the input embeddings so downstream
        # tensors carry requires_grad=True, restoring the gradient graph to the LoRA
        # matrices. It also casts layernorms to fp32 for numerical stability.
        if self.config.use_8bit:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        # Gemma 4 wraps every projection layer in Gemma4ClippableLinear(...(linear): ...).
        # PEFT only handles standard module types and raises ValueError on the wrapper.
        # We redirect each target name to its inner .linear child, which is a supported
        # Linear8bitLt (or torch.nn.Linear when not quantised).
        patched = dict(self.lora_config)
        patched["target_modules"] = [
            f"{m}.linear" for m in self.lora_config["target_modules"]
        ]
        original = self.lora_config
        self.lora_config = patched
        try:
            super()._setup_lora()
        finally:
            self.lora_config = original


Gemma4LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Gemma4LoRAMultiTaskTrainer)
