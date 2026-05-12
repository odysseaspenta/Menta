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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        if self.config.use_8bit:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=False
            )
            self.model.config.use_cache = False
            self.config.gradient_checkpointing = False

        # Identify all linear modules that are NOT the wrapper
        logger.info("🔍 Surgical targeting of Linear layers...")
        linear_layers = []
        for name, module in self.model.named_modules():
            m_type = str(type(module))
            if ("Linear" in m_type or "Linear8bitLt" in m_type) and "Gemma4ClippableLinear" not in m_type:
                # Check if it matches our target projections
                if any(name.endswith(f".{t}") or name.endswith(f".{t}.linear") for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                    linear_layers.append(name)
        
        if not linear_layers:
            raise RuntimeError("Could not find any supported Linear layers to target with LoRA!")

        logger.info(f"🎯 Found {len(linear_layers)} supported Linear layers.")
        
        patched = dict(self.lora_config)
        # Using the explicit list of full module names is the safest way to avoid partial matches
        patched["target_modules"] = linear_layers
        
        original = self.lora_config
        self.lora_config = patched
        
        try:
            super()._setup_lora()
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"📊 Trainable parameters: {trainable_params} ({100 * trainable_params / all_params:.4f}%)")
        finally:
            self.lora_config = original


Gemma4LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Gemma4LoRAMultiTaskTrainer)
