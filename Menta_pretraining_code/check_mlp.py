import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
import os

model_id = "google/gemma-4-E4B-it"

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    for name, module in model.named_modules():
        if "model.language_model.layers.0.mlp" in name:
            print(f"Module: {name} | Type: {type(module)}")

except Exception as e:
    print(f"Error: {e}")
