import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
import os

model_id = "google/gemma-4-E4B-it"
print(f"Checking language model structure for {model_id}...")

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Language model layers are usually in model.layers (for Gemma)
    # or model.model.layers
    
    found = False
    for name, module in model.named_modules():
        if "model.layers.0" in name and any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            print(f"Module: {name} | Type: {type(module)}")
            if hasattr(module, 'linear'):
                print(f"  -> Found .linear attribute: {type(module.linear)}")
            found = True
    
    if not found:
        print("Language layers (model.layers.0) not found. Listing top-level modules:")
        for name, _ in model.named_children():
            print(f"  Child: {name}")

except Exception as e:
    print(f"Error: {e}")
