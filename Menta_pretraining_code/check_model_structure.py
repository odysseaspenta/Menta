import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
import os

model_id = "google/gemma-4-E4B-it"
print(f"Checking model structure for {model_id} using init_empty_weights...")

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("Model Config loaded.")
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    count = 0
    for name, module in model.named_modules():
        # Check for projection layers
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            # Only print the first few layers and their structure
            if "layers.0" in name or "layers.1" in name:
                print(f"Module: {name} | Type: {type(module)}")
                if hasattr(module, 'linear'):
                    print(f"  -> Found .linear attribute: {type(module.linear)}")
                count += 1
        
        if count > 20:
            break

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
