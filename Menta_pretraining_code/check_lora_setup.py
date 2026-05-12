import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os

model_id = "google/gemma-4-E4B-it"

def check_lora_setup():
    print(f"Checking LoRA setup for {model_id}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    # Load model in 8-bit
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded in 8-bit.")
        
        # Prepare for kbit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("Model prepared for kbit training.")
        
        # Original target modules
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Patched target modules as in the code
        patched_target_modules = [f"{m}.linear" for m in target_modules]
        print(f"Target modules: {patched_target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=patched_target_modules,
            bias="none"
        )
        
        lora_model = get_peft_model(model, peft_config)
        lora_model.print_trainable_parameters()
        
        # Check if any modules actually have lora
        has_lora = False
        for name, module in lora_model.named_modules():
            if "lora_" in name:
                print(f"Found LoRA module: {name}")
                has_lora = True
                break
        
        if not has_lora:
            print("❌ NO LORA MODULES FOUND!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_lora_setup()
