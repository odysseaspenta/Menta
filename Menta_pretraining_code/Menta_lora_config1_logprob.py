#!/usr/bin/env python3
"""
Qwen3 LoRA Multi-Task Fine-tuning - Config 1 with Log-Prob Evaluation
Config: r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
Basic configuration - includes Log-Prob scoring method for comparative experiments
"""

import sys
import os
# Add current directory and parent directory to path
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen3_lora_multitask_weighted_optimized import (
    Qwen3LoRAMultiTaskTrainer, MultiTaskConfig, create_optimized_task_configs
)
from improved_logprob_implementation import ImprovedLogProbEvaluator, create_improved_logprob_trainer
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use improved Log-Prob trainer
Qwen3LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Qwen3LoRAMultiTaskTrainer)
    
# Old evaluation methods have been replaced by improved trainer

def main():
    """Main function - Config 1 with Log-Prob"""
    # LoRA configuration
    lora_config = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # Create task configuration
    tasks = create_optimized_task_configs()
    
    # Create multi-task configuration
    config = MultiTaskConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        use_weighted_loss=True,
        output_dir="./qwen3_trained_model_config1",
        lora_r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"]
    )
    
    logger.info("🎯 Config 1: Basic configuration + Log-Prob evaluation")
    logger.info(f"📁 Output directory: {config.output_dir}")
    
    # Create improved Log-Prob trainer and start training
    trainer = Qwen3LoRAMultiTaskTrainerWithLogProb(
        config, tasks, lora_config, 
        alpha=5.0,  # sigmoid sharpness
        beta=0.3    # CE vs BACC trade-off
    )
    trainer.train()

    # Save model before evaluation
    trainer.save_model()

    # Clear GPU memory after training
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # Use improved Log-Prob method to evaluate all tasks
    logger.info("🔍 Starting improved Log-Prob evaluation...")
    task_results = {}
    
    for task in tasks:
        # Load test data
        df = pd.read_csv(task.dataset_path)
        
        # Apply label mapping
        if task.name == "task2_depression_binary":
            mapping = {'minimum': '0', 'mild': '1', 'moderate': '1', 'severe': '1'}
            df[task.label_column] = df[task.label_column].map(mapping)
            df = df.dropna(subset=[task.label_column])
        elif task.name == "task3_depression_severity":
            mapping = {'minimum': '0', 'mild': '1', 'moderate': '2', 'severe': '3'}
            df[task.label_column] = df[task.label_column].map(mapping)
            df = df.dropna(subset=[task.label_column])
        elif task.name == "task5_suicide_risk_binary":
            mapping = {'Supportive': '0', 'Indicator': '1', 'Ideation': '1', 'Behavior': '1', 'Attempt': '1'}
            df[task.label_column] = df[task.label_column].map(mapping)
            df = df.dropna(subset=[task.label_column])
        elif task.name == "task6_suicide_risk_severity":
            mapping = {'Supportive': '1', 'Indicator': '2', 'Ideation': '3', 'Behavior': '4', 'Attempt': '5'}
            df[task.label_column] = df[task.label_column].map(mapping)
            df = df.dropna(subset=[task.label_column])
        
        # Data split (72% train, 8% eval, 20% test)
        train_df, temp_df = train_test_split(
            df, test_size=0.28, random_state=42, stratify=df[task.label_column]
        )
        eval_df, test_df = train_test_split(
            temp_df, test_size=0.714, random_state=42, stratify=temp_df[task.label_column]
        )
        
        # Improved Log-Prob evaluation
        try:
            metrics_dict = trainer.evaluate_task_with_logprob(task, test_df)
            task_results[task.name] = {"improved_logprob": metrics_dict}
            logger.info(f"✅ Task {task.name} completed successfully")
        except Exception as e:
            logger.error(f"❌ Error evaluating task {task.name}: {e}")
            task_results[task.name] = {"improved_logprob": {"error": str(e)}}
        
        # Clear memory after each task
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    import json
    results = {
        "config_1_logprob": {
            "lora_config": lora_config,
            "improved_logprob_params": {
                "alpha": 5.0,
                "beta": 0.3
            },
            "task_results": task_results
        }
    }
    
    with open("./qwen3_config1_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("🎉 Config 1 Log-Prob training and evaluation completed!")
    logger.info("📊 Results saved to: qwen3_config1_training_results.json")

if __name__ == "__main__":
    main()
