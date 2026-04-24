#!/usr/bin/env python3
"""
Gemma 4B LoRA Multi-Task Fine-tuning - Config 1 with Log-Prob Evaluation
Config: r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

Mirrors Menta_lora_config1_logprob.py but targets Gemma 4B instead of Qwen3-4B.
All task definitions, datasets, label mappings, and evaluation logic are unchanged.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemma4_lora_trainer import Gemma4LoRAMultiTaskTrainerWithLogProb, GEMMA4_MODEL_ID
from Menta_lora_multitask_weighted_optimized import MultiTaskConfig, create_optimized_task_configs
from improved_logprob_implementation import ImprovedLogProbEvaluator
import logging
import gc
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function - Gemma 4B Config 1 with Log-Prob"""
    lora_config = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }

    tasks = create_optimized_task_configs()

    config = MultiTaskConfig(
        model_name=GEMMA4_MODEL_ID,
        use_weighted_loss=True,
        output_dir="./gemma4_trained_model_config1",
        lora_r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"]
    )

    logger.info("Config 1: Basic configuration + Log-Prob evaluation (Gemma 4B)")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output directory: {config.output_dir}")

    trainer = Gemma4LoRAMultiTaskTrainerWithLogProb(
        config, tasks, lora_config,
        alpha=5.0,  # sigmoid sharpness
        beta=0.3    # CE vs BACC trade-off
    )
    trainer.train()

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Starting improved Log-Prob evaluation...")
    task_results = {}

    for task in tasks:
        df = pd.read_csv(task.dataset_path)

        # Apply label mappings (identical to Qwen3 entry point)
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
        _, test_df = train_test_split(
            temp_df, test_size=0.714, random_state=42, stratify=temp_df[task.label_column]
        )

        try:
            metrics_dict = trainer.evaluate_task_with_logprob(task, test_df)
            task_results[task.name] = {"improved_logprob": metrics_dict}
            logger.info(f"Task {task.name} completed successfully")
        except Exception as e:
            logger.error(f"Error evaluating task {task.name}: {e}")
            task_results[task.name] = {"improved_logprob": {"error": str(e)}}

        gc.collect()
        torch.cuda.empty_cache()

    results = {
        "config_1_logprob_gemma4": {
            "model": GEMMA4_MODEL_ID,
            "lora_config": lora_config,
            "improved_logprob_params": {
                "alpha": 5.0,
                "beta": 0.3
            },
            "task_results": task_results
        }
    }

    output_path = "./gemma4_config1_training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Config 1 Log-Prob training and evaluation completed (Gemma 4B)")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
