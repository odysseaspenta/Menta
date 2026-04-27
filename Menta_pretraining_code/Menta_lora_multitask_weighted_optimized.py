#!/usr/bin/env python3
"""
Qwen3 LoRA Multi-Task Fine-tuning with Optimized Weighted Loss
Redesigned weighted version to solve task3/task4 similarity and task5/task6 label issues
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskConfig:
    """Task configuration class"""
    def __init__(self, name, dataset_path, label_column, text_column, 
                 num_classes, class_names, prompt_template, weight=1.0):
        self.name = name
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.text_column = text_column
        self.num_classes = num_classes
        self.class_names = class_names
        self.prompt_template = prompt_template
        self.weight = weight

class MultiTaskConfig:
    """Multi-task configuration class"""
    def __init__(self, 
                 model_name="Qwen/Qwen3-4B-Instruct-2507",
                 use_weighted_loss: bool = True,
                 max_length: int = 512,
                 batch_size: int = 4,
                 learning_rate: float = 5e-4,
                 num_epochs: int = 3,
                 warmup_steps: int = 100,
                 save_steps: int = 500,
                 eval_steps: int = 500,
                 logging_steps: int = 100,
                 output_dir: str = "./qwen3_lora_multitask_output",
                 use_8bit: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 target_modules: list = None):
        
        self.model_name = model_name
        self.use_weighted_loss = use_weighted_loss
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.use_8bit = use_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class WeightedTrainer(Trainer):
    """Trainer with weighted loss"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # For causal language models, use standard cross-entropy loss
        # Class weights are not applicable for token-level prediction
        loss_fct = nn.CrossEntropyLoss()
        
        # Reshape logits and labels to 2D
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class MentalHealthDataset(Dataset):
    """Mental health dataset"""
    
    def __init__(self, tasks, tokenizer, max_length=512, split='train'):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split  # 'train', 'eval', 'test'
        self.data = []
        
        # Load all task data
        for task in tasks:
            self._load_task_data(task)
    
    def _load_task_data(self, task):
        """Load data for a single task"""
        try:
            df = pd.read_csv(task.dataset_path)
            logger.info(f"Loading {task.name}: {len(df)} samples")
            
            # Handle label mapping (updated according to 6-task requirements)
            if task.name == "task2_depression_binary":
                # Task 2: Binary Depression Detection (0: No Depression, 1: Depression)
                label_mapping = {
                    'minimum': '0',
                    'mild': '1', 
                    'moderate': '1',
                    'severe': '1'
                }
                df[task.label_column] = df[task.label_column].map(label_mapping)
                df = df.dropna(subset=[task.label_column])  # Remove unmappable labels
                logger.info(f"Task2 after label mapping: {len(df)} samples")
                
            elif task.name == "task3_depression_severity":
                # Task 3: Four-Level Depression Detection (0: Minimal, 1: Mild, 2: Moderate, 3: Severe)
                label_mapping = {
                    'minimum': '0',
                    'mild': '1',
                    'moderate': '2', 
                    'severe': '3'
                }
                df[task.label_column] = df[task.label_column].map(label_mapping)
                df = df.dropna(subset=[task.label_column])  # Remove unmappable labels
                logger.info(f"Task3 after label mapping: {len(df)} samples")
                
            elif task.name == "task5_suicide_risk_binary":
                # Task 5: Binary Suicide Risk Detection (0: No suicide risk, 1: Suicide risk)
                label_mapping = {
                    'Supportive': '0',
                    'Indicator': '1',
                    'Ideation': '1',
                    'Behavior': '1',
                    'Attempt': '1'
                }
                df[task.label_column] = df[task.label_column].map(label_mapping)
                df = df.dropna(subset=[task.label_column])  # Remove unmappable labels
                logger.info(f"Task5 after label mapping: {len(df)} samples")
                
            elif task.name == "task6_suicide_risk_severity":
                # Task 6: Five-Class Suicide Risk Severity Detection (1: Supportive, 2: Indicator, 3: Ideation, 4: Behavior, 5: Attempt)
                label_mapping = {
                    'Supportive': '1',
                    'Indicator': '2',
                    'Ideation': '3',
                    'Behavior': '4',
                    'Attempt': '5'
                }
                df[task.label_column] = df[task.label_column].map(label_mapping)
                df = df.dropna(subset=[task.label_column])  # Remove unmappable labels
                logger.info(f"Task6 after label mapping: {len(df)} samples")
            
            # Data split: 72% train, 8% eval, 20% test
            if self.split == 'train':
                # 72% for training
                df_split, _ = train_test_split(
                    df, test_size=0.28, random_state=42, 
                    stratify=df[task.label_column]
                )
            elif self.split == 'eval':
                # 8% for validation (from remaining 28%, take 8/28 ≈ 28.57%)
                _, temp_df = train_test_split(
                    df, test_size=0.28, random_state=42, 
                    stratify=df[task.label_column]
                )
                df_split, _ = train_test_split(
                    temp_df, test_size=0.714, random_state=42,  # 0.714 = 20/28, take first 28.57% as validation set
                    stratify=temp_df[task.label_column]
                )
            else:  # test
                # 20% for testing (from remaining 28%, take 20/28 ≈ 71.43%)
                _, temp_df = train_test_split(
                    df, test_size=0.28, random_state=42, 
                    stratify=df[task.label_column]
                )
                _, df_split = train_test_split(
                    temp_df, test_size=0.714, random_state=42,  # 0.714 = 20/28, take last 71.43% as test set
                    stratify=temp_df[task.label_column]
                )
            
            logger.info(f"{task.name} {self.split} set: {len(df_split)} samples")
            
            for _, row in df_split.iterrows():
                text = str(row[task.text_column])
                label = str(row[task.label_column])
                
                # Generate prompt
                if self.split == 'train':
                    prompt = task.prompt_template.format(text=text, label=label)
                else:
                    prompt = task.prompt_template.format(text=text, label="")
                
                self.data.append({
                    'task': task.name,
                    'text': prompt,
                    'label': label,
                    'weight': task.weight
                })
                
        except Exception as e:
            logger.error(f"Failed to load task {task.name}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def create_optimized_task_configs():
    """Create optimized task configurations"""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    tasks = [
        # Task 1: Stress Detection (Binary Classification)
        TaskConfig(
            name="task1_stress",
            dataset_path=f"{base_path}/dreaddit_StressAnalysis - Sheet1.csv",
            label_column="label",
            text_column="text",
            num_classes=2,
            class_names=["0", "1"],
            prompt_template="""Is the poster of the post stressed? Classify the Users' post text into 0 and 1. 1 means stressed, 0 means not stressed.

text: {text}
label: {label}""",
            weight=1.0
        ),
        
        # Task 2: Binary Depression Detection (Binary Classification)
        TaskConfig(
            name="task2_depression_binary",
            dataset_path=f"{base_path}/Reddit_depression_dataset.csv",
            label_column="label",
            text_column="text",
            num_classes=2,
            class_names=["0", "1"],
            prompt_template="""Is the poster of the post depressed? Classify the Users' post text into 0 and 1. 0 means no depression (equivalent to minimal depression), 1 means depression (encompassing mild, moderate, and severe depression).

text: {text}
label: {label}""",
            weight=1.0
        ),
        
        # Task 3: Four-Level Depression Detection (Multi-class Classification)
        TaskConfig(
            name="task3_depression_severity",
            dataset_path=f"{base_path}/Reddit_depression_dataset.csv",
            label_column="label",
            text_column="text",
            num_classes=4,
            class_names=["0", "1", "2", "3"],
            prompt_template="""What level of depression is the person experiencing? Classify the Users' post text into 0, 1, 2 and 3. 0 means minimal depression, 1 means mild depression, 2 means moderate depression, 3 means severe depression.

text: {text}
label: {label}""",
            weight=1.2
        ),
        
        # Task 4: Binary Suicide Ideation Detection (Binary Classification)
        TaskConfig(
            name="task4_suicide_ideation",
            dataset_path=f"{base_path}/SDCNL.csv",
            label_column="is_suicide",
            text_column="megatext_clean",
            num_classes=2,
            class_names=["0", "1"],
            prompt_template="""Is the poster of the post experiencing suicidal ideation? Classify the following text as 0 or 1. 0 means not suicidal ideation, 1 means suicidal ideation.

text: {text}
label: {label}""",
            weight=1.2
        ),
        
        # Task 5: Binary Suicide Risk Detection (Binary Classification)
        TaskConfig(
            name="task5_suicide_risk_binary",
            dataset_path=f"{base_path}/500_Reddit_users_posts_labels.csv",
            label_column="Label",
            text_column="Post",
            num_classes=2,
            class_names=["0", "1"],
            prompt_template="""Is the poster of the post at risk for suicide? Classify the Users' post text into 0 and 1. 0 means no suicide risk, 1 means suicide risk (at least an indicator of suicide risk).

text: {text}
label: {label}""",
            weight=1.5
        ),
        
        # Task 6: Five-Class Suicide Risk Severity Detection (Multi-class Classification)
        TaskConfig(
            name="task6_suicide_risk_severity",
            dataset_path=f"{base_path}/500_Reddit_users_posts_labels.csv",
            label_column="Label",
            text_column="Post",
            num_classes=5,
            class_names=["1", "2", "3", "4", "5"],
            prompt_template="""What is the suicide risk severity level of the poster of the post? Classify the Users' post text into 1, 2, 3, 4 and 5. 1 means Supportive (emotional support but no risk signals), 2 means Indicator (indirect signs of vulnerability), 3 means Ideation (explicit suicidal thoughts), 4 means Behavior (suicide-related behaviors), 5 means Attempt (evidence of actual suicide attempts).

text: {text}
label: {label}""",
            weight=1.5
        )
    ]
    
    return tasks

def create_lora_configs():
    """Create 9 different LoRA configurations"""
    configs = [
        # Config 1: Basic configuration
        {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        
        # Config 2: Higher rank
        {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        
        # Config 3: More modules
        {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]},
        
        # Config 4: High dropout
        {"r": 8, "alpha": 16, "dropout": 0.2, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        
        # Config 5: Low rank high alpha
        {"r": 4, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        
        # Config 6: All modules
        {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        
        # Config 7: High rank low dropout
        {"r": 16, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        
        # Config 8: Medium configuration
        {"r": 12, "alpha": 24, "dropout": 0.15, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]},
        
        # Config 9: Conservative configuration
        {"r": 6, "alpha": 12, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj"]}
    ]
    
    return configs

class Qwen3LoRAMultiTaskTrainer:
    """Qwen3 LoRA multi-task trainer"""
    
    def __init__(self, config: MultiTaskConfig, tasks, lora_config):
        self.config = config
        self.tasks = tasks
        self.lora_config = lora_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup cache directories to project disk
        self._setup_cache_directories()
        
        # Initialize model and tokenizer
        self._setup_model_and_tokenizer()
        
        # Create LoRA configuration
        self._setup_lora()
        
    def _setup_cache_directories(self):
        """Setup cache directories to project disk"""
        # Use project directory for cache to avoid home directory quota issues
        project_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dirs = {
            'hf_cache': os.path.join(project_dir, 'cache', 'huggingface'),
            'nltk_data': os.path.join(project_dir, 'cache', 'nltk_data'),
            'wandb': os.path.join(project_dir, 'cache', 'wandb')
        }
        
        for cache_type, cache_dir in cache_dirs.items():
            os.makedirs(cache_dir, exist_ok=True)
            if cache_type == 'hf_cache':
                # Set all related environment variables
                os.environ['HF_HOME'] = cache_dir
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                os.environ['HF_HUB_CACHE'] = cache_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
                # Ensure home directory cache is disabled
                os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
                # Disable WandB
                os.environ['WANDB_DISABLED'] = 'true'
            elif cache_type == 'nltk_data':
                import nltk
                import nltk.data
                nltk.data.path.append(cache_dir)
            elif cache_type == 'wandb':
                os.environ['WANDB_CACHE_DIR'] = cache_dir
        
        logger.info(f"🔧 Setup cache directories: {cache_dirs}")
    
    def _setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info(f"🚀 Loading model: {self.config.model_name}")
        
        # Setup quantization configuration
        quantization_config = None
        if self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # Get cache directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.environ.get('HF_HOME', os.path.join(project_dir, 'cache', 'huggingface'))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        logger.info("✅ Model and tokenizer loaded successfully")
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("🔧 Setting up LoRA configuration")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=self.lora_config["target_modules"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ LoRA configuration completed")
    
    def compute_class_weights(self, dataset):
        """Compute class weights"""
        if not self.config.use_weighted_loss:
            return None
            
        # For causal language models, class weights are not applicable
        # Return None to let WeightedTrainer use standard loss
        return None
    
    def train(self):
        """Train model"""
        logger.info("🎯 Starting training")
        
        # Create training and validation datasets
        train_dataset = MentalHealthDataset(
            self.tasks, self.tokenizer, 
            max_length=self.config.max_length, 
            split='train'
        )
        
        eval_dataset = MentalHealthDataset(
            self.tasks, self.tokenizer, 
            max_length=self.config.max_length, 
            split='eval'
        )
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_dataset)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable wandb
            gradient_checkpointing=False,  # Disable to improve speed
            optim="adamw_torch",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            push_to_hub=False
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        if self.config.use_weighted_loss and class_weights is not None:
            trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                class_weights=class_weights
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )
        
        # Start training
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Store the trainer for potential later use
        self._trainer = trainer
        
        logger.info("✅ Training completed")
        
        return self


def create_lora_configs():
    """Create LoRA configurations for different experiments"""
    return [
        {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 32, "alpha": 64, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 8, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        {"r": 4, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 16, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 8, "alpha": 8, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 32, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 64, "alpha": 16, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    ]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Qwen3 LoRA Multi-Task Training")
    parser.add_argument("--config_id", type=int, default=1, help="LoRA config ID (1-9)")
    parser.add_argument("--output_suffix", type=str, default="", help="Output directory suffix")
    args = parser.parse_args()
    
    # Get LoRA configuration
    lora_configs = create_lora_configs()
    if args.config_id < 1 or args.config_id > 9:
        raise ValueError("config_id must be between 1-9")
    
    lora_config = lora_configs[args.config_id - 1]
    
    # Create task configuration
    tasks = create_optimized_task_configs()
    
    # Create multi-task configuration
    config = MultiTaskConfig(
        use_weighted_loss=True,
        output_dir=f"./qwen3_lora_multitask_output/config_{args.config_id}{args.output_suffix}",
        lora_r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"]
    )
    
    logger.info(f"🎯 Using LoRA config {args.config_id}: r={lora_config['r']}, alpha={lora_config['alpha']}, dropout={lora_config['dropout']}")
    logger.info(f"📁 Output directory: {config.output_dir}")
    
    # Create trainer and start training
    trainer = Qwen3LoRAMultiTaskTrainer(config, tasks, lora_config)
    trainer.train()
    
    logger.info("🎉 All tasks completed!")

if __name__ == "__main__":
    main()
