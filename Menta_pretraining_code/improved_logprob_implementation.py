#!/usr/bin/env python3
"""
Improved Log-Prob implementation, strictly following document formulas
Includes standard Log-Prob scoring and BACC surrogate loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm.auto import tqdm
from transformers import Trainer

logger = logging.getLogger(__name__)

class ImprovedLogProbEvaluator:
    """Improved Log-Prob evaluator, implementing standard formulas"""
    
    def __init__(self, model, tokenizer, alpha: float = 5.0, beta: float = 0.3):
        """
        Initialize Log-Prob evaluator
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            alpha: sigmoid sharpness parameter (1-20)
            beta: CE vs BACC trade-off weight (0.1-0.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha  # sigmoid sharpness
        self.beta = beta    # CE vs BACC trade-off
        
    @torch.no_grad()
    def predict_logits(self, prompts: List[str], label_tokens: Tuple[str, ...] = ("0", "1"),
                      max_len: int = 1024, desc: str = "Scoring") -> List[int]:
        """
        Standard Log-Prob scoring method
        Calculate log p(0|prompt) vs log p(1|prompt) and choose the higher one
        """
        label_ids = []
        for token in label_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                label_ids.append(ids[0])
            else:
                label_ids.append(ids[0])
                logger.warning(f"Multi-token label '{token}' detected, using first token")

        preds = []
        for prompt in tqdm(prompts, desc=desc, unit="sample"):
            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            out = self.model(**inp, use_cache=False)
            logits = out.logits[:, -1, :]  # [1, vocab_size]

            log_probs = []
            for label_id in label_ids:
                lp = torch.log_softmax(logits, dim=-1)[0, label_id].item()
                log_probs.append(lp)

            pred = np.argmax(log_probs)
            preds.append(pred)
        return preds

    def evaluate_task_with_improved_logprob(self, task_config, test_data, 
                                          label_tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate task using improved Log-Prob method
        """
        if label_tokens is None:
            label_tokens = [str(x) for x in task_config.class_names]
        
        n_samples = len(test_data)
        logger.info(f"Evaluating '{task_config.name}' — {n_samples} samples, labels: {label_tokens}")

        # Prepare data
        prompts = []
        true_labels = []

        for _, row in test_data.iterrows():
            text = str(row[task_config.text_column])
            label = str(row[task_config.label_column])

            # Format prompt
            prompt = task_config.prompt_template.format(text=text, label="")
            prompts.append(prompt)
            true_labels.append(label)

        # Use improved Log-Prob prediction
        predictions = self.predict_logits(
            prompts, tuple(label_tokens), desc=task_config.name
        )
        
        # Convert to numeric labels
        label_map = {token: i for i, token in enumerate(label_tokens)}
        y_true = [label_map.get(str(t), 0) for t in true_labels]
        y_pred = predictions
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        
        logger.info(f"Task {task_config.name} improved Log-Prob results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compute_bacc_surrogate_loss(self, logits: torch.Tensor, true_labels: torch.Tensor, 
                                  task_classes: int, gamma_c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute BACC surrogate loss
        Args:
            logits: classification logits [batch_size, num_classes]
            true_labels: true labels [batch_size]
        """
        batch_size = logits.size(0)
        if gamma_c is None:
            gamma_c = torch.ones(task_classes, device=logits.device)
        
        # Step 1: Calculate margin for each class
        margins = torch.zeros(batch_size, task_classes, device=logits.device)
        for c in range(task_classes):
            z_c = logits[:, c]
            other_logits = torch.cat([logits[:, :c], logits[:, c+1:]], dim=1)
            log_sum_exp_others = torch.logsumexp(other_logits, dim=1)
            margins[:, c] = z_c - log_sum_exp_others
        
        # Step 2: Calculate soft "is correct" score
        sigmoid_scores = torch.sigmoid(self.alpha * margins)
        
        # Step 3: Calculate TPR for each class
        tpr_c = torch.zeros(task_classes, device=logits.device)
        for c in range(task_classes):
            mask_c = (true_labels == c)
            if mask_c.sum() > 0:
                tpr_c[c] = sigmoid_scores[mask_c, c].mean()
            else:
                tpr_c[c] = 0.0
        
        # Step 4: Calculate BACC surrogate loss
        gamma_sum = gamma_c.sum()
        if gamma_sum > 0:
            weighted_tpr = (gamma_c * tpr_c).sum()
            bacc_loss = 1.0 - weighted_tpr / gamma_sum
        else:
            bacc_loss = torch.tensor(1.0, device=logits.device)
        return bacc_loss

    def compute_combined_loss(self, logits: torch.Tensor, true_labels: torch.Tensor, 
                            task_classes: int, label_ids: List[int]) -> torch.Tensor:
        """
        Args:
            logits: classification logits [batch_size, vocab_size]
            true_labels: true labels [batch_size] (token IDs)
            task_classes: number of task classes (e.g. 2 for binary)
            label_ids: the actual token IDs for the classes (e.g. IDs of "0" and "1")
        """
        # Map true token IDs to class indices (0, 1, ...)
        # This is tricky because true_labels are token IDs.
        # Let's create a mapping
        token_to_idx = {token_id: i for i, token_id in enumerate(label_ids)}
        class_labels = torch.tensor([token_to_idx.get(tl.item(), 0) for tl in true_labels], device=logits.device)
        
        # Filter logits to only include the candidate label tokens
        reduced_logits = logits[:, label_ids] # [batch_size, num_classes]
        
        ce_loss = F.cross_entropy(reduced_logits, class_labels)
        bacc_loss = self.compute_bacc_surrogate_loss(reduced_logits, class_labels, len(label_ids))
        return ce_loss + self.beta * bacc_loss

class CustomLogProbTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Pop custom arguments before calling super().__init__
        self.alpha = kwargs.pop('alpha', 5.0)
        self.beta = kwargs.pop('beta', 0.3)
        self.label_tokens = kwargs.pop('label_tokens', ("0", "1", "2", "3", "4", "5"))
        self.label_ids = None
        
        # Pop tokenizer to avoid "unexpected keyword argument" in older or custom Trainer versions
        tokenizer = kwargs.pop('tokenizer', None)
        
        # Now kwargs only contains arguments valid for the base Trainer
        super().__init__(*args, **kwargs)
        
        # Manually set tokenizer if it was provided
        if tokenizer is not None and not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        
    def _get_label_ids(self):
        if self.label_ids is None:
            self.label_ids = []
            for token in self.label_tokens:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                self.label_ids.append(ids[0])
        return self.label_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard Causal LM loss for the whole sequence
        # This ensures the model learns the overall structure
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            base_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Additional BACC surrogate loss for the label token
            # Find the position of the label token (last non-pad token)
            batch_size = logits.size(0)
            label_pos = attention_mask.sum(dim=1) - 1 # [batch_size]
            
            # The logit that predicts the label is at label_pos - 1
            # (Since shift_logits[t] corresponds to shift_labels[t])
            # Wait, easier to use non-shifted logits: logits[i, label_pos[i]-1] predicts labels[i, label_pos[i]]
            
            target_logits = []
            target_labels = []
            
            label_ids_list = self._get_label_ids()
            
            for i in range(batch_size):
                pos = label_pos[i].item()
                if pos > 0:
                    # Logits at position pos-1 predict token at position pos
                    l_idx = pos - 1
                    target_logits.append(logits[i, l_idx].unsqueeze(0))
                    target_labels.append(labels[i, pos].unsqueeze(0))
            
            if len(target_logits) > 0:
                target_logits = torch.cat(target_logits, dim=0) # [B, V]
                target_labels = torch.cat(target_labels, dim=0) # [B]
                
                # Check if these labels are in our expected label_tokens
                # Some tasks might have different number of classes
                # For simplicity, we filter only those present in label_ids_list
                mask = torch.tensor([tl.item() in label_ids_list for tl in target_labels], device=logits.device)
                if mask.any():
                    filtered_logits = target_logits[mask]
                    filtered_labels = target_labels[mask]
                    
                    # We need to know how many classes for THIS sample/task
                    # But multi-task means classes vary.
                    # Simplified: Use all candidate labels ("0" to "5")
                    token_to_idx = {token_id: i for i, token_id in enumerate(label_ids_list)}
                    class_labels = torch.tensor([token_to_idx[tl.item()] for tl in filtered_labels], device=logits.device)
                    reduced_logits = filtered_logits[:, label_ids_list]
                    
                    evaluator = ImprovedLogProbEvaluator(None, None, self.alpha, self.beta)
                    bacc_loss = evaluator.compute_bacc_surrogate_loss(reduced_logits, class_labels, len(label_ids_list))
                    
                    loss = base_loss + self.beta * bacc_loss
                else:
                    loss = base_loss
            else:
                loss = base_loss
        else:
            loss = outputs.get("loss")
            
        return (loss, outputs) if return_outputs else loss

def create_improved_logprob_trainer(base_orchestrator_class):
    """Create improved Log-Prob orchestrator class"""
    
    class ImprovedLogProbOrchestrator(base_orchestrator_class):
        def __init__(self, *args, alpha: float = 5.0, beta: float = 0.3, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.beta = beta
            self.logprob_evaluator = None
            
        def _setup_logprob_evaluator(self):
            if self.logprob_evaluator is None:
                if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                    self.logprob_evaluator = ImprovedLogProbEvaluator(
                        self.model, self.tokenizer, self.alpha, self.beta
                    )
        
        def get_trainer(self, training_args, train_dataset, eval_dataset, data_collator, class_weights):
            # Use CustomLogProbTrainer instead of standard Trainer
            return CustomLogProbTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                alpha=self.alpha,
                beta=self.beta,
                tokenizer=self.tokenizer
            )
        
        def evaluate_task_with_logprob(self, task_config, test_data):
            self._setup_logprob_evaluator()
            return self.logprob_evaluator.evaluate_task_with_improved_logprob(
                task_config, test_data
            )
            
    return ImprovedLogProbOrchestrator
