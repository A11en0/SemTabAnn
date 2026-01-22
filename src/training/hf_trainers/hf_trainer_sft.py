import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.cluster import KMeans
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import EvalLoopOutput


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    micro_f1 = f1_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}

class SupervisedTrainer(Trainer):
    """
    Baseline: Train using only a small amount of fully supervised data with random sampling.
    No weak labels, no contrastive learning.
    """
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_correct_labels = getattr(args, "use_correct_labels", False)
        if self.use_correct_labels:
            corrected_path = os.path.join(self.args.output_dir, "llm_corrected_labels.json")
            self._apply_corrected_weak_labels(corrected_path)
        
        self.use_true_label = getattr(args, "use_true_label", False)
        self.active_budget_ratio = getattr(args, "active_budget_ratio", 0.05)
        self.num_train_columns = self._compute_num_train_columns(train_dataset)
        self.labeled_mask = self._select_random_samples()
        print(f"[Baseline] Randomly selected {self.labeled_mask.sum()} samples ({self.active_budget_ratio:.1%}) for Supervised Training.")

    def _compute_num_train_columns(self, train_dataset) -> int:
        if hasattr(train_dataset, "table_df"):
            max_gid = -1
            for _, row in train_dataset.table_df.iterrows():
                g_max = int(row["global_col_indices"].max().item())
                if g_max > max_gid: max_gid = g_max
            return max_gid + 1
        return len(train_dataset)

    def _select_random_samples(self) -> torch.Tensor:
        """Randomly generate mask"""
        mask = torch.zeros(self.num_train_columns, dtype=torch.bool, device=self.device)
        
        num_select = int(self.num_train_columns * self.active_budget_ratio)
        if num_select == 0:
            print("[Warning] Budget is too small, selecting at least 1 sample.")
            num_select = 1
        
        indices = torch.randperm(self.num_train_columns, device=self.device)[:num_select]
        mask[indices] = True
        return mask

    def _apply_corrected_weak_labels(self, corrected_path: str):
        """Overwrite weak labels with corrected results based on global_col_indices, skip if file doesn't exist."""
        corrections = json.load(open(corrected_path, "r", encoding="utf-8"))
        ontology = self.train_dataset.type_ontology
        label_to_id = {str(name): idx for idx, name in enumerate(ontology)}
        gid_index = {
            int(gid): (row_idx, pos)
            for row_idx, row in self.train_dataset.table_df.iterrows()
            for pos, gid in enumerate(row["global_col_indices"].tolist())
        }
        
        applied = 0
        for item in corrections:
            gid = item.get("g_id")
            label_text = item.get("new_label")
            if gid is None or label_text is None:
                continue
            
            label_id = label_to_id.get(str(label_text))
            loc = gid_index.get(int(gid))
            if label_id is None or loc is None:
                continue
            
            row_idx, pos = loc
            flag_val = self.train_dataset.table_df.at[row_idx, "is_threshold"][pos]
            flag_val = flag_val.item() if hasattr(flag_val, "item") else flag_val
            if int(flag_val) == 1:
                continue
            
            weak_tensor = self.train_dataset.table_df.at[row_idx, "weak_label_tensor"].clone()
            weak_tensor[pos] = label_id
            self.train_dataset.table_df.at[row_idx, "weak_label_tensor"] = weak_tensor
            applied += 1
        print(f"[Corrected Label] Applied {applied} corrected weak labels.")
    
    def _forward_with_features(self, model, input_ids):
        outputs = model(input_ids=input_ids, return_features=True)

        if isinstance(outputs, tuple):
            logits = outputs[0]
            pooled = outputs[1]
        else:
            logits = outputs.logits
            pooled = getattr(outputs, "hidden_states", None)

        if logits.dim() == 3:
            cls_token_id = self.processing_class.cls_token_id
            cls_positions = torch.nonzero(input_ids == cls_token_id, as_tuple=False)
            batch_idx, pos_idx = cls_positions[:, 0], cls_positions[:, 1]
            logits = logits[batch_idx, pos_idx, :]
            pooled = pooled[batch_idx, pos_idx, :]

        return logits, pooled

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        
        if "global_col_indices" not in inputs or not model.training:
            logits, _ = self._forward_with_features(model, input_ids)
            loss = torch.tensor(0.0, device=self.device)
            if "labels" in inputs:
                loss = self.ce_loss(logits, inputs["labels"])
            return (loss, logits) if return_outputs else loss

        gids = inputs["global_col_indices"].view(-1)
        is_labeled = self.labeled_mask[gids]
        logits, _ = self._forward_with_features(model, input_ids)
        
        if self.use_true_label:
            labels = inputs["labels"]
        else:
            labels = inputs["weak_labels"]
        
        loss = self.ce_loss(logits[is_labeled], labels[is_labeled])

        if return_outputs:
            return loss, logits
        return loss
    
    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_true_labels, all_logits = [], [], []

        if prediction_loss_only is None:
            prediction_loss_only = self.args.prediction_loss_only

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                batch = self._prepare_inputs(batch)
                input_ids = batch["input_ids"].to(self.args.device)

                logits, _ = self._forward_with_features(self.model, input_ids)

                if not prediction_loss_only:
                    if "labels" not in batch:
                        raise ValueError("Evaluation requires 'labels'.")
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"].to(self.args.device))
                    total_loss += loss.item()
                
                predictions = logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().detach().numpy().tolist())
                all_true_labels.extend(batch["labels"].cpu().detach().numpy().tolist())
                all_logits.extend(logits.cpu().detach().numpy().tolist())

        avg_loss = total_loss / max(len(dataloader), 1) if not prediction_loss_only else 0.0
        eval_pred = (np.array(all_logits), np.array(all_true_labels))

        metrics = {}
        if self.compute_metrics is not None:
            raw_metrics = self.compute_metrics(eval_pred)
            for k, v in raw_metrics.items():
                metrics[f"{metric_key_prefix}_{k}"] = v
        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_true_labels,
            metrics=metrics,
            num_samples=len(all_predictions),
        )
    
    def save_train_embeddings(self, output_dir: str, filename: str):
        pass
    
    def save_eval_embeddings(self, output_dir: str, filename: str):
        pass

    def save_aum_results(self, output_dir: str):
        pass


def create_hf_trainer_sft(
    model: nn.Module,
    config,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    data_collator: Optional[Any] = None,
    **kwargs
):
    import swanlab
    swanlab.init(project=f"{config.project_name}", experiment_name=f"{config.experiment_name}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=False,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        save_total_limit=config.save_total_limit,
        logging_steps=1,
        logging_dir=config.output_dir,
        report_to=["swanlab"],
        remove_unused_columns=False # Must keep, otherwise global_col_indices will be dropped
    )
    
    training_args.warmup_epochs = getattr(config, "warmup_epochs", 5)
    training_args.use_correct_labels = getattr(config, "use_correct_labels", False)
    training_args.use_true_label = getattr(config, "use_true_label", True)
    training_args.active_budget_ratio = getattr(config, "active_budget_ratio", 0.2) # 5% annotation amount
    
    trainer = SupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        **kwargs
    )
    
    return trainer