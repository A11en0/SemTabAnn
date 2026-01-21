import os
import json
import torch
import numpy as np

import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from transformers import TrainingArguments, Trainer
from src.training.hf_trainers.hf_trainer_sft import compute_metrics
from transformers.trainer_utils import EvalLoopOutput
from sklearn.metrics import classification_report, confusion_matrix


class GLCTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.use_aum = kwargs.pop("use_aum", True)
        self.anchor_budget = kwargs.pop("anchor_budget", 20)        
        self.trusted_weight = kwargs.pop("trusted_weight", 1.0) 

        self.noisy_weight = kwargs.pop("noisy_weight", 1.0)
        self.correction_type = kwargs.pop("correction_type", "glc") 
        self.use_corrected_labels = kwargs.pop("use_corrected_labels", True) 
        self.seed = kwargs.pop("seed", 42)
        super().__init__(*args, **kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_class = getattr(self, "processing_class", self.tokenizer)
        
        self.num_train_columns = self._compute_num_train_columns(self.train_dataset)

        self.corrected_mask = torch.zeros(self.num_train_columns, dtype=torch.bool, device=self.device)
        if self.use_corrected_labels:
            corrected_path = os.path.join(self.args.output_dir, "llm_corrected_labels.json")
            if not os.path.isfile(corrected_path):
                up_one_dir = os.path.abspath(os.path.join(self.args.output_dir, ".."))
                corrected_path_up = os.path.join(up_one_dir, "llm_corrected_labels.json")
                if os.path.isfile(corrected_path_up):
                    corrected_path = corrected_path_up
                    print(f"[Info] Corrected labels not found in output_dir, using parent dir: {corrected_path}")
                else:
                    print("[Warning] Corrected labels json not found in both output_dir and parent dir; skipping application.")
                    corrected_path = None
            if corrected_path is not None and os.path.isfile(corrected_path):
                print(f"[Config] Applying corrected labels from: {corrected_path}")
                self._apply_corrected_weak_labels(corrected_path)
            else:
                print("[Config] Skipping corrected weak labels application.")
        else:
            print("[Config] Skipping corrected weak labels application.")
        
        self.num_labels = self.model.config.num_labels

        # AUM init
        self.num_total_samples = getattr(self.train_dataset, "num_samples", -1)
        self.aum_sums = torch.zeros(self.num_total_samples, dtype=torch.float32)
        self.aum_counts = torch.zeros(self.num_total_samples, dtype=torch.long)
        
        self.entropy_sums = torch.zeros(self.num_total_samples, dtype=torch.float32)
        self.loss_sums = torch.zeros(self.num_total_samples, dtype=torch.float32)
        
        self.is_threshold_record = torch.zeros(self.num_total_samples, dtype=torch.bool)

        self.gt_record = torch.full((self.num_total_samples,), -1, dtype=torch.long)
        self.train_label_record = torch.full((self.num_total_samples,), -1, dtype=torch.long)        

        self.cached_gt_labels = self._cache_gt_labels()
        if self.anchor_budget > 0:
            self.labeled_mask = self._select_stratified_samples()
            self.save_trusted_info(self.args.output_dir)
        else:
            self.labeled_mask = torch.zeros(self.num_train_columns, dtype=torch.bool, device=self.device)
        
        print(f"[{self.correction_type.upper()}] Selected {self.labeled_mask.sum()} trusted samples.")
        if self.correction_type == "glc":
            self.C_hat = self._estimate_transition_matrix()
            self.C_tensor = torch.tensor(self.C_hat, dtype=torch.float32, device=self.device)
            print(f"[GLC] Transition Matrix Estimated.\nShape: {self.C_tensor.shape}")
        elif self.correction_type == "trusted_only":
            print(f"[Trusted Only] Skipping matrix estimation. Will only use trusted samples for fine-tuning.")
        elif self.correction_type == "weak_only":
            print(f"[Weak Only] Skipping matrix estimation. Will only use weak labels for fine-tuning.")
        elif self.correction_type == "trusted_weak_mix":
            print(f"[Trusted Weak Mix] Skipping matrix estimation. Will use trusted samples (GT) and weak labels for fine-tuning.")
        else:
            print(f"[Mix] Skipping matrix estimation. Will perform Direct SFT.")

    def save_trusted_info(self, output_dir):
        if not hasattr(self.train_dataset, "table_df"):
            print("[Warning] train_dataset does not have 'table_df', skipping trusted info save.")
            return

        print(f"[Trusted Info] Exporting trusted sample map to {output_dir}...")
        records = []

        mask_np = self.labeled_mask.cpu().numpy()
        for _, row in tqdm(self.train_dataset.table_df.iterrows(), total=len(self.train_dataset.table_df), desc="Exporting Trusted Info"):
            t_id = row.get("table_id", row.get("table_name", f"table_{_}"))
            t_id = t_id.split("|||")[0]
                        
            gids = row["global_col_indices"]
            if hasattr(gids, "tolist"):
                gids = gids.tolist()
            
            for col_idx, gid in enumerate(gids):
                gid = int(gid)
                is_trusted = False
                
                if gid < len(mask_np):
                    is_trusted = bool(mask_np[gid])
                
                records.append({
                    "global_col_id": gid,
                    "table_id": t_id,
                    "column_id": col_idx,
                    "is_trusted": is_trusted,
                    "is_trusted_int": 1 if is_trusted else 0
                })
                
        df = pd.DataFrame(records)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "trusted_samples_info.csv")
        df.to_csv(save_path, index=False)
        print(f"[Trusted Info] Saved {len(df)} records (Trusted: {df['is_trusted_int'].sum()}) to {save_path}")

    def _apply_corrected_weak_labels(self, corrected_path: str):
        if not os.path.isfile(corrected_path): return        
        corrections = json.load(open(corrected_path, "r", encoding="utf-8"))
        if not corrections: return

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

            if int(gid) < len(self.corrected_mask):
                self.corrected_mask[int(gid)] = True
            
            applied += 1
        self.corrected_mask = self.corrected_mask.to(self.device)
        print(f"[Corrected Label] Applied {applied} corrected weak labels.")
    
    def _cache_gt_labels(self) -> torch.Tensor:
        print("[GLC] Scanning dataset to cache GT labels...")
        cached_labels = torch.full((self.num_train_columns,), -1, dtype=torch.long, device="cpu")
        loader = self.get_train_dataloader()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Caching GT"):
                gids = batch["global_col_indices"].view(-1).cpu()
                if "labels" in batch:
                    cached_labels[gids] = batch["labels"].cpu()
        return cached_labels
    
    def _select_stratified_samples(self) -> torch.Tensor:
        gt_np = self.cached_gt_labels.numpy()
        mask = torch.zeros(self.num_train_columns, dtype=torch.bool, device=self.device)        
        
        current_seed = self.seed if self.seed is not None else 42
        rng = np.random.default_rng(current_seed)
        is_ratio_mode = 0.0 < self.anchor_budget < 1.0        
        for cls_id in range(self.num_labels):
            indices = np.where(gt_np == cls_id)[0]
            if len(indices) == 0:
                continue
            
            total_cls_samples = len(indices)            
            if is_ratio_mode:
                count = int(np.ceil(total_cls_samples * self.anchor_budget))
            else:
                count = int(self.anchor_budget)

            final_count = min(total_cls_samples, count)            
            if final_count > 0:
                chosen = rng.choice(indices, final_count, replace=False)
                mask[torch.tensor(chosen, device=self.device)] = True
        
        mode_str = f"Ratio={self.anchor_budget:.2%}" if is_ratio_mode else f"Fixed={int(self.anchor_budget)}"
        print(f"[Anchor] Anchor Selection ({mode_str}): Selected {mask.sum()} / {self.num_train_columns} samples total.")
        return mask
    
    def _compute_num_train_columns(self, train_dataset) -> int:
        if hasattr(train_dataset, "table_df"):
            max_gid = -1
            for _, row in train_dataset.table_df.iterrows():
                g_max = int(row["global_col_indices"].max().item())
                if g_max > max_gid: max_gid = g_max
            return max_gid + 1
        return len(train_dataset)
    
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
    
    def _estimate_transition_matrix(self, epsilon=1e-4) -> np.ndarray:
        print("[GLC] Estimating Transition Matrix C...")
        C_count = np.zeros((self.num_labels, self.num_labels))
        loader = self.get_train_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Estimating Matrix"):
                batch = self._prepare_inputs(batch)
                gids = batch["global_col_indices"].view(-1).to(self.device)                
                
                batch_mask = self.labeled_mask[gids]
                if batch_mask.sum() == 0: continue
                batch_mask_cpu = batch_mask.cpu()
                gt = batch["labels"][batch_mask].cpu().numpy()
                weak = batch["weak_labels"].cpu()[batch_mask_cpu].numpy()
                np.add.at(C_count, (gt, weak), 1)

        C_count += epsilon
        row_sums = C_count.sum(axis=1, keepdims=True)
        return C_count / row_sums

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        gids = inputs["global_col_indices"].view(-1)
        labels = inputs["labels"].view(-1)
        weak_labels = inputs["weak_labels"].view(-1)

        is_threshold_batch = inputs.get("is_threshold", torch.zeros_like(labels)).view(-1).bool()
        is_trusted = self.labeled_mask[gids]
        is_corrected = self.corrected_mask[gids]

        aum_target = weak_labels.clone()

        if is_trusted.sum() > 0:
            aum_target[is_trusted] = labels[is_trusted]
        
        logits, _ = self._forward_with_features(model, input_ids)
        if self.use_aum and model.training:
            with torch.no_grad():                
                target_idx = aum_target 
                gids_cpu = gids.cpu()
                
                # --- 1. AUM Calculation ---
                target_logits = logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                
                clone_logits = logits.clone()
                min_val = torch.finfo(logits.dtype).min
                clone_logits.scatter_(1, target_idx.unsqueeze(1), min_val)
                max_other_logits, _ = clone_logits.max(dim=1)
                
                margins = target_logits - max_other_logits
                
                self.aum_sums.index_add_(0, gids_cpu, margins.cpu())
                self.aum_counts.index_add_(0, gids_cpu, torch.ones_like(gids_cpu, dtype=torch.long))

                # --- 2. Entropy Calculation (For Entropy Sampling) ---
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                self.entropy_sums.index_add_(0, gids_cpu, entropy.cpu())
                
                # --- 3. Loss Calculation (For High-Loss Sampling) ---
                per_sample_loss = F.cross_entropy(logits, target_idx, reduction='none')
                self.loss_sums.index_add_(0, gids_cpu, per_sample_loss.cpu())

                # --- 4. Records ---
                self.gt_record[gids_cpu] = labels.cpu()
                self.train_label_record[gids_cpu] = target_idx.cpu()

                if is_threshold_batch.sum() > 0:
                    self.is_threshold_record[gids_cpu[is_threshold_batch.cpu()]] = True

        total_loss_sum = torch.tensor(0.0, device=self.device)
        total_samples = 0

        if self.correction_type == "trusted_only":
            if is_trusted.sum() > 0:
                loss_trusted = F.cross_entropy(logits[is_trusted], labels[is_trusted], reduction='sum')
                total_loss_sum += (loss_trusted * self.trusted_weight)
                total_samples += is_trusted.sum()
        
        elif self.correction_type == "weak_only":
            is_noisy = (~is_trusted) & (~is_threshold_batch)
            if is_noisy.sum() > 0:
                loss_weak = F.cross_entropy(logits[is_noisy], weak_labels[is_noisy], reduction='sum')
                total_loss_sum += (loss_weak * self.noisy_weight)
                total_samples += is_noisy.sum()
        
        elif self.correction_type == "trusted_weak_mix":
            is_noisy = (~is_trusted) & (~is_threshold_batch)
            if is_trusted.sum() > 0:
                loss_trusted = F.cross_entropy(logits[is_trusted], labels[is_trusted], reduction='sum')
                total_loss_sum += (loss_trusted * self.trusted_weight)
                total_samples += is_trusted.sum()
            if is_noisy.sum() > 0:
                loss_weak = F.cross_entropy(logits[is_noisy], weak_labels[is_noisy], reduction='sum')
                total_loss_sum += (loss_weak * self.noisy_weight)
                total_samples += is_noisy.sum()
        
        else:
            # Full GLC logic
            is_noisy = (~is_trusted) & (~is_threshold_batch)
            
            # A. Trusted (Standard CE, High Weight)
            if is_trusted.sum() > 0:
                loss_trusted = F.cross_entropy(logits[is_trusted], labels[is_trusted], reduction='sum')
                total_loss_sum += (loss_trusted * self.trusted_weight)
                total_samples += is_trusted.sum()

            # B. Threshold (Standard CE, Weight 1.0)
            if is_threshold_batch.sum() > 0:
                loss_threshold = F.cross_entropy(logits[is_threshold_batch], weak_labels[is_threshold_batch], reduction='sum')
                total_loss_sum += loss_threshold
                total_samples += is_threshold_batch.sum()

            # C. Noisy Data Handling
            if is_noisy.sum() > 0:
                is_truly_noisy = is_noisy & (~is_corrected)
                is_corrected_noisy = is_noisy & is_corrected
                
                if is_corrected_noisy.sum() > 0:
                    loss_corrected = F.cross_entropy(logits[is_corrected_noisy], weak_labels[is_corrected_noisy], reduction='sum')
                    total_loss_sum += (loss_corrected * self.trusted_weight) 
                    total_samples += is_corrected_noisy.sum()

                if is_truly_noisy.sum() > 0:
                    weak_labels_truly_noisy = weak_labels[is_truly_noisy]
                    
                    if self.correction_type == "glc":
                        probs = F.softmax(logits[is_truly_noisy], dim=-1)
                        probs_corrected = torch.matmul(probs, self.C_tensor)
                        probs_corrected = torch.clamp(probs_corrected, min=1e-7, max=1.0)
                        loss_noisy = F.nll_loss(torch.log(probs_corrected), weak_labels_truly_noisy, reduction='sum')
                    else:
                        # Fallback for "mix" mode
                        loss_noisy = F.cross_entropy(logits[is_truly_noisy], weak_labels_truly_noisy, reduction='sum')
                    
                    total_loss_sum += (loss_noisy * self.noisy_weight)
                    total_samples += is_truly_noisy.sum()

        final_loss = total_loss_sum / (total_samples + 1e-8)
        if return_outputs: return final_loss, logits
        return final_loss
    
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
    
    def save_aum_results(self, output_dir):
        print("[AUM] Saving scores (including Entropy & Loss)...")
        counts = self.aum_counts.float()
        counts[counts == 0] = 1.0
        
        avg_aum = self.aum_sums / counts        
        avg_entropy = self.entropy_sums / counts
        avg_loss = self.loss_sums / counts        
        
        df = pd.DataFrame({
            "global_col_indices": range(len(avg_aum)),
            "aum": avg_aum.numpy(),
            "entropy": avg_entropy.numpy(),
            "loss": avg_loss.numpy(),            
            "count": self.aum_counts.numpy(),
            "is_threshold_sample": self.is_threshold_record.numpy(),
            "train_label": self.train_label_record.numpy(),
            "gt_label": self.gt_record.numpy()
        })
        df["is_clean_label"] = (df["train_label"] == df["gt_label"])
        
        df = df[df["count"] > 0]
        df.to_csv(os.path.join(output_dir, "aum_scores.csv"), index=False)

    def _export_embeddings(self, dataloader, split_name: str, output_dir: str, file_name: str):        
        print(f"[Embedding] Exporting {split_name} embeddings (CLS pooled)...")
        self.model.eval()

        embeddings = []
        gids = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Embeddings-{split_name}"):
                batch = self._prepare_inputs(batch)
                input_ids = batch["input_ids"]
                _, cls_vec = self._forward_with_features(self.model, input_ids)
                embeddings.append(cls_vec.cpu().numpy())
                gids.append(batch["global_col_indices"].view(-1).cpu().numpy())
                if "labels" in batch:
                    labels.append(batch["labels"].view(-1).cpu().numpy())

        if not embeddings:
            print(f"[Embedding] No data to export for {split_name}.")
            return
        
        emb_arr = np.concatenate(embeddings, axis=0)
        gid_arr = np.concatenate(gids, axis=0)
        meta = {
            "embeddings": emb_arr,
            "global_col_indices": gid_arr,
        }
        if labels:
            meta["labels"] = np.concatenate(labels, axis=0)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, file_name)
        np.save(out_path, meta, allow_pickle=True)
        print(f"[Embedding] Saved {len(emb_arr)} {split_name} embeddings to {out_path}")
    
    def save_performance_by_frequency(self, dataset, output_dir, split_groups=[0.2, 0.5]):
        print("[Analysis] Running performance breakdown by class frequency...")
        self.model.eval()
        
        output = self.predict(dataset)
        y_true = np.array(output.label_ids)
        y_pred = np.array(output.predictions)
        
        report = classification_report(
            y_true, y_pred, 
            labels=range(self.num_labels), 
            target_names=self.train_dataset.type_ontology, 
            output_dict=True,
            zero_division=0
        )
        
        train_labels = self.cached_gt_labels.numpy()
        class_counts = Counter(train_labels[train_labels != -1])
        
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        total_classes = len(sorted_classes)

        head_idx = int(total_classes * split_groups[0])
        torso_idx = int(total_classes * split_groups[1])
        
        groups = {
            "Head": [c[0] for c in sorted_classes[:head_idx]],
            "Torso": [c[0] for c in sorted_classes[head_idx:torso_idx]],
            "Tail": [c[0] for c in sorted_classes[torso_idx:]]
        }

        ontology = self.train_dataset.type_ontology
        results = []
        
        for group_name, class_indices in groups.items():
            f1_scores = []
            for idx in class_indices:
                class_name = ontology[idx]
                if class_name in report:
                    f1_scores.append(report[class_name]['f1-score'])
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            results.append({
                "Group": group_name,
                "Macro-F1": avg_f1,
                "Class_Count": len(class_indices)
            })
            print(f"  - {group_name}: {avg_f1:.4f} (Classes: {len(class_indices)})")

        df_results = pd.DataFrame(results)
        save_path = Path(output_dir) / "long_tail_analysis.csv"
        df_results.to_csv(save_path, index=False)
        
        with open(Path(output_dir) / "full_class_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Long-tail analysis saved to {save_path}")

    def save_noise_matrix(self, dataset, output_dir, filename_prefix="model"):
        from sklearn.metrics import confusion_matrix
        print(f"[Noise Matrix] Generating {filename_prefix} noise matrix...")
        self.model.eval()
        
        output = self.predict(dataset)

        y_true = np.array(output.label_ids)
        y_pred = np.array(output.predictions)

        num_labels = self.model.config.num_labels
        cm = confusion_matrix(y_true, y_pred, labels=range(num_labels), normalize='true')
        
        ontology = getattr(dataset, "type_ontology", [])
        if not ontology and hasattr(self.train_dataset, "type_ontology"):
            ontology = self.train_dataset.type_ontology

        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
            
        matrix_file = save_path / f"{filename_prefix}_noise_matrix.npy"
        np.save(matrix_file, cm)
        
        ontology_file = save_path / f"{filename_prefix}_ontology.json"
        with open(ontology_file, "w", encoding="utf-8") as f:
            json.dump(ontology, f, ensure_ascii=False)
            
        print(f"✅ {filename_prefix.upper()} Noise Matrix saved to {matrix_file}")

    def save_train_embeddings(self, output_dir, file_name="train_embedding.npy"):
        loader = self.get_train_dataloader()
        self._export_embeddings(loader, "train", output_dir, file_name)

    def save_eval_embeddings(self, output_dir, file_name="test_embedding.npy"):
        loader = self.get_eval_dataloader()
        self._export_embeddings(loader, "eval", output_dir, file_name)


def create_hf_trainer(
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
        save_total_limit=3,
        logging_steps=1,
        logging_dir=config.output_dir,
        report_to=["swanlab"],
        remove_unused_columns=False
    )
    
    training_args.warmup_epochs = 3
    seed = getattr(config, "seed", 42)
    use_aum = getattr(config, "use_aum", False)
    anchor_budget = getattr(config, "anchor_budget", 50)
    correction_type = getattr(config, "correction_type", "glc").lower()
    use_corrected_labels = getattr(config, "use_corrected_labels", True)
    
    if correction_type == "trusted_only" or correction_type == "weak_only" or correction_type == "trusted_weak_mix":
        default_weight = 1.0

    trusted_weight = getattr(config, "trusted_weight", default_weight)
    noisy_weight = getattr(config, "noisy_weight", 1.0)
    
    print(f"🚀 Trainer Type: {correction_type.upper()} | Budget: {anchor_budget} | Trusted Weight: {trusted_weight} | Use Corrected Labels: {use_corrected_labels}")
    
    trainer = GLCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,        
        use_aum=use_aum,
        anchor_budget=anchor_budget,
        correction_type=correction_type,
        trusted_weight=trusted_weight,
        noisy_weight=noisy_weight,
        use_corrected_labels=use_corrected_labels,
        seed=seed,
        **kwargs,
    )
    
    return trainer