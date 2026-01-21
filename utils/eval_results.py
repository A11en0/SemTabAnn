# -*- coding: utf-8 -*-
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

from src.training.hf_trainers.igd_hf_trainer import IGDTrainer, compute_metrics
from typing import Dict, Any, Tuple
from src.llm.data_utils import load_data_and_ontology
from src.training.adapters.config_adapter import HFTrainingConfig
from src.llm.llm_bootstrapper import LLMBootstrapper

from src.models.qwen_models import Qwen3ForMultiOutputClassification
from src.models.bert_models import BertForMultiOutputClassification
from src.dataset.data_converters import MultiPLLDataset
from src.training.adapters.igd_data_adapter import IGDDataCollator
from transformers import TrainingArguments


def load_model(model_path: str, device: torch.device) -> BertForMultiOutputClassification:
    """Loads a trained model from a checkpoint."""
    print(f"Loading model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = BertForMultiOutputClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully to {device}.")
    return model

def analyze_errors(
    igd_preds: np.ndarray,
    kd_preds: np.ndarray,
    llm_top1: np.ndarray,
    llm_candidates: Any, # Keep original dataset for detailed info    
    labels: np.ndarray,
    label_map: Dict
) -> Dict[str, Any]:
    """Analyzes and categorizes prediction differences."""
    n_samples = len(labels)
    results = {
        "total_samples": n_samples,
        "both_correct": 0, "both_wrong": 0,
        "kd_correct_igd_wrong": 0, "kd_wrong_igd_correct": 0,
        "details_type_a": [], "details_type_b": [], "details_type_c": [],
    }
    
    for i in range(n_samples):
        true_label = labels[i]
        kd_pred = kd_preds[i]
        igd_pred = igd_preds[i]
        if llm_candidates:
            llm_pred = llm_top1[i]
            llm_candidate = llm_candidates[i]
        
        kd_is_correct = (kd_pred == true_label)
        igd_is_correct = (igd_pred == true_label)

        # Simplified sample info gathering
        if llm_candidates:
            sample_info = {
                "index": i,
                "true_label": label_map.get(true_label, true_label) if label_map else true_label,
                "kd_pred": label_map.get(kd_pred, kd_pred) if label_map else kd_pred,
                "igd_pred": label_map.get(igd_pred, igd_pred) if label_map else igd_pred,
                "llm_top1": label_map.get(llm_pred, llm_pred) if label_map else llm_pred,
                "llm_candidates": [label_map.get(idx, idx) for idx in llm_candidate if idx >= 0]
            }
        else:
            sample_info = {
                "index": i,
                "true_label": label_map.get(true_label, true_label) if label_map else true_label,
                "kd_pred": label_map.get(kd_pred, kd_pred) if label_map else kd_pred,
                "igd_pred": label_map.get(igd_pred, igd_pred) if label_map else igd_pred,
            }

        # Categorize
        if kd_is_correct and igd_is_correct:
            results["both_correct"] += 1
        elif not kd_is_correct and not igd_is_correct:
            results["both_wrong"] += 1
            results["details_type_a"].append(sample_info)
        elif not kd_is_correct and igd_is_correct:
            results["kd_wrong_igd_correct"] += 1
            results["details_type_b"].append(sample_info)
        elif kd_is_correct and not igd_is_correct:
            results["kd_correct_igd_wrong"] += 1
            results["details_type_c"].append(sample_info)            

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate IGD vs. Top-1 KD models (Simplified).")
    parser.add_argument("--model_name", required=True, help="Path to IGD checkpoint.")
    # parser.add_argument("--data_dir", required=True, default="./data/sotab_v2/rand_small_cta_1000")
    parser.add_argument("--config_path", required=True, help="Path to IGD checkpoint.")
    parser.add_argument("--igd_model_path", required=True, help="Path to Top-1 KD checkpoint.")
    parser.add_argument("--kd_model_path", required=True, help="Path to Top-1 KD checkpoint.")
    parser.add_argument("--eval_data_path", required=True, help="Path to evaluation data.")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--output_analysis_file", type=str, default=None, help="Optional JSON output for analysis.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for main process).") # Default 0 simpler for debugging
    args = parser.parse_args()
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Tokenizer & Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data config for load_data_and_ontology
    print(f"📋 Loading config: {args.config_path}")
    config = HFTrainingConfig.from_yaml(args.config_path)
    data_config = {
        'data': {
            'base_path': config.data_dir,
            'dataset_type': config.dataset_type,
            'cv_index': 0,
            'multicol_only': config.dataset_name.startswith("msato"),
            'limit_test': getattr(config, 'limit_test', None),
            'type_ontology_path': config.data_dir + "/type_ontology.txt"
        }
    }

    # Load training data with type ontology
    print("📊 Loading training data...")
    train_table_instances, type_ontology = load_data_and_ontology(data_config, split="train")
    label_map = {k: v for k, v in enumerate(type_ontology)}
    
    # Load test data
    print("📊 Loading test data...")
    # test_table_instances, _ = load_data_and_ontology(data_config, split="test")
    
    ### DEBUG: For quick test
    # train_table_instances = train_table_instances[:1000]
    # test_table_instances = test_table_instances[:100]
    
    # print(f"✅ Loaded {len(train_table_instances)} training tables, {len(test_table_instances)} test tables")
    
    # Setup LLM bootstrapping for IGD training
    print("\n🔧 Setting up LLM bootstrapping...")
    
    bootstrapper = LLMBootstrapper(
        config=config,
        seed=config.seed,
        dataset_type=config.dataset_name,
        max_workers=config.max_workers
    )
    
    # Apply LLM bootstrapping to training dataset
    print("🔧 Applying LLM bootstrapping to training data...")
    
    # Choose bootstrapping method based on configuration
    print("   Using LLM bootstrapping...")
    train_instances = bootstrapper.apply_llm_bootstrapping(train_table_instances, type_ontology)

    eval_dataset = MultiPLLDataset(
        instances=train_instances,
        tokenizer=tokenizer,
        max_length=config.max_length,
        device='cpu',
        type_ontology=type_ontology,
        config=config
    )
    
    # Create IGD data collator
    print("🔧 Creating IGD data collator...")
    data_collator = IGDDataCollator(
        tokenizer=tokenizer,
        max_candidates=getattr(config, 'igd_max_candidates', 10),
    )
    print(f"Loaded evaluation data: {len(eval_dataset)} samples.")
    
    # --- Load Models ---
    eval_args = TrainingArguments(
        output_dir=f"{config.output_dir}/eval",
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        dataloader_num_workers=config.dataloader_num_workers,
        logging_dir=f"{config.output_dir}/logs",
        report_to=[],
        remove_unused_columns=False,
        do_train=False,
        do_eval=True
    )
    
    def build_trainer(model_path):
        # Create model
        model = Qwen3ForMultiOutputClassification.from_pretrained(model_path, num_labels=config.num_classes)
        # model = BertForMultiOutputClassification.from_pretrained(model_path, num_labels=config.num_classes)
        print(f"✅ Model created from {config.model_name}")        
        trainer = IGDTrainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        return trainer

    igd_trainer = build_trainer(args.igd_model_path)
    kd_trainer = build_trainer(args.kd_model_path)
    
    # --- Evaluate ---
    # print("\n🚀 Evaluating IGD model...")
    igd_results = igd_trainer.evaluate()
    print("\n🚀 Evaluating KD model...")
    kd_results = kd_trainer.evaluate()

    # --- Metrics ---
    print("\n=== 📈 IGD vs KD Performance ===")
    for name, res in [("IGD", igd_results), ("KD", kd_results)]:
        micro = res.get("eval_micro_f1", 0)
        macro = res.get("eval_macro_f1", 0)
        print(f"{name} Model - Micro F1: {micro:.4f}, Macro F1: {macro:.4f}")

    # --- Prediction arrays for analysis ---
    print("\nExtracting raw predictions for error analysis...")
    igd_preds = np.argmax(igd_results["predictions"], axis=-1)
    kd_preds = np.argmax(kd_results["predictions"], axis=-1)
    labels = igd_results["label_ids"]
    llm_top1, llm_candidates = [], []
    for x in eval_dataset:
        tmp_top1, tmp_candidates = [], []
        for i, _ in enumerate(x['col_idx']):
            tmp_top1.append(x["candidate_labels"][i][0])
            tmp_candidates.append(x["candidate_labels"][i])
        llm_top1.extend(tmp_top1)
        llm_candidates.extend(tmp_candidates)
    llm_top1 = np.array(llm_top1)
    
    errors = analyze_errors(igd_preds, kd_preds, llm_top1, llm_candidates, labels, label_map)
    print("\n=== 🧩 Error Breakdown ===")
    print(json.dumps({k: v for k, v in errors.items() if not k.startswith("details_")}, indent=2))

    if args.output_analysis_file:
        with open(args.output_analysis_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved error details to {args.output_analysis_file}")


if __name__ == "__main__":
    main()

