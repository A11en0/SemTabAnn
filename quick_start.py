#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.training.adapters.config_adapter import HFTrainingConfig
from src.llm.data_utils import _set_seed
from src.dataset.data_converters import MultiPLLDataset, TablewiseDataset
from src.models.bert_models import BertForMultiOutputClassification, BertForMultiPairClassification
from src.utils.util import export_col_meta_from_multipll_dataset
from src.llm.data_utils import load_data_and_ontology
from src.llm.llm_bootstrapper import LLMBootstrapper
from src.training.hf_trainers.hf_trainer import create_hf_trainer
from src.training.hf_trainers.hf_trainer_sft import create_hf_trainer_sft
from src.training.hf_trainers.hf_trainer_dividmix import create_hf_trainer_dividmix
from src.training.adapters.data_adapter import DataCollator


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training interface")
    parser.add_argument("--config", type=str, default="configs/config_default.yaml", help="Path to YAML config file.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment_name from config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir from config.")
    parser.add_argument("--logging_dir", type=str, default=None, help="Override logging_dir from config.")
    parser.add_argument("--llm_save_dir", type=str, default=None, help="Override logging_dir from config.")
    parser.add_argument("--trainer_mode", type=str, default="sft", help="Override trainer_mode from config.")
    parser.add_argument("--correction_type", type=str, default="glc", help="Override correction_type from config.")
    parser.add_argument("--use_aum", action="store_true", help="Use AUM mode.")
    parser.add_argument("--anchor_budget", type=float, default=0.05, help="Override anchor_budget from config.")
    parser.add_argument("--trusted_weight", type=float, default=5.0, help="Override trusted_weight from config.")
    parser.add_argument("--noisy_weight", type=float, default=1.0, help="Override noisy_weight from config.")
    parser.add_argument("--use_corrected_labels", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use corrected weak labels (True/False).")
    parser.add_argument("--task", type=str, default="cta", choices=["cta", "cpa"], help="Task type: cta (default) or cpa.")    
    
    args, _ = parser.parse_known_args()
    return args


def main():
    """Run training using the unified training interface."""

    args = parse_args()
    config_path = args.config

    print(f"📋 Loading config: {config_path}")
    config = HFTrainingConfig.from_yaml(config_path)
    
    if args.task is not None:
        config.task = args.task
    else:
        config.task = "cta"
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.logging_dir is not None:
        config.logging_dir = args.logging_dir
    if args.llm_save_dir is not None:
        config.llm_save_dir = args.llm_save_dir
    if args.trainer_mode is not None:
        config.trainer_mode = args.trainer_mode
    if args.correction_type is not None:
        config.correction_type = args.correction_type
    if args.use_aum is not None:
        config.use_aum = args.use_aum
    if args.anchor_budget is not None:
        config.anchor_budget = args.anchor_budget
    if args.trusted_weight is not None:
        config.trusted_weight = args.trusted_weight
    if args.noisy_weight is not None:
        config.noisy_weight = args.noisy_weight
    if args.use_corrected_labels is not None:
        config.use_corrected_labels = args.use_corrected_labels
    
    print("=" * 60)
    print("🚀 UNIFIED TRAINING INTERFACE")
    print("=" * 60)

    print("📋 Basic Configuration:")
    print(f"   Task Type: {config.task.upper()}")
    print(f"   Method: {config.method}")
    print(f"   Training Mode: {config.training_mode}")
    print(f"   Dataset: {config.dataset_name}")
    print(f"   Data Directory: {config.data_dir}")
    print(f"   Loss Function: {config.loss_function}")
    print(f"   Model: {config.model_name}")
    print(f"   Max Length: {config.max_length}")
    print(f"   Num Classes: {config.num_classes}")

    print("\n📋 Training Parameters:")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Weight Decay: {config.weight_decay}")
    print(f"   Warmup Ratio: {config.warmup_ratio}")
    print(f"   Random Seed: {config.random_seed}")
    print(f"   Output Directory: {config.output_dir}")
    print(f"   Use Correct Labels: {config.use_corrected_labels}")
    
    print("=" * 60)
    
    try:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        _set_seed(config.seed)

        print("\n📊 Creating datasets with LLM bootstrapping...")
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
        
        print("📊 Loading datasets...")
        train_table_instances, type_ontology = load_data_and_ontology(data_config, split="train")
        test_table_instances, _ = load_data_and_ontology(data_config, split="test")
        
        # train_table_instances = train_table_instances
        print(f"✅ Loaded {len(train_table_instances)} training tables, {len(test_table_instances)} test tables")
        
        if config.use_aum:
            original_num_classes = len(type_ontology)
            config.num_classes = original_num_classes + 1
            print(f"⚠️ [AUM Mode] Adjusted num_classes from {original_num_classes} to {config.num_classes} (Added 1 Noise Class)")

        print("🔧 Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if "qwen" in config.model_name.lower():
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer created from {config.model_name}")

        print("🔧 Creating model...")
        # if "qwen" in config.model_name.lower():
        #     from src.models.qwen_models import Qwen3ForMultiOutputClassification
        #     model = Qwen3ForMultiOutputClassification.from_pretrained(
        #         config.model_name,
        #         num_labels=config.num_classes
        #     )
        # else:
        #     model = BertForMultiOutputClassification.from_pretrained(
        #         config.model_name,
        #         num_labels=config.num_classes
        #     )
        if config.task == 'cpa':
            print("   ➡️  Using BertForMultiPairClassification for CPA")
            model = BertForMultiPairClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_classes
            )
        else:
            print("   ➡️  Using BertForMultiOutputClassification for CTA")
            model = BertForMultiOutputClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_classes
            )
        print(f"✅ Model created from {config.model_name}")

        if getattr(config, 'use_lora', False):
            print("🔧 Applying LoRA (PEFT)...")

            lora_r = getattr(config, 'lora_r', 16)
            lora_alpha = getattr(config, 'lora_alpha', 32)
            lora_dropout = getattr(config, 'lora_dropout', 0.1)

            if not hasattr(config, 'lora_target_modules'):
                print("❌ Error: 'use_lora' is True, but 'lora_target_modules' is not defined in the config.")
                print("   Please specify target_modules (e.g., ['q_proj', 'v_proj']) in your YAML config.")
                sys.exit(1)

            lora_target_modules = config.lora_target_modules
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=None
            )
            model = get_peft_model(model, peft_config)
            print("✅ LoRA applied successfully.")
            model.print_trainable_parameters()
        
        print("\n🔧 Setting up LLM bootstrapping...")
        bootstrapper = LLMBootstrapper(
            config=config,
            seed=config.seed,
            dataset_type=config.dataset_name,
            max_workers=config.max_workers
        )

        print("🔧 Applying LLM bootstrapping to training data...")
        train_instances = bootstrapper.apply_llm_bootstrapping(train_table_instances, type_ontology)
        
        train_dataset = MultiPLLDataset(
            instances=train_instances,
            tokenizer=tokenizer,
            max_length=config.max_length,
            device='cpu',
            type_ontology=type_ontology,
            config=config,
            threshold_ratio=0.1 # Set 1% of samples as AUM threshold anchors
        )

        test_dataset = MultiPLLDataset(
            instances=test_table_instances,
            tokenizer=tokenizer,
            max_length=config.max_length,
            device='cpu',
            type_ontology=type_ontology,
            config=config,
            threshold_ratio=0.0  # No noise injection for test set
        )
        valid_dataset = test_dataset
        
        export_col_meta_from_multipll_dataset(train_dataset, config.output_dir)
        
        print("🔧 Creating trainer...")
        trainer_mode = getattr(config, "trainer_mode", "sft")
        if trainer_mode == "sft":        
            print("Create hf trainer....SFT...")
            trainer = create_hf_trainer_sft(
                model=model, 
                config=config, 
                train_dataset=train_dataset, 
                eval_dataset=valid_dataset, 
                tokenizer=tokenizer, 
                data_collator=DataCollator(tokenizer=tokenizer)
            )
        elif trainer_mode == "dividemix":
            print("Create hf trainer....DivideMix...")
            trainer = create_hf_trainer_dividmix(
                model=model, 
                config=config, 
                train_dataset=train_dataset, 
                eval_dataset=valid_dataset, 
                tokenizer=tokenizer, 
                data_collator=DataCollator(tokenizer=tokenizer)
            )
        else:
            trainer = create_hf_trainer(model=model, 
                config=config, 
                train_dataset=train_dataset, 
                eval_dataset=valid_dataset, 
                tokenizer=tokenizer, 
                data_collator=DataCollator(tokenizer=tokenizer)
            )
        
        print(f"✅ trainer created successfully")
        print(f"   Datasets: Train-tables={len(train_dataset)}, Test-tables={len(test_dataset)}")
        
        print(f"\n🏋️ Starting Training...")
        trainer.train()
        
        if config.use_aum and hasattr(trainer, "save_aum_results"):
            print("\n💾 Saving AUM Scores for Data Cleaning...")
            trainer.save_aum_results(config.output_dir)
        
        trainer.save_train_embeddings(config.output_dir, "train_embedding.npy")
        trainer.save_eval_embeddings(config.output_dir, "test_embedding.npy")
        
        print("\n📈 Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)        
        micro_f1 = test_results.get('eval_micro_f1', 'N/A')
        macro_f1 = test_results.get('eval_macro_f1', 'N/A')

        print("\n📊 Exporting Model Noise Matrix data for plotting...")
        trainer.save_noise_matrix(test_dataset, config.output_dir, "model_final")
        trainer.save_performance_by_frequency(test_dataset, config.output_dir)

        print("\n🎯 Final Results:")
        print("=" * 40)
        print(f"   Micro F1: {micro_f1:.4f}" if isinstance(micro_f1, (int, float)) else f"   Micro F1: {micro_f1}")
        print(f"   Macro F1: {macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else f"   Macro F1: {macro_f1}")
        
        # Save results to files
        print("\n💾 Saving evaluation results...")
        results_dir = Path(config.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (complete results)
        json_path = results_dir / "test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"   ✅ JSON results saved to: {json_path}")
        
        # Save as formatted text file
        txt_path = results_dir / "test_results.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Task: {config.task}\n")
            f.write(f"Experiment: {getattr(config, 'experiment_name', 'N/A')}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Metrics:\n")
            f.write("=" * 60 + "\n")
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"   {key}: {value:.6f}\n")
                else:
                    f.write(f"   {key}: {value}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Summary:\n")
            f.write("=" * 60 + "\n")
            f.write(f"   Micro F1: {micro_f1:.6f}\n" if isinstance(micro_f1, (int, float)) else f"   Micro F1: {micro_f1}\n")
            f.write(f"   Macro F1: {macro_f1:.6f}\n" if isinstance(macro_f1, (int, float)) else f"   Macro F1: {macro_f1}\n")
        print(f"   ✅ Text results saved to: {txt_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

