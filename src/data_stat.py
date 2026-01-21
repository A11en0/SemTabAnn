#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.training.adapters.config_adapter import HFTrainingConfig
from src.llm.data_utils import load_data_and_ontology


def compute_stats(table_instances, type_ontology):
    num_tables = len(table_instances)
    num_types = len(type_ontology) if type_ontology is not None else 0
    total_cols = 0
    labeled_cols = 0
    cols_per_table = []
    
    for inst in table_instances:
        try:
            cols = len(inst.table.columns) if getattr(inst, "table", None) is not None else 0
            total_cols += cols
            cols_per_table.append(cols)
            # labels is a dict mapping column_name -> label (may be empty)
            labels = getattr(inst, "labels", {}) or {}
            labeled_cols += len(labels)
        except Exception:
            continue

    pct_labeled = (labeled_cols / total_cols * 100) if total_cols > 0 else 0.0
    min_cols = min(cols_per_table) if cols_per_table else 0
    max_cols = max(cols_per_table) if cols_per_table else 0
    avg_cols = (total_cols / len(cols_per_table)) if cols_per_table else 0.0

    return {
        "num_tables": num_tables,
        "num_types": num_types,
        "total_cols": total_cols,
        "labeled_cols": labeled_cols,
        "pct_labeled": pct_labeled,
        "min_cols_per_table": min_cols,
        "max_cols_per_table": max_cols,
        "avg_cols_per_table": avg_cols,
    }


def print_summary(name, stats):
    print(f"\n=== {name} ===")
    print(f"# Tables:\t{stats['num_tables']}")
    print(f"# Types:\t{stats['num_types']}")
    print(f"Total # Cols:\t{stats['total_cols']}")
    print(f"# Labeled Cols:\t{stats['labeled_cols']}")
    print(f"% Labeled:\t{stats['pct_labeled']:.2f}%")
    print(f"Min Cols/Table:\t{stats.get('min_cols_per_table', 0)}")
    print(f"Max Cols/Table:\t{stats.get('max_cols_per_table', 0)}")
    print(f"Avg Cols/Table:\t{stats.get('avg_cols_per_table', 0.0):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics for table-column datasets")
    parser.add_argument("--config", type=str, default="configs/config_default.yaml", help="Path to YAML config file.")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both", help="Which split to analyze")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for the stats")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    config = HFTrainingConfig.from_yaml(config_path)

    data_cfg = {
        "data": {
            "base_path": config.data_dir,
            "dataset_type": config.dataset_type,
            "cv_index": 0,
            "multicol_only": config.dataset_name.startswith("msato"),
            "limit_test": getattr(config, "limit_test", None),
            "type_ontology_path": config.data_dir + "/type_ontology.txt",
        }
    }

    results = {}
    
    if args.split in ("train", "both"):
        print("Loading train split...")
        train_tables, type_ont = load_data_and_ontology(data_cfg, split="train")
        train_stats = compute_stats(train_tables, type_ont)
        print_summary("Train", train_stats)
        results["train"] = train_stats
    
    if args.split in ("test", "both"):
        print("Loading test split...")
        test_tables, type_ont_test = load_data_and_ontology(data_cfg, split="test")
        # prefer ontology from train if available
        type_ont_final = type_ont if args.split == "both" and 'type_ont' in locals() else type_ont_test
        test_stats = compute_stats(test_tables, type_ont_final)
        print_summary("Test", test_stats)
        results["test"] = test_stats

    if args.split == "both":
        # combined
        combined_tables = []
        if "train" in results:
            combined_tables.extend(train_tables)
        if "test" in results:
            combined_tables.extend(test_tables)
        combined_stats = compute_stats(combined_tables, type_ont_final)
        print_summary("Combined", combined_stats)
        results["combined"] = combined_stats

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved stats to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


