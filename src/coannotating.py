import os
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm.data_utils import serialize_df_to_string, load_data_and_ontology
from src.utils.util import OpenAIProvider
from src.training.adapters.config_adapter import HFTrainingConfig
from sklearn.metrics import f1_score

class CoAnnotatingSystem:
    def __init__(self, config, type_ontology, save_dir):
        """
        初始化系统，配置 LLM 提供者 [cite: 51, 196]
        """
        self.type_ontology = type_ontology
        self.save_dir = save_dir
        
        # 从您的配置文件中读取 API 配置
        prov_cfg = config.llm_bootstrapper.get("providers", {})
        self.provider = OpenAIProvider(
            api_key=prov_cfg.get("qwen", {}).get("api_key", ""),
            base_url=prov_cfg.get("qwen", {}).get("base_url", "")
        )

    def generate_7_prompts(self, item):
        """
        严格实现论文 Table 1 的 7 种提示词扰动类型
        """
        onto_str = ", ".join(self.type_ontology)
        # Use a column reference like "Column {idx}" to avoid leaking ground-truth labels.
        col_idx = item.get("col_idx", None)
        col_ref = f"Column {col_idx+1}"
        vals = item['values']
        context = item['table_text']
        
        # 1. 基础指令, 2. 序列交换, 3. 语义改写, 4. T/F, 5. 问答, 6. 多选, 7. 确认偏差
        return [
            f"Table Context:\n{context}\nPlease label the type of {col_ref} from: [{onto_str}].", 
            f"Values: {vals}\nColumn Reference: {col_ref}\nIn the provided table context, what is the type? Choices: [{onto_str}].",
            f"Classify the category of the following column data: {col_ref}. Examples: {vals}. Ontology: {onto_str}.",
            # f"Is the semantic type of column '{col_ref}' in the list [{onto_str}]? Provide the label name only.",
            # f"Identify the most suitable semantic type for '{col_ref}' in this table. Choices: [{onto_str}].",
            # f"Choose the best option for column '{col_ref}':\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(self.type_ontology)]),
            # f"I think the type of '{col_ref}' is in our ontology. Please provide the specific label from [{onto_str}]." 
        ]

    def calculate_entropy(self, labels):
        """
        实现公式: u_i = -sum(P * ln P) [cite: 153]
        """
        if not labels: return 1.0
        counts = Counter(labels)
        total = len(labels)
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def extract_label(self, response):
        """将 LLM 回复映射回本体标签 [cite: 196]"""
        for label in self.type_ontology:
            if label.lower() in response.lower():
                return label
        return "Ambiguous"

    def process_instance(self, item, model):
        """
        实例级不确定性估算 (Instance-level Expertise Estimation) [cite: 55]
        """
        prompts = self.generate_7_prompts(item)
        responses = []
        for p in prompts:
            try:
                resp = self.provider.generate_response(p, model, {})
                responses.append(self.extract_label(resp))
            except:
                responses.append("Error")

        # 计算不同提示词下的预测熵值
        entropy = self.calculate_entropy(responses)
        # 多数投票确定 LLM 最终标签 
        final_label = Counter(responses).most_common(1)[0][0]

        return {
            "g_id": item['global_id'],
            "entropy": round(entropy, 4),
            "final_llm_label": final_label,
            "raw_votes": responses
        }

def run_co_annotation(args):
    config = HFTrainingConfig.from_yaml(args.config)
    train_tables, type_ont = load_data_and_ontology({
        "data": {
            "base_path": config.data_dir, 
            "dataset_type": config.dataset_type, 
            "cv_index": 0, 
            "multicol_only": False, 
            "type_ontology_path": os.path.join(config.data_dir, "type_ontology.txt")
        }
    }, split="test")
    table_map = {t.table_id: t for t in train_tables}

    all_instances = []    
    for t_inst in train_tables:
        df = t_inst.table
        for col_idx, col_name in enumerate(df.columns):
            all_instances.append({
                "global_id": f"{t_inst.table_id}",
                "col_idx": col_idx,
                "col_name": col_name,
                "values": df.iloc[:, col_idx].dropna().astype(str).tolist()[:10],
                "table_text": serialize_df_to_string(df, sample_size=5)
            })
    
    # print(len(train_tables))
    # 构建 ground truth map
    gt_map = {}
    for t_inst in train_tables:
        df = t_inst.table
        for col_idx, col_name in enumerate(df.columns):
            key = t_inst.table_id
            label = t_inst.labels
            label = None
            if isinstance(t_inst.labels, dict):
                if col_name in t_inst.labels:
                    label = t_inst.labels[col_name]

            if label is not None:
                gt_map[key] = label

    current_seed = 42
    rng = np.random.default_rng(current_seed)
    
    system = CoAnnotatingSystem(config, type_ont, args.save_dir)
    raw_results = []        
    print(f"ALL instances: {len(all_instances)}, Label length: {config.num_classes}.")

    print(f"🚀 Processing {len(all_instances)} instances to estimate uncertainty...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(system.process_instance, item, args.model) for item in all_instances]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CoAnnotating"):
            res = fut.result()
            if res: raw_results.append(res)
    
    # 按 class 分组并分层采样（每类固定采样数量）
    raw_results.sort(key=lambda x: x['entropy'])
    class_to_samples = {}
    for res in raw_results:
        gid = res['g_id']
        if gid not in gt_map:
            continue
        true_label = gt_map[gid]
        class_to_samples.setdefault(true_label, []).append(res)

    trusted_set = set()
    for label, samples in class_to_samples.items():
        # 按熵值降序（不确定性高的优先人工）
        samples.sort(key=lambda x: x['entropy'], reverse=True)
        class_total = len(samples)
        if class_total == 0:
            continue
        # 预算解释：args.budget < 1 -> 百分比（每类比例）；>=1 -> 绝对数量（每类）
        if args.budget < 1:
            class_budget = max(1, int(class_total * args.budget))
        else:
            class_budget = min(class_total, int(args.budget))

        # 使用 RNG 选择 class_budget 个样本（从排序后的 topK 中随机挑选以避免顺序偏差）
        top_candidates = samples[:max(class_budget, 1)]
        chosen = rng.choice(range(len(top_candidates)), size=min(len(top_candidates), class_budget), replace=False)
        for idx in chosen:
            trusted_set.add(top_candidates[idx]['g_id'])

    print(f"📊 Stratified per-class sampling: selected {len(trusted_set)} samples across {len(class_to_samples)} classes")
    
    y_true = []
    y_pred = []
    final_output = []

    for res in raw_results:
        gid = res['g_id']
        if gid not in gt_map: continue
        
        true_label = gt_map[gid]
        y_true.append(true_label)
        
        if gid in trusted_set:
            # 分配给人工：直接读取 GT
            res["decision"] = "HUMAN"
            res["final_label"] = true_label
        else:
            # 分配给 LLM
            res["decision"] = "LLM"
            res["final_label"] = res["final_llm_label"]
        
        y_pred.append(res["final_label"])
        final_output.append(res)
    
    label_to_id = {str(label): idx for idx, label in enumerate(type_ont)}
    label_to_id_lower = {k.lower(): v for k, v in label_to_id.items()}

    def _map_label_to_id(label):
        if isinstance(label, int):
            return int(label)
        s = str(label).strip()
        # direct match with original string keys
        if s in label_to_id:
            return int(label_to_id[s])
        # case-insensitive match
        s_lower = s.lower()
        if s_lower in label_to_id_lower:
            return int(label_to_id_lower[s_lower])
        # numeric string fallback
        try:
            if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # unknown label
        return -1

    y_true_mapped = [_map_label_to_id(x) for x in y_true]
    y_pred_mapped = [_map_label_to_id(x) for x in y_pred]

    # Filter out pairs where either true or pred couldn't be mapped (-1).
    valid_pairs = [(t, p) for t, p in zip(y_true_mapped, y_pred_mapped) if t != -1 and p != -1]
    if len(valid_pairs) == 0:
        raise ValueError("No valid mapped labels for evaluation after normalization.")
    y_true_mapped, y_pred_mapped = zip(*valid_pairs)
    y_true_mapped = list(y_true_mapped)
    y_pred_mapped = list(y_pred_mapped)

    micro_f1 = f1_score(y_true_mapped, y_pred_mapped, average='micro')
    macro_f1 = f1_score(y_true_mapped, y_pred_mapped, average='macro')
    
    human_count = len([r for r in final_output if r["decision"] == "HUMAN"])
    llm_count = len([r for r in final_output if r["decision"] == "LLM"])
    
    print(f"\n📊 CoAnnotating Evaluation (Budget: {args.budget}):")
    print(f"   - Total Eval Samples: {len(y_true)}")
    print(f"   - LLM Auto-labeled: {llm_count}")
    print(f"   - Human (GT) used: {human_count}")
    print(f"   - ✅ Micro-F1: {micro_f1:.4f}")
    print(f"   - ✅ Macro-F1: {macro_f1:.4f}")

    save_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": {"micro_f1": micro_f1, "macro_f1": macro_f1}, "data": final_output}, f, indent=2, ensure_ascii=False)

    # # 保存最终预测结果（含决策、最终标签、LLM 原始投票等）
    # preds_path = os.path.join(args.save_dir, "co_annotation_predictions.json")
    # with open(preds_path, "w", encoding="utf-8") as f:
    #     json.dump({"predictions": final_output}, f, indent=2, ensure_ascii=False)

    # # 同时保存原始 LLM 结果以便后续分析
    # raw_path = os.path.join(args.save_dir, "co_annotation_raw_results.json")
    # with open(raw_path, "w", encoding="utf-8") as f:
    #     json.dump({"raw_results": raw_results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--col_meta_path", required=True, help="Path to col_meta.json")
    parser.add_argument("--save_dir", required=True, help="Directory to save output")
    parser.add_argument("--budget", type=float, default=0, help="Percentage (0-1) or count of samples for human annotation")
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--workers", type=int, default=10)    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)        
    run_co_annotation(args)

