import os
import re
import json
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).parent.parent))

from src.llm.data_utils import serialize_df_to_string, load_data_and_ontology
from src.utils.util import OpenAIProvider, ResponseCache
from src.training.adapters.config_adapter import HFTrainingConfig

try:
    import faiss
except ModuleNotFoundError:
    faiss = None


def load_col_meta(col_meta_path):
    """
    读取 col_meta.json 构建映射: g_id (int) -> meta_dict
    """
    print(f"📖 Loading column metadata from {col_meta_path}...")
    with open(col_meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta_map = {}
    for item in data:
        g_id = int(item['g_id'])
        meta_map[g_id] = {
            "g_id": g_id,
            "table_id": item['table_id'].split("|||")[0],
            "col_idx": int(item['col_idx']),
            "col_name": item.get('col_name', f"Column {int(item['col_idx'])+1}"),
            "ground_truth_id": item.get('ground_truth'),
            "pseudo_label_id": item.get('pseudo_label')
        }
    return meta_map


def _parse_aum_value(aum):
    """
    Robustly parse an AUM value which may be a float, int, or a string that contains
    logging text with a numeric value (e.g. ">>> ...\\n0.1932"). Returns a float or
    raises ValueError if parsing fails.
    """
    if aum is None:
        return None
    if isinstance(aum, (int, float)):
        return float(aum)
    if isinstance(aum, str):
        s = aum.strip()
        # Try direct float conversion first
        try:
            return float(s)
        except Exception:
            # Extract numeric tokens (floats or ints) and use the last one if present
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if matches:
                try:
                    return float(matches[-1])
                except Exception:
                    pass
    raise ValueError(f"Could not parse AUM value from: {repr(aum)}")


# =====================================================================
# Helper: AUM Analysis (integrated from utils/analyze_aum.py)
# =====================================================================
def verify_cleaning_quality(csv_path, save_dir=None):
    """
    读取 AUM 评分文件，计算噪音检测的查准率/查全率，并生成可视化。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
        
    print(f"📖 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. 确定阈值 (只看 Threshold Samples)
    threshold_samples = df[df['is_threshold_sample'] == True]
    if len(threshold_samples) == 0:
        print("Error: No threshold samples.")
        return
    
    cutoff = np.percentile(threshold_samples['aum'], 0.5)
    print(f"✂️  Cutoff (50th percentile / median): {cutoff:.4f}")
    
    # cutoff = -1.0
    # 2. 准备数据
    real_data = df[df['is_threshold_sample'] == False].copy()
    
    # 真实情况 (Ground Truth)
    real_data['is_actually_noise'] = (real_data['train_label'] != real_data['gt_label'])
    # AUM 预测 (Prediction)
    real_data['predicted_as_noise'] = (real_data['aum'] <= cutoff)
    
    # 3. 计算混淆矩阵元素
    tp = len(real_data[(real_data['predicted_as_noise'] == True) & (real_data['is_actually_noise'] == True)])
    fp = len(real_data[(real_data['predicted_as_noise'] == True) & (real_data['is_actually_noise'] == False)])
    fn = len(real_data[(real_data['predicted_as_noise'] == False) & (real_data['is_actually_noise'] == True)])
    tn = len(real_data[(real_data['predicted_as_noise'] == False) & (real_data['is_actually_noise'] == False)])
    
    # 4. 指标计算
    total_removed = tp + fp
    precision = tp / total_removed if total_removed > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("\n🔍 Verification Results:")
    print("-" * 30)
    print(f"   Total Samples: {len(real_data)}")
    print(f"   Actually Noisy: {tp + fn} | Actually Clean: {tn + fp}")
    print(f"   Removed by AUM: {total_removed} (TP={tp}, FP={fp})")
    print("-" * 30)
    print(f"   ✅ Precision (查准率): {precision:.2%} (被删掉的数据里，多少是真的噪音)")
    print(f"   ✅ Recall    (查全率): {recall:.2%} (所有的噪音里，你抓出了多少)")
    
    if fp > 0:
        print("\n⚠️  WARNING: You deleted useful data (Hard Samples)!")
        hard_samples = real_data[(real_data['predicted_as_noise'] == True) & (real_data['is_actually_noise'] == False)]
        print(hard_samples[['global_col_indices', 'aum', 'train_label', 'gt_label']].head())

    # =========================================================
    # 📊 可视化部分 (Visualization)
    # =========================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- 图 1: AUM 分布直方图 (Distribution) ---
    ax_hist = axes[0]
    
    # 分离数据用于画图
    clean_aum = real_data[real_data['is_actually_noise'] == False]['aum']
    noisy_aum = real_data[real_data['is_actually_noise'] == True]['aum']
    threshold_aum = threshold_samples['aum']

    # 画分布 (Hist + KDE)
    sns.histplot(clean_aum, color="green", label=f'Actual Clean (N={len(clean_aum)})', 
                 kde=True, stat="density", element="step", alpha=0.3, ax=ax_hist)
    sns.histplot(noisy_aum, color="red", label=f'Actual Noise (N={len(noisy_aum)})', 
                 kde=True, stat="density", element="step", alpha=0.3, ax=ax_hist)
    
    # 画阈值样本的分布 (参考线)
    sns.kdeplot(threshold_aum, color="gray", linestyle="--", label='Threshold Samples (Pure Noise)', ax=ax_hist)

    # 画 Cutoff 线
    ax_hist.axvline(cutoff, color='black', linestyle='-', linewidth=2, label=f'Cutoff ({cutoff:.2f})')
    
    # 标注区域
    ax_hist.set_title(f"AUM Distribution: Precision={precision:.2%} | Recall={recall:.2%}", fontsize=14)
    ax_hist.set_xlabel("AUM Score")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    
    # 在图上标注 FP 区域 (误删区)
    # 只要绿色分布跑到了黑线左边，那就是误删
    xlims = ax_hist.get_xlim()
    ylims = ax_hist.get_ylim()
    ax_hist.text(cutoff - (abs(cutoff)*0.1), ylims[1]*0.8, "Predicted NOISE\n(Remove)", 
                 horizontalalignment='right', color='darkred', fontweight='bold')
    ax_hist.text(cutoff + (abs(cutoff)*0.1), ylims[1]*0.8, "Predicted CLEAN\n(Keep)", 
                 horizontalalignment='left', color='darkgreen', fontweight='bold')
    
    # --- 图 2: 混淆矩阵 (Confusion Matrix) ---
    ax_cm = axes[1]
    
    # 构建矩阵数据
    cm_data = np.array([[tn, fp], [fn, tp]])
    # 标签
    labels = [['TN (Kept Clean)', 'FP (Deleted Clean)\n⚠️ "Hard Samples"'], 
              ['FN (Missed Noise)', 'TP (Deleted Noise)\n✅ "Success"']]
    
    # 归一化颜色便于观察 (按行归一化，看召回情况)
    sns.heatmap(cm_data, annot=np.array(labels), fmt='', cmap='Blues', 
                annot_kws={'size': 12, 'weight': 'bold'}, cbar=False, ax=ax_cm)
    
    # 在热力图上覆盖具体数值
    for i in range(2):
        for j in range(2):
            ax_cm.text(j+0.5, i+0.7, f"Count: {cm_data[i, j]}", 
                       ha="center", va="center", color="black", fontsize=11)

    ax_cm.set_title("Cleaning Confusion Matrix", fontsize=14)
    ax_cm.set_xlabel("AUM Prediction (Action)")
    ax_cm.set_xticklabels(['Keep', 'Remove'])
    ax_cm.set_ylabel("Actual Ground Truth")
    ax_cm.set_yticklabels(['Clean Data', 'Noisy Data'])

    # 保存
    save_path = os.path.join(save_dir, 'aum_analysis_dashboard.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.close()


# =====================================================================
# 1. Active Learning Sampler
# =====================================================================
class ActiveLearningSampler:
    def __init__(self, col_meta_path=None, aum=0):
        self.meta_map = load_col_meta(col_meta_path)
        try:
            self.aum = _parse_aum_value(aum)
        except ValueError as e:
            # print(f"⚠️ Warning: failed to parse AUM value provided to ActiveLearningSampler: {e}")
            self.aum = None
    
    def select_samples(self, table_instances, col_meta_path, aum_csv_path, strategy="aum", budget_ratio=0.1, num_clusters=20, output_path="to_be_annotated.json", train_embedding_path=None):
        table_map = {t.table_id: t for t in table_instances}

        print(f"📖 Reading AUM scores from {aum_csv_path}...")
        df_aum = pd.read_csv(aum_csv_path)

        candidates = df_aum[df_aum['is_threshold_sample'] == False]
        n_budget = int(len(candidates) * budget_ratio)
        
        target_gids = []

        # ==================================================
        # ✅ STRATEGY SELECTION LOGIC
        # ==================================================
        if strategy == "random":
            print(f"🎲 Strategy: Random Selection (Budget: {n_budget}, Ratio: {budget_ratio})")
            if n_budget > 0:
                selected_df = candidates.sample(n=n_budget, random_state=42)
                target_gids = selected_df["global_col_indices"].tolist()
            else:
                print("⚠️ Budget too small, selecting 0 samples.")
                
        elif strategy == "entropy":
            print(f"😕 Strategy: Uncertainty Sampling (Entropy) (Top-{n_budget})")
            # ✅ Check if entropy column exists (Trainer must save it)
            if 'entropy' not in candidates.columns:
                print("❌ Error: 'entropy' column missing in aum_scores.csv. Please update Trainer to save entropy.")
                return
            # Select highest entropy (Most uncertain)
            selected_df = candidates.nlargest(n_budget, 'entropy')
            target_gids = selected_df["global_col_indices"].tolist()

        elif strategy == "high_loss":
            print(f"🏋️ Strategy: Hardness Sampling (High Loss) (Top-{n_budget})")
            # ✅ Check if loss column exists
            if 'loss' not in candidates.columns:
                print("❌ Error: 'loss' column missing in aum_scores.csv. Please update Trainer to save loss.")
                return
            # Select highest loss (Most difficult)
            selected_df = candidates.nlargest(n_budget, 'loss')
            target_gids = selected_df["global_col_indices"].tolist()

        elif strategy == "aum":
            # ✅ AUM supports two modes: Threshold-based OR Budget-based
            if self.aum is not None:
                print(f"📉 Strategy: AUM (Threshold < {self.aum})")
                high_risk_df = candidates[candidates['aum'] < self.aum]
                target_gids = high_risk_df["global_col_indices"].tolist()
            else:
                print(f"📉 Strategy: AUM (Bottom-{n_budget} Lowest Values)")
                # Select lowest AUM (Most likely errors)
                selected_df = candidates.nsmallest(n_budget, 'aum')
                target_gids = selected_df["global_col_indices"].tolist()
        else:
             print(f"⚠️ Unknown strategy: {strategy}")
             return       

        # if strategy == "random":
        #     print(f"🎲 Strategy: Random Selection (Budget Ratio: {budget_ratio})")
        #     # Exclude threshold samples, sample from remaining
        #     candidates = df_aum[df_aum['is_threshold_sample'] == False]
        #     n_samples = int(len(candidates) * budget_ratio)
            
        #     if n_samples > 0:
        #         selected_df = candidates.sample(n=n_samples, random_state=42)
        #         target_gids = selected_df["global_col_indices"].tolist()
        #     else:
        #         print("⚠️ Budget too small, selecting 0 samples.")
        
        # # 🔄 AUM Strategy: Select samples below threshold
        # elif strategy == "aum":
        #     print(f"📉 Strategy: AUM (Threshold: {self.aum})")
        #     if self.aum is None:
        #         raise ValueError("AUM threshold must be provided for 'aum' strategy.")
        #     high_risk_df = df_aum[(df_aum['is_threshold_sample'] == False) & (df_aum['aum'] < self.aum)]
        #     target_gids = high_risk_df["global_col_indices"].tolist()
        # else:
        #      print(f"⚠️ Unknown strategy: {strategy}")
        #      return        
        
        # else:
        #     # Original AUM logic
        #     print(f"📉 Strategy: AUM (Threshold: {self.aum})")
        #     if self.aum is None:
        #         raise ValueError("AUM threshold must be provided for 'aum' strategy.")
        #     high_risk_df = df_aum[(df_aum['is_threshold_sample'] == False) & (df_aum['aum'] < self.aum)]
        #     target_gids = high_risk_df["global_col_indices"].tolist()
        
        if not target_gids:
            print("⚠️ No samples selected.")
            return
        
        print(f"✅ Selected {len(target_gids)} target samples.")

        texts = []
        sample_metadata = []
        
        precomputed_map = {}
        if train_embedding_path and os.path.exists(train_embedding_path):
            try:
                emb_obj = np.load(train_embedding_path, allow_pickle=True).item()
                gids_arr = emb_obj.get("global_col_indices")
                emb_arr = emb_obj.get("embeddings")
                if gids_arr is not None and emb_arr is not None and len(gids_arr) == len(emb_arr):
                    precomputed_map = {int(g): emb_arr[idx] for idx, g in enumerate(gids_arr)}
                    print(f"📥 Loaded {len(precomputed_map)} precomputed embeddings from {train_embedding_path}")
            except Exception as e:
                print(f"⚠️ Failed to load precomputed embeddings: {e}")

        embeds_buffer = [None] * len(target_gids)
        
        for g_id in tqdm(target_gids, desc="Extracting content"):
            if g_id not in self.meta_map: 
                continue
            
            meta = self.meta_map[g_id]
            tid = meta['table_id']
            cidx = meta['col_idx']
            if tid not in table_map: 
                continue
            
            t_instance = table_map[tid]
            if cidx >= len(t_instance.table.columns):
                print(f"⚠️ Column index {cidx} out of bounds for table {tid}, skipping g_id={g_id}")
                continue

            col_name = t_instance.table.columns[cidx]
            vals = t_instance.table.iloc[:, cidx].dropna().astype(str).tolist()[:10]
            col_str = serialize_df_to_string(t_instance.table[[col_name]], sample_size=10)
            # current_label = list(t_instance.labels.keys())[cidx]

            texts.append(col_str)
            sample_metadata.append({
                "g_id": g_id,
                "table_id": tid,
                "col_idx": cidx,
                "col_name": col_name,
                "ground_truth": meta.get("ground_truth_id"), # 保留 GT ID
                "current_label": meta.get("pseudo_label_id"),
                "values": vals
            })
            idx = len(sample_metadata) - 1
            if g_id in precomputed_map:
                embeds_buffer[idx] = precomputed_map[g_id]
            else:
                # print(f"⚠️ Missing precomputed embedding for gid={g_id}, skipping.")
                embeds_buffer[idx] = None

        # if strategy == "random":
        if strategy in ["random", "entropy", "high_loss"]:            
             selected_samples = sample_metadata
        else:        
            # 过滤掉缺失 embedding 的样本
            filtered = [(emb, meta) for emb, meta in zip(embeds_buffer, sample_metadata) if emb is not None]
            if not filtered:
                # print("❌ No embeddings available after filtering missing entries.")
                # return
                selected_samples = sample_metadata # Fallback
            else:
                embeddings, meta_subset = zip(*filtered)
                embeddings = np.array(embeddings)                
                # Only cluster if we have significantly more samples than clusters                
                n_clusters = min(num_clusters, len(embeddings))
                if n_clusters < len(embeddings):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(embeddings)
                    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
                    selected_samples = [meta_subset[idx] for idx in closest]
                else:
                    selected_samples = list(meta_subset)
            
            # embeddings, sample_metadata = zip(*filtered)
            # embeddings = np.array(embeddings)
            # kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42, n_init=10)
            # kmeans.fit(embeddings)
            # closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            
            # selected_samples = [sample_metadata[idx] for idx in closest]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected_samples, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(selected_samples)} samples to {output_path}")


class HITLCorrector:
    def __init__(self, config, aum=0, cache_dir=".llm_cache", col_meta_path=None, train_embedding_path=None, save_dir=None):
        self.config = config.llm_bootstrapper
        self.save_dir = save_dir
        
        try:
            self.aum = _parse_aum_value(aum)
        except ValueError as e:
            print(f"⚠️ Warning: failed to parse AUM value provided to HITLCorrector: {e}")
            self.aum = None
        self.providers = {}
        prov_cfg = self.config.get("providers", {})
        if "qwen" in prov_cfg:
            self.providers["qwen"] = OpenAIProvider(
                api_key=prov_cfg["qwen"].get("api_key", ""),
                base_url=prov_cfg["qwen"].get("base_url", "")
            )
        if "openai" in prov_cfg:
            self.providers["openai"] = OpenAIProvider(api_key=prov_cfg["openai"].get("api_key", ""))
        
        self.meta_map = load_col_meta(col_meta_path)
        self.human_examples = []
        self.index = None
        self.precomputed_map = {}
        if train_embedding_path and os.path.exists(train_embedding_path):
            try:
                emb_obj = np.load(train_embedding_path, allow_pickle=True).item()
                gids_arr = emb_obj.get("global_col_indices")
                emb_arr = emb_obj.get("embeddings")
                if gids_arr is not None and emb_arr is not None and len(gids_arr) == len(emb_arr):
                    self.precomputed_map = {int(g): emb_arr[idx] for idx, g in enumerate(gids_arr)}
                    print(f"📥 Loaded {len(self.precomputed_map)} precomputed embeddings from {train_embedding_path}")
            except Exception as e:
                print(f"⚠️ Failed to load precomputed embeddings: {e}")

    def load_human_annotations(self, annotation_path, type_ontology):
        """
        加载人工标注 (利用 Ground Truth ID 转换)
        """
        print(f"📖 Loading human examples from {annotation_path}...")
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        valid_examples = []
        for item in data:
            # 逻辑修正：只要有 ground_truth 字段，就视为有效
            if "ground_truth" in item:
                gt_id = int(item["ground_truth"])
                # 越界检查
                if 0 <= gt_id < len(type_ontology):
                    # 自动转换 ID -> Label String
                    item["corrected_label"] = type_ontology[gt_id]
                    item["reason"] = "Derived from validated Ground Truth." # 自动填充理由
                    valid_examples.append(item)
                else:
                    print(f"⚠️ GT ID {gt_id} out of bounds.")
                    
        self.human_examples = valid_examples
        print(f"📥 Loaded {len(self.human_examples)} valid human examples (with GT labels).")

    def build_human_index(self, table_instances, type_ontology):
        """
        从原始 Table Instances 读取内容构建索引
        """
        if faiss is None:
            raise ModuleNotFoundError("faiss is required for --step correct. Please install faiss-cpu/faiss-gpu.")
        if not self.human_examples: return
        print("⚡ Building index for human examples (from raw tables)...")
        
        table_map = {t.table_id: t for t in table_instances}
        final_examples = []
        embeddings = []

        for ex in tqdm(self.human_examples, desc="Indexing"):
            g_id = int(ex.get('g_id', -1))
            raw_tid = ex['table_id'].split("|||")[0]
            col_idx = int(ex['col_idx'])

            pseudo_label_str = "Unknown"
            if g_id in self.meta_map:
                p_id = self.meta_map[g_id].get("pseudo_label_id", -1)
                if p_id is not None and 0 <= p_id < len(type_ontology):
                    pseudo_label_str = type_ontology[p_id]            

            ex['pseudo_label'] = pseudo_label_str
            
            if raw_tid not in table_map: continue
            t_inst = table_map[raw_tid]
            
            if col_idx >= len(t_inst.table.columns): continue
            
            real_col_name = t_inst.table.columns[col_idx]
            ex['table_text'] = serialize_df_to_string(t_inst.table, sample_size=10)
            # col_df = t_inst.table[[real_col_name]]
            
            # 序列化：确保检索时的 query 和 index 格式一致
            # col_str = serialize_df_to_string(col_df, sample_size=10)
            # 更新 Example 数据，确保 Prompt 里用到的是真实数据
            ex['col_name'] = real_col_name
            # ex['pseudo_label'] = ...
            ex['values'] = t_inst.table.iloc[:, col_idx].dropna().astype(str).tolist()[:10]

            if ex['g_id'] in self.precomputed_map:
                embeddings.append(self.precomputed_map[ex['g_id']])
                final_examples.append(ex)
            else:
                print(f"⚠️ Missing precomputed embedding for gid={ex['g_id']}, skip indexing.")

        if not embeddings:
            print("❌ No embeddings available for human examples.")
            return
        
        self.human_examples = final_examples # 更新过滤后的列表
        embs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embs)
        
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        print("✅ Index built.")

    def retrieve_examples(self, target_gid, k=3):
        if faiss is None:
            raise ModuleNotFoundError("faiss is required for --step correct. Please install faiss-cpu/faiss-gpu.")
        if not self.index:
            print("❌ No index built for human examples.")
            return []
        if target_gid not in self.precomputed_map:
            print(f"⚠️ Missing embedding for target gid={target_gid}, skip retrieve.")
            return []
        qvec = np.array([self.precomputed_map[target_gid]], dtype="float32")
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec, k)
        return [self.human_examples[i] for i in I[0] if i < len(self.human_examples)]

    def compute_dataset_micro_f1(self, corrections, type_ontology):
        """
        使用修正结果重新计算全量数据集的 micro-f1。
        - y_true: meta_map 中存在的 ground_truth_id
        - y_pred: 优先使用修正后的标签，否则使用原 pseudo_label_id
        """
        label_to_id = {str(name): idx for idx, name in enumerate(type_ontology)}
        correction_map = {}
        for item in corrections:
            gid = item.get("g_id")
            new_label = item.get("new_label")
            if gid is None or new_label is None:
                continue
            lid = label_to_id.get(str(new_label))
            if lid is not None:
                correction_map[int(gid)] = lid

        y_true, y_pred_before, y_pred_after = [], [], []
        for gid, meta in self.meta_map.items():
            gt_id = meta.get("ground_truth_id")
            if gt_id is None:
                continue
            pred_before = meta.get("pseudo_label_id")
            pred_after = correction_map.get(gid, pred_before)
            # pred_after = gt_id
            if pred_before is None or pred_after is None:
                continue
            y_true.append(gt_id)
            y_pred_before.append(pred_before)
            y_pred_after.append(pred_after)
        
        if not y_true:
            print("⚠️ No ground truth labels available; skip micro-f1 recomputation.")
            return None, None

        micro_before = f1_score(y_true, y_pred_before, average="micro")
        micro_after = f1_score(y_true, y_pred_after, average="micro")
        print(f"📊 Dataset micro-f1 before correction: {micro_before:.4f}")
        print(f"📈 Dataset micro-f1 after correction:  {micro_after:.4f}")
        return micro_before, micro_after
    
    def build_correction_prompt(self, meta, type_ontology, examples, method="cot"):
        onto_str = ", ".join(type_ontology)
        target_alias = f"Column {meta.get('col_idx', '?') + 1}" if meta.get('col_idx') is not None else "Column"
        
        # === Zero-shot Prompt (Simple) ===
        if method == "zero_shot":
            return f"""You are a Data Quality Expert.
Task: Classify the Semantic Type of the target column.
Output JSON: {{"corrected_label": "..."}}

Ontology: [{onto_str}]

Target Table:
{meta['table_text']}

Target Column: {target_alias} (Name: {meta['col_name']})
Original Label (Likely Wrong): "{meta['current_label']}"

Please provide the correct label from the Ontology."""

        # === CoT Prompt (With Examples) ===
        examples_str = ""
        for i, ex in enumerate(examples):
            cidx = ex.get("col_idx")
            col_alias = f"Column {cidx+1}" if cidx is not None else "Column"
            examples_str += f"""
--- Example {i+1} ---
Table Context:
{ex.get('table_text', 'N/A')}
Target: {col_alias}
Original: "{ex.get('pseudo_label', 'Unknown')}"
Correct: "{ex['corrected_label']}"
"""

        return f"""You are a Data Quality Expert specializing in Column Type Annotation. The current label is known to be WRONG; do NOT output it again.

Rules:
1) Choose the corrected label ONLY from [Ontology].
2) The field [Original] is incorrect. You must pick a DIFFERENT label if it appears in the ontology.
3) Base your decision on column values and table context.
4) Output JSON: {{"original_label": "...", "reason": "...", "corrected_label": "..."}}

Ontology: [{onto_str}]

Manually corrected examples (for reasoning):
{examples_str}

Target:
Table Context: 
{meta['table_text']}
Target: {target_alias}
Original (wrong): "{meta['current_label']}"

Correction:"""        
    
#     def build_correction_prompt(self, meta, type_ontology, examples):
#         onto_str = ", ".join(type_ontology)
#         examples_str = ""
#         for i, ex in enumerate(examples):
#             cidx = ex.get("col_idx")
#             col_alias = f"Column {cidx+1}" if cidx is not None else "Column"
#             examples_str += f"""
# --- Example {i+1} ---
# Table Context:
# {ex.get('table_text', 'N/A')}
# Target: {col_alias}
# Original: "{ex.get('pseudo_label', 'Unknown')}"
# Correct: "{ex['corrected_label']}"
# """
#         # examples_str = ""
#         target_alias = f"Column {meta.get('col_idx', '?') + 1}" if meta.get('col_idx') is not None else "Column"
#         return f"""You are a Data Quality Expert specializing in Column Type Annotation. The current label is known to be WRONG; do NOT output it again.

# Rules:
# 1) Choose the corrected label ONLY from [Ontology].
# 2) The field [Original] is incorrect. You must pick a DIFFERENT label if it appears in the ontology.
# 3) Base your decision on column values and table context.
# 4) Output JSON: {{"original_label": "...", "reason": "...", "corrected_label": "..."}}

# Ontology: [{onto_str}]

# Manually corrected examples (for reasoning):
# {examples_str}

# Target:
# Table Context: 
# {meta['table_text']}
# Target: {target_alias}
# Original (wrong): "{meta['current_label']}"

# Correction:"""

    # def run_correction(self, table_instances, col_meta_path, aum_csv_path, type_ontology, model, k=3):
    def run_correction(self, table_instances, col_meta_path, aum_csv_path, type_ontology, model, k=3, method="cot"):
        # Check if correction metrics already exist
        metrics_path = os.path.join(self.save_dir, "llm_correction_metrics.json")
        if os.path.exists(metrics_path):
            print(f"📖 Found existing correction metrics at {metrics_path}, loading...")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            print(f"✅ Loaded existing metrics:")
            print(f"   Micro F1 before: {metrics.get('micro_f1_before', 'N/A')}")
            print(f"   Micro F1 after:  {metrics.get('micro_f1_after', 'N/A')}")
            print(f"   Number of corrections: {metrics.get('num_corrections', 'N/A')}")
            print("⏭️  Skipping correction step (metrics already exist).")
            return
        
        table_map = {t.table_id.split("|||")[0]: t for t in table_instances}
        
        df_aum = pd.read_csv(aum_csv_path)
        high_risk_df = df_aum[(df_aum['is_threshold_sample'] == False) & (df_aum['aum'] < self.aum)]
        target_gids = [gid for gid in high_risk_df["global_col_indices"]]
        target_metas = []
        
        for gid in target_gids:
            if gid not in self.meta_map:
                continue
            
            meta = self.meta_map[gid]
            tid = meta['table_id']
            if tid not in table_map:
                continue
            
            t_inst = table_map[tid]
            cidx = meta['col_idx']
            
            real_col_name = t_inst.table.columns[cidx]
            vals = t_inst.table.iloc[:, cidx].dropna().astype(str).tolist()[:10]
            col_str = serialize_df_to_string(t_inst.table[[real_col_name]], sample_size=10)
            table_text = serialize_df_to_string(t_inst.table, sample_size=10)
            current_label = type_ontology[meta['pseudo_label_id']]
            
            target_metas.append({
                "global_id": gid,
                "col_idx": cidx,
                "col_name": real_col_name,
                "values": vals,
                "current_label": current_label,
                "retrieval_text": col_str,
                "table_text": table_text
            })

        # _len, cnt = len(target_metas), 0
        # for v in target_metas:
            # if v['col_name'] == v['current_label']:
                # cnt += 1
        # ret = (_len - cnt) / _len
        # print(ret)
        print(f"🚀 Running correction on {len(target_metas)} samples via method: {method.upper()}...")
        
        target_metas = target_metas
        results = []
        def _worker(item):
            try:
                # demos = self.retrieve_examples(item['global_id'], k=k)
                # prompt = self.build_correction_prompt(item, type_ontology, demos)

                # 只有 COT 需要检索
                demos = []
                if method == "cot":
                    demos = self.retrieve_examples(item['global_id'], k=k)
                
                ### [MODIFIED] 传入 method
                prompt = self.build_correction_prompt(item, type_ontology, demos, method=method)

                if "qwen" in model:
                    resp = self.providers["qwen"].generate_response(prompt, model, {})
                else:
                    resp = self.providers["openai"].generate_response(prompt, model, {})

                match = re.search(r"\{.*\}", resp, re.DOTALL)
                if match:
                    res_json = json.loads(match.group(0))
                    print(res_json)
                    
                    gt_label = None
                    gt_id = self.meta_map.get(item['global_id'], {}).get("ground_truth_id")
                    if gt_id is not None and 0 <= gt_id < len(type_ontology):
                        gt_label = type_ontology[gt_id]

                    new_label = res_json.get("corrected_label")
                    return {
                        "g_id": item['global_id'],
                        "old_label": item['current_label'],
                        "new_label": new_label,
                        "reason": res_json.get("reason"),
                        "gt_label": gt_label,
                        "fixed": (gt_label is not None and new_label == gt_label)
                    }
            except Exception as e:
                pass
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, m) for m in target_metas]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                res = fut.result()
                if res: results.append(res)
        
        total = len(results)
        hit = 0
        for r in results:
            gid = r["g_id"]
            if gid in self.meta_map:
                gt_id = self.meta_map[gid].get("ground_truth_id")
                if gt_id is not None and 0 <= gt_id < len(type_ontology):
                    gt_label = type_ontology[gt_id]
                    if r.get("new_label") == gt_label:
                        hit += 1
        if total > 0:
            rate = hit / total * 100
            print(f"✅ Correction rate vs GT: {hit}/{total} ({rate:.2f}%)")
        else:
            print("⚠️ No correction results to evaluate.")
        
        micro_before, micro_after = self.compute_dataset_micro_f1(results, type_ontology)
        if micro_after is not None:
            metrics_path = os.path.join(self.save_dir, "llm_correction_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "micro_f1_before": micro_before,
                        "micro_f1_after": micro_after,
                        "num_corrections": len(results),
                    },
                    mf,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"✅ Saved micro-f1 metrics to {metrics_path}")

        out_path = os.path.join(self.save_dir, "llm_corrected_labels.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to {out_path}")
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["sample", "correct", "verify"])
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--base_dir", default=None, help="Base directory for outputs (e.g., outputs/default_1211). If provided, other paths will be auto-filled.")
    parser.add_argument("--col_meta_path", default=None, help="Path to col_meta.json (auto-filled from base_dir if not provided)")
    parser.add_argument("--aum_path", default=None, help="Path to aum_scores.csv (auto-filled from base_dir if not provided)")
    parser.add_argument("--annotation_path", default=None, help="Path to annotation file (auto-filled from base_dir if not provided)")
    parser.add_argument("--train_embedding_path", default=None, help="Path to precomputed train embeddings (auto-filled from base_dir if not provided)")
    parser.add_argument("--save_dir", default=None, help="Directory to save outputs (auto-filled from base_dir if not provided)")
    parser.add_argument("--aum", default=None, help="AUM cutoff threshold (float). Samples with aum < cutoff are selected.")
    parser.add_argument("--strategy", type=str, default="aum", help="Sampling strategy")
    parser.add_argument("--budget_ratio", type=float, default=0.1, help="Budget ratio for random sampling")
    parser.add_argument("--method", type=str, default="cot", help="Correction method")    
    
    args = parser.parse_args()
    
    if args.base_dir:
        base_dir = args.base_dir.rstrip('/')
        if args.col_meta_path is None:
            args.col_meta_path = os.path.join(base_dir, "col_meta.json")
        if args.aum_path is None:
            args.aum_path = os.path.join(base_dir, "aum_scores.csv")
        if args.annotation_path is None:
            args.annotation_path = os.path.join(base_dir, "to_be_annotated.json")
        if args.train_embedding_path is None:
            args.train_embedding_path = os.path.join(base_dir, "train_embedding.npy")
        if args.save_dir is None:
            args.save_dir = base_dir
    
    # Validate required paths
    if args.step in ["sample", "correct"]:
        if args.aum_path is None:
            parser.error("--aum_path is required (or provide --base_dir)")
        if args.col_meta_path is None:
            parser.error("--col_meta_path is required (or provide --base_dir)")
        if args.step == "correct" and args.annotation_path is None:
            parser.error("--annotation_path is required for 'correct' step (or provide --base_dir)")
    
    config = HFTrainingConfig.from_yaml(args.config)

    # Load Data
    data_cfg = {
        "data": {
            "base_path": config.data_dir,
            "dataset_type": config.dataset_type,
            "cv_index": 0,
            "multicol_only": config.dataset_name.startswith("msato"),
            "type_ontology_path": config.data_dir + "/type_ontology.txt"
        }
    }
    print("⏳ Loading tables...")
    train_tables, type_ont = load_data_and_ontology(data_cfg, split="train")
    
    if args.step == "sample":
        if args.strategy == "aum" and args.aum is None:
            parser.error("--aum is required when strategy is 'aum'")
        
        sampler = ActiveLearningSampler(col_meta_path=args.col_meta_path, aum=args.aum)
        
        sampler.select_samples(
            train_tables,
            str(args.col_meta_path),
            args.aum_path,
            strategy=args.strategy,
            budget_ratio=args.budget_ratio,
            num_clusters=20,
            output_path=args.annotation_path,
            train_embedding_path=args.train_embedding_path,
        )
    
    elif args.step == "correct":
        # if args.aum is None:
            # parser.error("--aum is required for 'correct' step")
        corrector = HITLCorrector(
            config,
            aum=args.aum,
            col_meta_path=args.col_meta_path,
            train_embedding_path=args.train_embedding_path,
            save_dir=args.save_dir
        )
        corrector.load_human_annotations(args.annotation_path, type_ontology=type_ont)
        corrector.build_human_index(train_tables, type_ontology=type_ont)
        corrector.run_correction(train_tables, str(args.col_meta_path), args.aum_path, type_ont, model="qwen3-max", k=5, method=args.method)



