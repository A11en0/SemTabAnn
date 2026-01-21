"""
Utility functions for FIESTA system
"""
import random
from typing import Tuple
from pathlib import Path
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import os

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate response from LLM."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize OpenAI provider."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI client initialized with base_url: {base_url}")
            self.available = True
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.available = False
    
    def generate_response(self, prompt: str, model: str, config: Dict[str, Any]) -> str:
        """Generate response using OpenAI API."""
        print(f"Using {model} API...")
        if not self.available:
            raise RuntimeError("OpenAI client not available")
        if 'gpt-5' in model:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 1000),
                top_p=config.get('top_p', 0.95),
                presence_penalty=config.get('presence_penalty', 0.0),
                frequency_penalty=config.get('frequency_penalty', 0.0),
            )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str):
        """Initialize Anthropic provider."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("Anthropic library not installed. Install with: pip install anthropic")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.available = False
    
    def generate_response(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate response using Anthropic API."""
        if not self.available:
            raise RuntimeError("Anthropic client not available")
        
        response = self.client.messages.create(
            model=config.get('model', 'claude-3-sonnet-20240229'),
            max_tokens=config.get('max_tokens', 1000),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.95),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class ResponseCache:
    """Simple file-based cache for LLM responses."""
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        """Initialize cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enabled = True
    
    def _get_cache_key(self, table_id: str, config: Dict[str, Any] = None, cache_id: str = None) -> str:
        """Generate cache key from table_id and optional cache_id."""
        # Combine table_id and cache_id for more specific caching
        key_parts = [table_id]
        if cache_id:
            key_parts.append(f"cache_id:{cache_id}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get(self, table_id: str, config: Dict[str, Any] = None, cache_id: str = None) -> Optional[str]:
        """Get cached response."""
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(table_id, config, cache_id)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data['response']
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return None
    
    def set(self, table_id: str, config: Dict[str, Any] = None, response: str = None, cache_id: str = None) -> None:
        """Cache response."""
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(table_id, config, cache_id)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            cached_data = {
                'response': response,
                'timestamp': time.time(),
                'table_id': table_id,
                'config': config,
                'cache_id': cache_id
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            logger.debug(f"Cached response for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

def export_col_meta_from_multipll_dataset(train_dataset, save_dir):
    """
    导出包含 ground-truth 的 col_meta.json
    每个样本格式：
      {
        "g_id": int,
        "table_id": str,
        "col_idx": int,
        "col_name": str,
        "ground_truth": str
      }
    """
    rows = []
    
    for _, row in train_dataset.table_df.iterrows():
        table_id = row["table_id"]
        g_ids = row["global_col_indices"]
        col_names = row.get("col_names", None)
        labels = row.get("label_tensor", None)  # optional, may contain ground truth
        candidate_labels = row.get("candidate_labels", None)
        
        # tensor → list
        if isinstance(g_ids, torch.Tensor):
            g_ids = g_ids.tolist()

        # 如果没有列名，用 col_indices 占位
        if col_names is None:
            col_names = [f"Column {i+1}" for i in range(len(g_ids))]

        for col_idx, (col_name, gid) in enumerate(zip(col_names, g_ids)):
            rows.append({
                "g_id": int(gid),
                "table_id": str(table_id),
                "col_idx": int(col_idx),
                "col_name": str(col_name),
                "ground_truth": labels[col_idx].item(),
                "pseudo_label": candidate_labels[col_idx][0]
            })

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "col_meta.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Exported {len(rows)} entries with ground-truth to {save_path}")

# def export_col_meta_from_multipll_dataset(train_dataset, save_dir: str):
#     """
#     从 MultiPLLDataset.table_df 导出全局列元信息：
    
#     每一行是：
#       {
#         "g_id": 全局列 id （global_col_indices 里的值）,
#         "table_id": 这一列对应的 table_id,
#         "col_idx": 这一列在表里的局部索引（0-based）,
#         "col_name": 一个占位列名（这里没有真实列名，用 Column {i+1}）
#       }

#     保存到: save_dir/col_meta.json
#     """
#     if not hasattr(train_dataset, "table_df"):
#         raise ValueError("train_dataset 缺少 table_df，无法导出 col_meta。")

#     rows = []
#     for _, row in train_dataset.table_df.iterrows():
#         table_id = row["table_id"]
#         g_ids = row["global_col_indices"]   # Tensor([g0, g1, ...]) or list
#         col_indices = row["col_indices"]    # Tensor([c0, c1, ...])

#         # 转成 list 方便 zip
#         if hasattr(g_ids, "tolist"):
#             g_ids = g_ids.tolist()
#         if hasattr(col_indices, "tolist"):
#             col_indices = col_indices.tolist()

#         for gid, col_idx in zip(g_ids, col_indices):
#             rows.append(
#                 {
#                     "g_id": int(gid),
#                     "table_id": str(table_id),
#                     "col_idx": int(col_idx),
#                     # 这里我们没有真正的列名，用一个统一格式占位
#                     "col_name": f"Column {int(col_idx) + 1}",
#                 }
#             )

#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "col_meta.json")
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(rows, f, ensure_ascii=False, indent=2)

#     print(f"[FreeAL] col_meta 导出完毕: {save_path}")
#     return save_path

def f1_score_multilabel(true_list: np.ndarray, pred_list: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Calculate F1 scores for multilabel classification"""
    conf_mat = multilabel_confusion_matrix(np.array(true_list), np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] / conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] / conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)


def parse_tagname(tag_name: str) -> Tuple[str, int, int]:
    """Parse tag name to get shortcut name, batch size, and max length"""
    if "__" in tag_name:
        # Remove training ratio
        tag_name = tag_name.split("__")[0]
    tokens = tag_name.split("_")[-1].split("-")
    shortcut_name = "-".join(tokens[:-3])
    max_length = int(tokens[-1])
    batch_size = int(tokens[-3].replace("bs", ""))
    return shortcut_name, batch_size, max_length


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default


def flatten_list(nested_list: list) -> list:
    """Flatten a nested list"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    import os
    os.makedirs(directory, exist_ok=True)


def get_file_size(file_path: str) -> str:
    """Get human readable file size"""
    import os
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def validate_config(config, required_fields: list) -> bool:
    """Validate that config has all required fields"""
    missing_fields = []
    for field in required_fields:
        if not hasattr(config, field):
            missing_fields.append(field)
    
    if missing_fields:
        print(f"Missing required config fields: {missing_fields}")
        return False
    
    return True


def print_config_summary(config) -> None:
    """Print a summary of the configuration"""
    print("Configuration Summary:")
    print("=" * 50)
    
    # Group related fields
    data_fields = ['dataset_name', 'data_dir', 'batch_size', 'eval_batch_size', 'max_length']
    training_fields = ['num_train_epochs', 'learning_rate', 'weight_decay', 'warmup_ratio']
    model_fields = ['model_path', 'num_classes', 'method', 'loss_function']
    
    field_groups = [
        ("Data", data_fields),
        ("Training", training_fields),
        ("Model", model_fields)
    ]
    
    for group_name, fields in field_groups:
        print(f"\n{group_name}:")
        for field in fields:
            if hasattr(config, field):
                value = getattr(config, field)
                print(f"  {field}: {value}")
    
    print("=" * 50)
