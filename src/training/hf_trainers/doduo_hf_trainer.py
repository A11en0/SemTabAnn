"""
Doduo HuggingFace Trainer 实现

这个模块包含了基于 HuggingFace Trainer 的 Doduo 训练器实现，
完全保持现有 DoduoTrainer 的功能和逻辑。
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from transformers import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base_hf_trainer import BaseHFTrainer
from ..hf_utils.data_collator import PLLDataCollator, PLLSingleDataCollator, create_data_collator
from ..hf_utils.metrics_computer import MetricsComputer, compute_metrics_for_hf_trainer
from ..adapters.config_adapter import HFTrainingConfig, create_hf_training_args


class DoduoHFTrainer(BaseHFTrainer):
    """
    Doduo HuggingFace Trainer 类
    
    这个类实现了基于 HuggingFace Trainer 的 Doduo 训练器，
    完全保持现有 DoduoTrainer 的功能和逻辑。
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: Union[TrainingArguments, HFTrainingConfig],
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[callable] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Optional[tuple] = None,
        preprocess_logits_for_metrics: Optional[callable] = None,
        **kwargs
    ):
        """
        初始化 Doduo HF Trainer
        
        Args:
            model: 要训练的模型
            args: 训练参数（TrainingArguments 或 HFTrainingConfig）
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            tokenizer: 分词器
            data_collator: 数据整理器
            compute_metrics: 指标计算函数
            callbacks: 回调函数列表
            optimizers: 优化器元组
            preprocess_logits_for_metrics: 预处理 logits 的函数
            **kwargs: 其他参数
        """
        # 确保使用标准数据整理器
        if data_collator is None:
            if tokenizer is None:
                raise ValueError("tokenizer is required when data_collator is not provided")
            
            # 根据配置选择数据整理器
            is_single_column = getattr(args, 'is_single_column', False)
            data_collator = create_data_collator(tokenizer, is_single_column=is_single_column)
        
        # 设置默认的指标计算函数
        if compute_metrics is None:
            compute_metrics = compute_metrics_for_hf_trainer
        
        # 调用父类初始化
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Doduo 特定属性
        self.model_type = getattr(args, 'model_type', 'viznet')
        self.coltype_mlb = None
        self.colrel_mlb = None
        
        # 为 wikitable 模型加载 MLB
        if self.model_type == 'wikitable':
            self._load_mlb()
        
        # 初始化标准损失函数
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _load_mlb(self):
        """
        为 wikitable 模型加载 MultiLabelBinarizer
        
        完全保持现有 DoduoTrainer 的逻辑。
        """
        try:
            data_dir = getattr(self.hf_config, 'data_dir', './data') if self.hf_config else './data'
            with open(os.path.join(data_dir, "turl_coltype_mlb.pickle"), "rb") as fin:
                self.coltype_mlb = pickle.load(fin)
            with open(os.path.join(data_dir, "turl_colrel_mlb.pickle"), "rb") as fin:
                self.colrel_mlb = pickle.load(fin)
        except FileNotFoundError:
            print("Warning: MLB files not found. Using default labels.")
    
    def setup_pll_components(self) -> None:
        """
        设置 Doduo 训练器组件
        
        重写父类方法以添加 Doduo 特定的初始化。
        """
        # 调用父类方法
        super().setup_pll_components()
        
        # Doduo 训练器不需要 PLL 损失计算器
        self.pll_loss_computer = None
    
    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Any], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        计算标准交叉熵损失
        
        这个方法完全保持现有 DoduoTrainer 的损失计算逻辑。
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回模型输出
            
        Returns:
            损失张量或 (损失, 输出) 元组
        """
        # 获取模型输出
        outputs = model(**inputs)
        
        # 提取 logits
        if isinstance(outputs, tuple):
            logits = outputs[0]  # 第一个元素是 logits
        else:
            logits = outputs.logits
        
        # 处理多列情况（表格级别）
        if len(logits.shape) == 3:  # (batch_size, seq_len, num_classes)
            # 找到 CLS token 位置
            cls_indexes = torch.nonzero(
                inputs["input_ids"] == self.processing_class.cls_token_id
            )
            
            # 在 CLS 位置提取 logits
            filtered_logits = torch.zeros(
                cls_indexes.shape[0], logits.shape[2]
            ).to(self.device)
            
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                filtered_logits[n] = logits[i, j, :]
            
            logits = filtered_logits
        
        # 获取真实标签
        true_labels = inputs["label"]
        
        # 计算损失
        loss = self.loss_fn(logits, true_labels)
        
        # 返回结果
        if return_outputs:
            return (loss, outputs)
        else:
            return loss
    
    def _extract_logits_from_model_output(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        从模型输出中提取 logits
        
        完全保持现有 DoduoTrainer 的逻辑。
        
        Args:
            batch: 批次数据
            
        Returns:
            提取的 logits
        """
        # 获取模型输出
        model_output = self.model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        
        if isinstance(model_output, tuple):
            logits = model_output[0]  # 第一个元素是 logits
        else:
            logits = model_output.logits
        
        # 处理多列情况（表格级别）
        if len(logits.shape) == 3:  # (batch_size, seq_len, num_classes)
            # 找到 CLS token 位置
            cls_indexes = torch.nonzero(
                batch["input_ids"] == self.processing_class.cls_token_id
            )
            
            # 在 CLS 位置提取 logits
            filtered_logits = torch.zeros(
                cls_indexes.shape[0], logits.shape[2]
            ).to(self.device)
            
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                filtered_logits[n] = logits[i, j, :]
            
            logits = filtered_logits
        
        return logits
    
    def _get_label_names(self, predictions: List[int]) -> List[str]:
        """
        将预测转换为标签名称
        
        完全保持现有 DoduoTrainer 的逻辑。
        
        Args:
            predictions: 预测列表
            
        Returns:
            标签名称列表
        """
        if self.model_type == "viznet":
            # Sato 列类型
            sato_coltypes = [
                "address", "affiliate", "affiliation", "age", "album", "area", "artist",
                "birthDate", "birthPlace", "brand", "capacity", "category", "city",
                "class", "classification", "club", "code", "collection", "command",
                "company", "component", "continent", "country", "county", "creator",
                "credit", "currency", "day", "depth", "description", "director",
                "duration", "education", "elevation", "family", "fileSize", "format",
                "gender", "genre", "grades", "isbn", "industry", "jockey", "language",
                "location", "manufacturer", "name", "nationality", "notes", "operator",
                "order", "organisation", "origin", "owner", "person", "plays", "position",
                "product", "publisher", "range", "rank", "ranking", "region", "religion",
                "requirement", "result", "sales", "service", "sex", "species", "state",
                "status", "symbol", "team", "teamName", "type", "weight", "year"
            ]
            return [sato_coltypes[p] for p in predictions]
        elif self.model_type == "wikitable" and self.coltype_mlb is not None:
            return [self.coltype_mlb.classes_[p] for p in predictions]
        else:
            return [str(p) for p in predictions]
    
    def evaluate(
        self, 
        eval_dataset: Optional[Any] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        评估模型 - 使用 HuggingFace Trainer 的标准评估机制
        
        Args:
            eval_dataset: 评估数据集
            ignore_keys: 忽略的键
            metric_key_prefix: 指标键前缀

        Returns:
            metrics
        """
        return super(BaseHFTrainer, self).evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def predict(
        self, 
        test_dataset: Any, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        预测
        
        重写父类方法以添加 Doduo 特定的预测逻辑。
        
        Args:
            test_dataset: 测试数据集
            ignore_keys: 忽略的键
            metric_key_prefix: 指标键前缀
            
        Returns:
            预测结果字典
        """
        # 调用父类方法
        predictions = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        # 添加 Doduo 特定的预测逻辑
        # 例如：标签名称转换等
        if "predictions" in predictions:
            predictions["prediction_names"] = self._get_label_names(predictions["predictions"])
        
        return predictions
    
    def predict_dataframe(self, df) -> Dict[str, Any]:
        """
        使用 Doduo 方法为 DataFrame 预测列类型
        
        完全保持现有 DoduoTrainer 的逻辑。
        
        Args:
            df: 要预测的 DataFrame
            
        Returns:
            预测结果字典
        """
        try:
            from ..dataset import DFColTypeTablewiseDataset
            from ..util import collate_fn
        except ImportError:
            # 如果无法导入，返回错误信息
            return {
                "error": "Required modules (dataset, util) not available for DataFrame prediction",
                "predictions": [],
                "prediction_names": [],
                "logits": []
            }
        
        # 创建数据集
        input_dataset = DFColTypeTablewiseDataset(df, self.processing_class)
        input_dataloader = torch.utils.data.DataLoader(
            input_dataset,
            batch_size=getattr(self.hf_config, 'per_device_eval_batch_size', 8),
            collate_fn=collate_fn
        )
        
        # 获取预测
        batch = next(iter(input_dataloader))
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 前向传播
        logits, = self.model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        
        # 在 CLS 位置提取 logits
        cls_indexes = torch.nonzero(batch["input_ids"] == self.processing_class.cls_token_id)
        filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(self.device)
        
        for n in range(cls_indexes.shape[0]):
            i, j = cls_indexes[n]
            filtered_logits[n] = logits[i, j, :]
        
        # 获取预测
        predictions = filtered_logits.argmax(1).cpu().detach().numpy()
        prediction_names = self._get_label_names(predictions.tolist())
        
        return {
            "predictions": predictions,
            "prediction_names": prediction_names,
            "logits": filtered_logits.cpu().detach().numpy()
        }


def create_doduo_hf_trainer(
    model: nn.Module,
    config: HFTrainingConfig,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    **kwargs
) -> DoduoHFTrainer:
    """
    创建 Doduo HF Trainer
    
    Args:
        model: 要训练的模型
        config: HF 训练配置
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        tokenizer: 分词器
        **kwargs: 其他参数
        
    Returns:
        DoduoHFTrainer 实例
    """
    # 创建训练参数
    training_args = create_hf_training_args(config)
    
    # 创建训练器
    trainer = DoduoHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        **kwargs
    )
    
    return trainer
