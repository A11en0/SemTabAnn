"""
Standard HuggingFace Trainer 实现

这个模块包含了基于 HuggingFace Trainer 的标准训练器实现，
使用交叉熵损失进行监督学习。
"""

from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from transformers import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base_hf_trainer import BaseHFTrainer
from ..hf_utils.data_collator import PLLDataCollator, PLLSingleDataCollator, create_data_collator
from ..hf_utils.metrics_computer import MetricsComputer, compute_metrics_for_hf_trainer
from ..adapters.config_adapter import HFTrainingConfig, create_hf_training_args


class StandardHFTrainer(BaseHFTrainer):
    """
    Standard HuggingFace Trainer 类
    
    这个类实现了基于 HuggingFace Trainer 的标准训练器，
    使用交叉熵损失进行监督学习，完全保持现有 StandardTrainer 的功能。
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
        初始化 Standard HF Trainer
        
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
        
        # 初始化标准损失函数
        self.loss_fn = nn.CrossEntropyLoss()
    
    def setup_pll_components(self) -> None:
        """
        设置标准训练器组件
        
        重写父类方法以添加标准训练特定的初始化。
        """
        # 调用父类方法
        super().setup_pll_components()
        
        # 标准训练器不需要 PLL 损失计算器
        self.pll_loss_computer = None
    
    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Any], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        计算标准交叉熵损失
        
        这个方法完全保持现有 StandardTrainer 的损失计算逻辑。
        
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
        
        # 获取真实标签（不是候选标签）
        true_labels = inputs.get("true_label", inputs["label"])
        
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
        
        完全保持现有 StandardTrainer 的逻辑。
        
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
        
        重写父类方法以添加标准训练特定的预测逻辑。
        
        Args:
            test_dataset: 测试数据集
            ignore_keys: 忽略的键
            metric_key_prefix: 指标键前缀
            
        Returns:
            预测结果字典
        """
        # 调用父类方法
        predictions = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        
        # 添加标准训练特定的预测逻辑
        # 例如：置信度分析等
        
        return predictions


def create_standard_hf_trainer(
    model: nn.Module,
    config: HFTrainingConfig,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    **kwargs
) -> StandardHFTrainer:
    """
    创建 Standard HF Trainer
    
    Args:
        model: 要训练的模型
        config: HF 训练配置
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        tokenizer: 分词器
        **kwargs: 其他参数
        
    Returns:
        StandardHFTrainer 实例
    """
    # 确保配置是标准配置
    if config.method == "pll" or config.training_mode == "pll":
        raise ValueError("StandardHFTrainer requires non-PLL configuration")
    
    # 创建训练参数
    training_args = create_hf_training_args(config)
    
    # 创建训练器
    trainer = StandardHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        **kwargs
    )
    
    return trainer
