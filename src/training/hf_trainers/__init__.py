"""
HuggingFace Trainer 实现模块

这个模块包含了基于 HuggingFace Trainer 的现代化训练器实现，
用于替代现有的自定义训练器系统。

主要组件：
- BaseHFTrainer: 基础 HF 训练器
- PLLHFTrainer: 支持 PLL 的 HF 训练器
- StandardHFTrainer: 标准 HF 训练器
- DoduoHFTrainer: DoDuo HF 训练器
- SelfDistillationHFTrainer: 自蒸馏 HF 训练器
- TrainerFactory: 训练器工厂函数
"""

from .base_hf_trainer import BaseHFTrainer
from .pll_hf_trainer import PLLHFTrainer, create_pll_hf_trainer
from .standard_hf_trainer import StandardHFTrainer, create_standard_hf_trainer
from .doduo_hf_trainer import DoduoHFTrainer, create_doduo_hf_trainer
from .self_distillation_hf_trainer import SelfDistillationHFTrainer, create_self_distillation_hf_trainer
from .trainer_factory import (
    create_hf_trainer,
    get_trainer_class,
    get_supported_trainer_types,
    validate_trainer_config,
    create_trainer,
    get_trainer
)

__all__ = [
    "BaseHFTrainer",
    "PLLHFTrainer", 
    "StandardHFTrainer",
    "DoduoHFTrainer",
    "SelfDistillationHFTrainer",
    "create_pll_hf_trainer",
    "create_standard_hf_trainer",
    "create_doduo_hf_trainer",
    "create_self_distillation_hf_trainer",
    "create_hf_trainer",
    "get_trainer_class",
    "get_supported_trainer_types",
    "validate_trainer_config",
    "create_trainer_from_legacy_config",
    "create_trainer",
    "get_trainer",
]
