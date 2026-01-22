"""
HuggingFace Trainer Implementation Module

This module contains modern trainer implementations based on HuggingFace Trainer,
designed to replace the existing custom trainer system.

Main Components:
- BaseHFTrainer: Base HF trainer
- PLLHFTrainer: PLL-enabled HF trainer
- StandardHFTrainer: Standard HF trainer
- DoduoHFTrainer: DoDuo HF trainer
- SelfDistillationHFTrainer: Self-distillation HF trainer
- TrainerFactory: Trainer factory functions
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
