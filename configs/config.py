"""
Configuration management for Naive PLL
"""
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class NaivePLLConfig:
    """Configuration class for Naive PLL baseline"""
    # Training parameters
    batch_size: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    
    # Method and training mode
    method: str = "pll"  # "pll", "standard", "doduo", "doduo_single", "self_distillation"
    training_mode: str = "pll"  # "pll" for partial label learning, "standard" for cross-entropy
    loss_function: str = "naive_pll"  # uniform_pll, max_pll, avg_pll, weighted_pll, cross_entropy, standard
    
    # Random noise bootstrapping configuration
    use_random_noise: bool = False
    noise_level: float = 0.3
    num_candidates: int = 5  # Number of candidate labels to generate (only used in PLL mode)

    # PLL specific configuration
    pll_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Self-distillation specific parameters
    warmup_epochs: int = 5  # Number of warmup epochs before sample selection
    tau_loss: float = 0.5  # Threshold for clean sample selection (small loss principle)
    tau_conf: float = 0.8  # Threshold for high confidence sample selection
    sharpen: float = 0.5  # Temperature for label sharpening
    temp_u: float = 1.0  # Temperature for consistency loss
    is_mu: bool = True  # Enable mix-up augmentation
    is_cr: bool = True  # Enable consistency regularization
    
    # Single column mode configuration
    single_column_mode: bool = False  # Enable single column prediction mode
    use_colwise_dataset: bool = False  # Use column-wise dataset instead of table-wise
    model_path: Optional[str] = None  # Path to pre-trained model for single column mode
    
    # LLM bootstrapping configuration
    enable_llm_bootstrapping: bool = False  # Enable LLM bootstrapping for candidate generation
    
    # Model parameters
    max_length: int = 128
    num_classes: int = 78  # Default for SATO dataset
    model_type: str = "viznet"  # Model type for doduo (viznet, wikitable)
    model_name: str = "bert-base-uncased"  # Model name for transformers
    
    # Inference configuration
    inference_mode: bool = False  # Set to true to skip training and run inference only
    checkpoint_path: Optional[str] = None  # Path to checkpoint file for inference
    
    # Data parameters
    dataset_name: str = "sato0"  # sato0, sato1, ..., sato4, msato0, etc.
    data_dir: str = "../data/test_small/sato_cv"
    multicol_only: bool = False
    limit_test: int = -1

    # Evaluation parameters
    eval_batch_size: int = 32
    save_predictions: bool = True
    save_model: bool = True
    save_config: bool = True
    output_dir: str = None  # Will be auto-generated based on method and parameters
    
    # Random seed
    random_seed: int = 42
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", or "cuda"
    num_workers: int = 0
    
    # Mixed precision training
    fp16: bool = False  # Enable half precision training
    bf16: bool = False  # Enable bfloat16 precision training
    use_mixed_precision: bool = False  # Enable mixed precision training
    
    # DataLoader optimization
    dataloader_num_workers: int = 0  # Number of workers for data loading
    dataloader_pin_memory: bool = False  # Pin memory for faster GPU transfer
    dataloader_persistent_workers: bool = False  # Keep workers alive between epochs
    dataloader_prefetch_factor: int = 2  # Prefetch factor for data loading
    
    # SwanLab monitoring
    use_swanlab: bool = True  # Enable SwanLab monitoring
    project_name: str = "naive-pll-project"
    experiment_name: Optional[str] = None  # Custom experiment name for SwanLab
    
    # Evaluation configuration
    evaluation: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Inference configuration
    inference_mode: bool = False  # Enable inference mode (skip training)
    checkpoint_path: Optional[str] = None  # Path to model checkpoint for inference
    
    # LLM Bootstrapper configuration
    llm_bootstrapper: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NaivePLLConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract naive_pll section if exists
        if 'naive_pll' in config_dict:
            config_dict = config_dict['naive_pll']
        
        # Convert string learning_rate to float if needed
        if 'learning_rate' in config_dict and isinstance(config_dict['learning_rate'], str):
            config_dict['learning_rate'] = float(config_dict['learning_rate'])
        
        # Ensure pll_config is properly handled
        if 'pll_config' not in config_dict:
            config_dict['pll_config'] = {}
        
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'naive_pll': {
                'method': self.method,
                'num_train_epochs': self.num_train_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_ratio': self.warmup_ratio,
                'training_mode': self.training_mode,
                'loss_function': self.loss_function,                
                'noise_level': self.noise_level,
                'use_random_noise': self.use_random_noise,
                'num_candidates': self.num_candidates,
                'max_length': self.max_length,
                'num_classes': self.num_classes,
                'dataset_name': self.dataset_name,
                'data_dir': self.data_dir,
                'multicol_only': self.multicol_only,
                'eval_batch_size': self.eval_batch_size,
                'save_predictions': self.save_predictions,
                'output_dir': self.output_dir,
                'random_seed': self.random_seed,
                'limit_test': self.limit_test,
                'use_swanlab': self.use_swanlab,
                'project_name': self.project_name,
                'inference_mode': self.inference_mode,
                'checkpoint_path': self.checkpoint_path,
                'pll_config': self.pll_config,
                'llm_bootstrapper': self.llm_bootstrapper
            }
        }
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        import os
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
