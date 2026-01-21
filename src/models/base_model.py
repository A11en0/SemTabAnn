"""
Base model interface for FIESTA system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base interface for all models in FIESTA system."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_embeddings(self, *args, **kwargs) -> torch.Tensor:
        """Get model embeddings."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        self.load_state_dict(torch.load(path, map_location='cpu'))


class BaseTrainer(ABC):
    """Base interface for all trainers in FIESTA system."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate the model."""
        pass


class BaseDataset(ABC):
    """Base interface for all datasets in FIESTA system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        pass


class BaseEvaluator(ABC):
    """Base interface for all evaluators in FIESTA system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate predictions against targets."""
        pass
