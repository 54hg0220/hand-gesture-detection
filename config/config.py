"""
Configuration management for RGRNet training
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = 'rgrnet'
    variant: str = 'standard'  # standard, lightweight, efficient
    input_channels: int = 3
    num_classes: int = 10
    image_size: int = 224
    use_lightweight: bool = False
    dropout_rate: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'  # adamw, adam, sgd
    scheduler: str = 'cosine'  # cosine, step, plateau, none
    mixed_precision: bool = True
    early_stopping_patience: int = 15
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = './data'
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augmentation: bool = True
    normalize: bool = True
    resize_method: str = 'bilinear'  # bilinear, bicubic, nearest


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    output_dir: str = './outputs'
    
    # Experiment metadata
    experiment_name: str = 'rgrnet_gesture_recognition'
    description: str = 'RGRNet training for gesture recognition'
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'data']}
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'tags': self.tags
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)