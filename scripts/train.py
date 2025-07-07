"""
RGRNet Training Script

Train RGRNet models with comprehensive configuration support.

Usage:
    python scripts/train.py --config configs/mobile_config.yaml
    python scripts/train.py --config configs/server_config.yaml --resume checkpoints/latest.pth
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import ExperimentConfig
from models import create_rgrnet_for_gesture_recognition
from training.trainer import GestureTrainer
from data.dataset import GestureDataset
from utils.visualization import setup_tensorboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RGRNet-Train')


def create_data_loaders(config: ExperimentConfig):
    """Create training and validation data loaders"""
    
    # Training dataset
    train_dataset = GestureDataset(
        data_dir=os.path.join(config.data.dataset_path, 'train'),
        image_size=config.model.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    # Validation dataset
    val_dataset = GestureDataset(
        data_dir=os.path.join(config.data.dataset_path, 'val'),
        image_size=config.model.image_size,
        augmentation=False,  # No augmentation for validation
        normalize=config.data.normalize
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train RGRNet model')
    parser.add_argument('--config', '-c', required=True, type=str,
                       help='Path to configuration file')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ExperimentConfig.from_yaml(args.config)
    
    # Override paths if provided
    if args.data_dir:
        config.data.dataset_path = args.data_dir
    if args.output_dir:
        config.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        config.log_dir = os.path.join(args.output_dir, 'logs')
    
    # Create output directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_yaml(os.path.join(config.log_dir, 'config.yaml'))
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {config.model.variant} model...")
    model = create_rgrnet_for_gesture_recognition(
        num_gestures=config.model.num_classes,
        input_type='rgb' if config.model.input_channels == 3 else 'grayscale',
        deployment_target='server' if config.model.variant == 'standard' else 'mobile'
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Setup TensorBoard
    tb_writer = setup_tensorboard(config.log_dir)
    
    # Create trainer
    trainer = GestureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        device=device,
        mixed_precision=config.training.mixed_precision,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()