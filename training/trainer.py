"""
Training utilities for RGRNet models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Callable
import time
import logging
from pathlib import Path
import json


class GestureTrainer:
    """
    Professional trainer class for RGRNet gesture recognition models
    
    Features:
    - Multiple optimizers and schedulers support
    - Automatic mixed precision training
    - Model checkpointing and resuming
    - Comprehensive logging and metrics tracking
    - Early stopping and learning rate scheduling
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: str = 'adamw',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler: str = 'cosine',
                 device: Optional[str] = None,
                 mixed_precision: bool = True,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate, weight_decay)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler(scheduler)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Setup mixed precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _create_optimizer(self, optimizer: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if optimizer.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    def _create_scheduler(self, scheduler: str) -> Optional[_LRScheduler]:
        """Create learning rate scheduler"""
        if scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
        elif scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GestureTrainer')
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
        
        # Save epoch checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.current_epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, epochs: int, early_stopping_patience: int = 15):
        """Main training loop"""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(current_lr)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            self.logger.info(
                f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.2e}'
            )
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
