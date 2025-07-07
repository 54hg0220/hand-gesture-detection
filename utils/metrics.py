"""
Evaluation metrics for gesture recognition
"""

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class GestureMetrics:
    """Comprehensive metrics calculator for gesture recognition"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Gesture_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch"""
        pred_classes = predictions.argmax(dim=1).cpu().numpy()
        target_classes = targets.cpu().numpy()
        
        self.predictions.extend(pred_classes)
        self.targets.extend(target_classes)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
            metrics[f'{class_name}_support'] = support[i]
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.targets, self.predictions)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = True):
        """Plot confusion matrix"""
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.targets, self.predictions,
            target_names=self.class_names,
            zero_division=0
        )