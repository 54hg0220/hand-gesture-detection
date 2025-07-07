"""
Visualization utilities for RGRNet

Tools for plotting training curves, confusion matrices, feature maps, and model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
from tensorboard import summary
from torch.utils.tensorboard import SummaryWriter


def setup_tensorboard(log_dir: str) -> Optional[SummaryWriter]:
    """Setup TensorBoard logging"""
    try:
        writer = SummaryWriter(log_dir)
        return writer
    except Exception as e:
        print(f"Warning: Could not setup TensorBoard: {e}")
        return None


def plot_training_curves(history: Dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if 'val_acc' in history:
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Loss difference (overfitting indicator)
    if 'val_loss' in history:
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(epochs, loss_diff, 'purple', label='Val Loss - Train Loss')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix_detailed(cm: np.ndarray, class_names: List[str], 
                                 save_path: Optional[str] = None, normalize: bool = True):
    """
    Plot detailed confusion matrix with additional statistics
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_norm = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add statistics
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def visualize_feature_maps(feature_maps: torch.Tensor, save_path: Optional[str] = None,
                          max_channels: int = 16, title: str = "Feature Maps"):
    """
    Visualize feature maps from CNN layers
    
    Args:
        feature_maps: Feature maps tensor (B, C, H, W)
        save_path: Path to save the visualization
        max_channels: Maximum number of channels to visualize
        title: Title for the plot
    """
    
    # Take first batch and limit channels
    if feature_maps.dim() == 4:
        feature_maps = feature_maps[0]  # Take first batch
    
    num_channels = min(feature_maps.size(0), max_channels)
    
    # Calculate grid size
    cols = 4
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        if i < num_channels:
            # Normalize feature map for visualization
            fmap = feature_maps[i].detach().cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            
            axes[row, col].imshow(fmap, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")
    
    plt.show()


def visualize_predictions(images: torch.Tensor, predictions: torch.Tensor, 
                         targets: torch.Tensor, class_names: List[str],
                         save_path: Optional[str] = None, num_samples: int = 8):
    """
    Visualize model predictions on sample images
    
    Args:
        images: Input images tensor (B, C, H, W)
        predictions: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        class_names: List of class names
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    
    num_samples = min(num_samples, images.size(0))
    
    # Get prediction probabilities and classes
    probs = torch.softmax(predictions, dim=1)
    pred_classes = torch.argmax(predictions, dim=1)
    
    # Setup plot
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle('Model Predictions', fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        if i < num_samples:
            # Prepare image for display
            img = images[i].detach().cpu()
            if img.size(0) == 3:  # RGB
                img = img.permute(1, 2, 0)
                # Denormalize if needed (assuming ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
            else:  # Grayscale
                img = img.squeeze()
            
            # Display image
            axes[row, col].imshow(img, cmap='gray' if img.dim() == 2 else None)
            
            # Add prediction info
            pred_class = pred_classes[i].item()
            true_class = targets[i].item()
            confidence = probs[i, pred_class].item()
            
            pred_name = class_names[pred_class]
            true_name = class_names[true_class]
            
            color = 'green' if pred_class == true_class else 'red'
            title = f'Pred: {pred_name}\nTrue: {true_name}\nConf: {confidence:.3f}'
            
            axes[row, col].set_title(title, color=color, fontsize=10)
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: List[int], class_names: List[str],
                           save_path: Optional[str] = None, title: str = "Class Distribution"):
    """
    Plot distribution of classes in dataset
    
    Args:
        labels: List of class labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Title for the plot
    """
    
    # Count occurrences
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(unique_labels)), counts, color='skyblue', alpha=0.7)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Gesture Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(unique_labels)), [class_names[i] for i in unique_labels], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Add statistics
    total_samples = len(labels)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    plt.figtext(0.02, 0.02, f'Total: {total_samples}, Mean: {mean_count:.1f}, Std: {std_count:.1f}', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def create_training_report(history: Dict, metrics: Dict, output_dir: str):
    """
    Create comprehensive training report with multiple visualizations
    
    Args:
        history: Training history dictionary
        metrics: Final evaluation metrics
        output_dir: Directory to save all visualizations
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training curves
    plot_training_curves(history, save_path=output_path / 'training_curves.png', show=False)
    
    # Create summary report
    report = {
        'training_summary': {
            'total_epochs': len(history['train_loss']),
            'best_train_acc': max(history['train_acc']),
            'best_val_acc': max(history['val_acc']) if 'val_acc' in history else None,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
        },
        'evaluation_metrics': metrics,
        'files_generated': [
            'training_curves.png',
            'training_report.json'
        ]
    }
    
    # Save report
    with open(output_path / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Training report saved to {output_dir}")
    return report


def visualize_model_architecture(model: torch.nn.Module, input_shape: tuple,
                                save_path: Optional[str] = None):
    """
    Visualize model architecture (requires torchviz)
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        save_path: Path to save the visualization
    """
    
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Forward pass
        output = model(dummy_input)
        
        # Create visualization
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"Model architecture saved to {save_path}.png")
        else:
            dot.view()
            
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")
    except Exception as e:
        print(f"Could not visualize model architecture: {e}")


def plot_inference_speed_benchmark(model: torch.nn.Module, input_shapes: List[tuple],
                                  device: str = 'cpu', num_iterations: int = 100,
                                  save_path: Optional[str] = None):
    """
    Benchmark and plot inference speed for different input sizes
    
    Args:
        model: PyTorch model
        input_shapes: List of input shapes to test (C, H, W)
        device: Device to run benchmark on
        num_iterations: Number of iterations for timing
        save_path: Path to save the plot
    """
    
    model.eval()
    model.to(device)
    
    results = []
    
    for shape in input_shapes:
        # Warmup
        dummy_input = torch.randn(1, *shape).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        
        # Timing
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = model(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = 1 / avg_time
        
        results.append({
            'shape': f"{shape[1]}x{shape[2]}",
            'time_ms': avg_time * 1000,
            'fps': fps,
            'pixels': shape[1] * shape[2]
        })
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    shapes = [r['shape'] for r in results]
    times = [r['time_ms'] for r in results]
    fps_values = [r['fps'] for r in results]
    
    # Inference time plot
    ax1.bar(shapes, times, color='skyblue', alpha=0.7)
    ax1.set_title('Inference Time vs Input Size')
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # FPS plot
    ax2.bar(shapes, fps_values, color='lightcoral', alpha=0.7)
    ax2.set_title('FPS vs Input Size')
    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('FPS')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark results saved to {save_path}")
    
    plt.show()
    
    return results
