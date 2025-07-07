"""
RGRNet Evaluation Script

Evaluate trained RGRNet models on test datasets.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --data_dir data/test
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/eval_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import create_rgrnet_for_gesture_recognition
from data.dataset import GestureDataset
from utils.metrics import GestureMetrics
from utils.visualization import plot_training_curves, visualize_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RGRNet-Eval')


def load_model(checkpoint_path: str, num_classes: int = 10, device: str = 'cpu'):
    """Load trained model from checkpoint"""
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (you might need to adjust these parameters based on your checkpoint)
    model = create_rgrnet_for_gesture_recognition(
        num_gestures=num_classes,
        input_type='rgb',
        deployment_target='mobile'  # Adjust based on your model variant
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'best_val_acc' in checkpoint:
        logger.info(f"Best validation accuracy: {checkpoint['best_val_acc']:.3f}%")
    
    return model, checkpoint


def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate model on test dataset"""
    
    # Initialize metrics
    num_classes = len(class_names) if class_names else 10
    metrics = GestureMetrics(num_classes=num_classes, class_names=class_names)
    
    # Evaluation loop
    logger.info("Running evaluation...")
    total_samples = 0
    predictions_list = []
    targets_list = []
    confidences_list = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(outputs, targets)
            
            # Store predictions for detailed analysis
            pred_classes = outputs.argmax(dim=1).cpu().numpy()
            target_classes = targets.cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()
            
            predictions_list.extend(pred_classes)
            targets_list.extend(target_classes)
            confidences_list.extend(confidences)
            
            total_samples += targets.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {total_samples} samples...")
    
    logger.info(f"Evaluation completed on {total_samples} samples")
    
    return metrics, predictions_list, targets_list, confidences_list


def main():
    parser = argparse.ArgumentParser(description='Evaluate RGRNet model')
    parser.add_argument('--checkpoint', '-c', required=True, type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names for better visualization')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions to file')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    num_classes = len(args.class_names) if args.class_names else 10
    model, checkpoint = load_model(args.checkpoint, num_classes, device)
    
    # Create test dataset and loader
    logger.info(f"Loading test data from {args.data_dir}")
    test_dataset = GestureDataset(
        data_dir=args.data_dir,
        image_size=224,  # Adjust based on your model
        augmentation=False,
        normalize=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    
    # Evaluate model
    metrics, predictions, targets, confidences = evaluate_model(
        model, test_loader, device, args.class_names
    )
    
    # Compute detailed results
    results = metrics.compute()
    
    # Log results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro Precision: {results['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {results['macro_recall']:.4f}")
    logger.info(f"Macro F1-Score: {results['macro_f1']:.4f}")
    logger.info(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Generate and save confusion matrix
    logger.info("Generating confusion matrix...")
    metrics.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )
    
    # Save classification report
    report_file = output_dir / 'classification_report.txt'
    with open(report_file, 'w') as f:
        f.write(metrics.get_classification_report())
    logger.info(f"Classification report saved to {report_file}")
    
    # Save individual predictions if requested
    if args.save_predictions:
        predictions_file = output_dir / 'predictions.json'
        prediction_data = {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'class_names': args.class_names
        }
        with open(predictions_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        logger.info(f"Predictions saved to {predictions_file}")
    
    # Analyze confidence distribution
    confidence_stats = {
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences))
    }
    
    logger.info("=== Confidence Statistics ===")
    for key, value in confidence_stats.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save confidence stats
    with open(output_dir / 'confidence_stats.json', 'w') as f:
        json.dump(confidence_stats, f, indent=2)
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()