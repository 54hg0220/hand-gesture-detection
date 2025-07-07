"""
RGRNet Inference Script

Run inference on images or videos using trained RGRNet models.

Usage:
    python scripts/infer.py --checkpoint checkpoints/best.pth --input image.jpg
    python scripts/infer.py --checkpoint checkpoints/best.pth --input video.mp4 --output results.mp4
    python scripts/infer.py --checkpoint checkpoints/best.pth --input webcam --realtime
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import create_rgrnet_for_gesture_recognition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RGRNet-Infer')


class GestureInference:
    """Real-time gesture inference class"""
    
    def __init__(self, checkpoint_path: str, class_names=None, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names or [f'Gesture_{i}' for i in range(10)]
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Inference ready on {self.device}")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Create model
        model = create_rgrnet_for_gesture_recognition(
            num_gestures=len(self.class_names),
            input_type='rgb',
            deployment_target='mobile'
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if from OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return input_tensor
    
    def predict(self, image, return_confidence=True):
        """Run inference on a single image"""
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        if return_confidence:
            return predicted_class, confidence
        return predicted_class
    
    def predict_with_timing(self, image):
        """Run inference with timing information"""
        start_time = time.time()
        predicted_class, confidence = self.predict(image)
        inference_time = time.time() - start_time
        
        return predicted_class, confidence, inference_time


def process_image(inference_engine, image_path, output_path=None):
    """Process a single image"""
    logger.info(f"Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Run inference
    predicted_class, confidence, inference_time = inference_engine.predict_with_timing(image)
    
    # Get class name
    class_name = inference_engine.class_names[predicted_class]
    
    # Log results
    logger.info(f"Prediction: {class_name} (Class {predicted_class})")
    logger.info(f"Confidence: {confidence:.3f}")
    logger.info(f"Inference time: {inference_time*1000:.1f}ms")
    
    # Visualize result
    if output_path:
        # Draw prediction on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{class_name}: {confidence:.3f}"
        cv2.putText(image, text, (10, 30), font, 1, (0, 255, 0), 2)
        
        # Save result
        cv2.imwrite(output_path, image)
        logger.info(f"Result saved to: {output_path}")


def process_video(inference_engine, video_path, output_path=None, show_realtime=False):
    """Process video file or webcam"""
    
    # Open video source
    if video_path.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        logger.info("Starting webcam inference...")
    else:
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Processing video: {video_path}")
    
    if not cap.isOpened():
        logger.error("Could not open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing variables
    frame_count = 0
    total_inference_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            predicted_class, confidence, inference_time = inference_engine.predict_with_timing(frame)
            class_name = inference_engine.class_names[predicted_class]
            
            total_inference_time += inference_time
            frame_count += 1
            
            # Draw prediction on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{class_name}: {confidence:.3f}"
            fps_text = f"FPS: {1/inference_time:.1f}"
            
            cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, fps_text, (10, 70), font, 0.7, (255, 0, 0), 2)
            
            # Show frame if realtime
            if show_realtime:
                cv2.imshow('RGRNet Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame if output path provided
            if output_path:
                out.write(frame)
            
            # Log progress
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_inference_time
                logger.info(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
        # Final statistics
        if frame_count > 0:
            avg_inference_time = total_inference_time / frame_count
            avg_fps = 1 / avg_inference_time
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average inference time: {avg_inference_time*1000:.1f}ms")
            logger.info(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Run RGRNet inference')
    parser.add_argument('--checkpoint', '-c', required=True, type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--input', '-i', required=True, type=str,
                       help='Input path (image, video, or "webcam")')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--realtime', action='store_true',
                       help='Show real-time visualization for video/webcam')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode (measure performance)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = GestureInference(
        checkpoint_path=args.checkpoint,
        class_names=args.class_names,
        device=args.device
    )
    
    # Determine input type and process accordingly
    input_path = args.input.lower()
    
    if input_path == 'webcam' or input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video processing
        process_video(
            inference_engine, 
            args.input, 
            args.output, 
            args.realtime
        )
    elif input_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Image processing
        process_image(
            inference_engine, 
            args.input, 
            args.output
        )
    else:
        logger.error(f"Unsupported input format: {args.input}")
        logger.info("Supported formats: images (.jpg, .png, etc.), videos (.mp4, .avi, etc.), or 'webcam'")


if __name__ == '__main__':
    main()
