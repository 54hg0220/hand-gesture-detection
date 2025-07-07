"""
RGRNet Dataset Implementation

Custom dataset classes for gesture recognition with support for various data formats.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union, Callable
import json
import random


class GestureDataset(Dataset):
    """
    Custom dataset for gesture recognition
    
    Expected directory structure:
    data_dir/
    ├── class_0/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_1/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_size: int = 224,
                 augmentation: bool = True,
                 normalize: bool = True,
                 transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None):
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.normalize = normalize
        self.transform = transform
        
        # Load data
        self.samples, self.labels, self.class_names = self._load_data(class_names)
        self.num_classes = len(self.class_names)
        
        # Setup transforms
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        print(f"Loaded {len(self.samples)} samples from {self.num_classes} classes")
    
    def _load_data(self, class_names: Optional[List[str]] = None) -> Tuple[List[str], List[int], List[str]]:
        """Load image paths and labels"""
        
        samples = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        if class_names is None:
            class_names = [d.name for d in class_dirs]
        
        # Supported image extensions
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_idx, class_dir in enumerate(class_dirs):
            if class_dir.name not in class_names:
                continue
            
            class_label = class_names.index(class_dir.name)
            
            # Get all images in class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    samples.append(str(img_path))
                    labels.append(class_label)
        
        return samples, labels, class_names
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms"""
        
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Augmentation
        if self.augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = torch.zeros(3, self.image_size, self.image_size)
            return blank_image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weights (inverse frequency)
        weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights)
    
    def get_data_statistics(self) -> dict:
        """Get dataset statistics"""
        
        class_counts = np.bincount(self.labels, minlength=self.num_classes)
        
        stats = {
            'total_samples': len(self.samples),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_counts': class_counts.tolist(),
            'class_distribution': (class_counts / len(self.samples)).tolist(),
            'min_class_samples': int(np.min(class_counts)),
            'max_class_samples': int(np.max(class_counts)),
            'mean_class_samples': float(np.mean(class_counts)),
            'std_class_samples': float(np.std(class_counts))
        }
        
        return stats


class VideoGestureDataset(Dataset):
    """
    Dataset for video-based gesture recognition
    
    Expected structure:
    data_dir/
    ├── class_0/
    │   ├── video1.mp4
    │   ├── video2.avi
    │   └── ...
    ├── class_1/
    │   └── ...
    """
    
    def __init__(self,
                 data_dir: str,
                 sequence_length: int = 16,
                 image_size: int = 224,
                 augmentation: bool = True,
                 normalize: bool = True,
                 class_names: Optional[List[str]] = None):
        
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augmentation = augmentation
        self.normalize = normalize
        
        # Load data
        self.samples, self.labels, self.class_names = self._load_video_data(class_names)
        self.num_classes = len(self.class_names)
        
        # Setup transforms
        self.transform = self._get_video_transform()
        
        print(f"Loaded {len(self.samples)} video samples from {self.num_classes} classes")
    
    def _load_video_data(self, class_names: Optional[List[str]] = None) -> Tuple[List[str], List[int], List[str]]:
        """Load video paths and labels"""
        
        samples = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        if class_names is None:
            class_names = [d.name for d in class_dirs]
        
        # Supported video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for class_dir in class_dirs:
            if class_dir.name not in class_names:
                continue
            
            class_label = class_names.index(class_dir.name)
            
            # Get all videos in class directory
            for video_path in class_dir.iterdir():
                if video_path.suffix.lower() in video_extensions:
                    samples.append(str(video_path))
                    labels.append(class_label)
        
        return samples, labels, class_names
    
    def _get_video_transform(self) -> transforms.Compose:
        """Get video transforms"""
        
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        if self.augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transform_list)
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample frames to create fixed-length sequence"""
        
        if len(frames) <= self.sequence_length:
            # Repeat frames if video is too short
            while len(frames) < self.sequence_length:
                frames.extend(frames)
            return frames[:self.sequence_length]
        else:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            return [frames[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Extract and sample frames
            frames = self._extract_frames(video_path)
            frames = self._sample_frames(frames)
            
            # Transform frames
            transformed_frames = []
            for frame in frames:
                if self.transform:
                    frame = self.transform(frame)
                transformed_frames.append(frame)
            
            # Stack frames into tensor (T, C, H, W)
            video_tensor = torch.stack(transformed_frames)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return blank video if loading fails
            blank_video = torch.zeros(self.sequence_length, 3, self.image_size, self.image_size)
            return blank_video, label


def create_data_loaders(data_dir: str, 
                       batch_size: int = 32,
                       image_size: int = 224,
                       num_workers: int = 4,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Root directory containing class subdirectories
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        num_workers: Number of worker processes for data loading
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create full dataset
    full_dataset = GestureDataset(
        data_dir=data_dir,
        image_size=image_size,
        augmentation=False,  # We'll handle augmentation separately
        normalize=True
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create augmented training dataset
    train_dataset.dataset.augmentation = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing RGRNet dataset loading...")
    
    # Test image dataset
    try:
        dataset = GestureDataset(
            data_dir="./data/gestures",
            image_size=224,
            augmentation=True
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Class names: {dataset.class_names}")
        
        # Test data loading
        sample_image, sample_label = dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample label: {sample_label}")
        
        # Test data statistics
        stats = dataset.get_data_statistics()
        print(f"Dataset statistics: {stats}")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("Make sure to create a ./data/gestures directory with class subdirectories")