"""
Factory functions for creating RGRNet models
"""

import torch
import torch.nn as nn
from typing import Optional

from .rgrnet import RGRNet


def create_rgrnet(
    input_channels: int = 3,
    num_classes: int = 10,
    image_size: int = 224,
    model_variant: str = 'standard',
    pretrained: bool = False,
    **kwargs
) -> RGRNet:
    """
    Factory function to create RGRNet models with different configurations
    
    Args:
        input_channels (int): Number of input channels
        num_classes (int): Number of output classes
        image_size (int): Input image size
        model_variant (str): Model variant ('standard', 'lightweight', 'efficient')
        pretrained (bool): Whether to load pretrained weights
        **kwargs: Additional arguments passed to RGRNet
        
    Returns:
        RGRNet: Configured RGRNet model
    """
    
    if model_variant == 'lightweight':
        # Lightweight variant with depthwise separable convolutions
        model_config = {
            'use_lightweight': True,
            'dropout_rate': 0.3,
            **kwargs
        }
    elif model_variant == 'efficient':
        # Efficient variant optimized for mobile deployment
        model_config = {
            'use_lightweight': True,
            'dropout_rate': 0.2,
            **kwargs
        }
    else:  # standard
        # Standard variant with full convolutions
        model_config = {
            'use_lightweight': False,
            'dropout_rate': 0.5,
            **kwargs
        }
    
    model = RGRNet(
        input_channels=input_channels,
        num_classes=num_classes,
        image_size=image_size,
        **model_config
    )
    
    if pretrained:
        # TODO: Load pretrained weights when available
        print("Warning: Pretrained weights not yet available for RGRNet")
    
    return model


def create_rgrnet_for_gesture_recognition(
    num_gestures: int = 10,
    input_type: str = 'rgb',
    deployment_target: str = 'server'
) -> RGRNet:
    """
    Create RGRNet specifically configured for gesture recognition
    
    Args:
        num_gestures (int): Number of gesture classes
        input_type (str): 'rgb', 'grayscale', or 'depth'
        deployment_target (str): 'server', 'mobile', or 'edge'
        
    Returns:
        RGRNet: Configured model for gesture recognition
    """
    
    # Determine input channels based on input type
    if input_type == 'rgb':
        input_channels = 3
    elif input_type == 'grayscale':
        input_channels = 1
    elif input_type == 'depth':
        input_channels = 1
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    
    # Choose model variant based on deployment target
    if deployment_target == 'mobile':
        variant = 'lightweight'
        image_size = 224
    elif deployment_target == 'edge':
        variant = 'efficient'
        image_size = 128
    else:  # server
        variant = 'standard'
        image_size = 224
    
    return create_rgrnet(
        input_channels=input_channels,
        num_classes=num_gestures,
        image_size=image_size,
        model_variant=variant
    )
