"""
RGRNet (Rapid Gesture Recognition Network) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .components import (
    SELayer, TransformerLayer, SpatialPyramidPooling, 
    ConvBlock, DepthwiseSeparableConv
)


class RGRNet(nn.Module):
    """
    Rapid Gesture Recognition Network (RGRNet)
    
    A CNN-based architecture with attention mechanisms designed for real-time
    gesture recognition tasks. The network consists of three main blocks:
    1. Feature Extraction Block with attention mechanisms
    2. Multi-scale Feature Fusion Block  
    3. Classification Head
    
    Args:
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        num_classes (int): Number of gesture classes to recognize
        image_size (int): Input image size (assumes square images)
        use_lightweight (bool): Whether to use lightweight depthwise separable convolutions
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 10,
                 image_size: int = 224,
                 use_lightweight: bool = False,
                 dropout_rate: float = 0.5):
        super(RGRNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.use_lightweight = use_lightweight
        self.dropout_rate = dropout_rate
        
        # Initialize network components
        self._build_feature_extraction_block()
        self._build_attention_layers()
        self._build_fusion_block()
        self._build_classification_head()
        
        # Skip connection storage
        self.skip_connections: Dict[str, torch.Tensor] = {}
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_feature_extraction_block(self):
        """Build the feature extraction block (Block 1)"""
        conv_layer = DepthwiseSeparableConv if self.use_lightweight else ConvBlock
        
        # Initial feature extraction layers
        self.stem = nn.Sequential(
            ConvBlock(self.input_channels, 32, kernel_size=3, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
        )
        
        # Progressive feature extraction with increasing channels
        self.conv_block_1 = conv_layer(64, 128, kernel_size=1, padding=0)
        self.conv_block_2 = conv_layer(128, 128, kernel_size=3)
        self.conv_block_3 = conv_layer(128, 256, kernel_size=3, stride=2)
        self.conv_block_4 = conv_layer(256, 256, kernel_size=3)
        self.conv_block_5 = conv_layer(256, 512, kernel_size=3, stride=2)
        self.conv_block_6 = conv_layer(512, 512, kernel_size=3)
        self.conv_block_7 = conv_layer(512, 512, kernel_size=1, padding=0)
        self.conv_block_8 = conv_layer(512, 512, kernel_size=3)
    
    def _build_attention_layers(self):
        """Build attention mechanism layers"""
        # Channel attention layers
        self.se_layer_1 = SELayer(256, reduction=16)
        self.se_layer_2 = SELayer(512, reduction=16)
        self.se_layer_3 = SELayer(512 * 4, reduction=16)  # After SPP concatenation
        
        # Spatial pyramid pooling
        self.spp = SpatialPyramidPooling(pool_sizes=[5, 9, 13])
        
        # Spatial attention via transformer
        self.spatial_transform_prep = nn.AdaptiveAvgPool2d((8, 8))
        self.spatial_transformer = TransformerLayer(
            embed_dim=512 * 4, 
            num_heads=8, 
            dropout=self.dropout_rate * 0.5
        )
    
    def _build_fusion_block(self):
        """Build the feature fusion block (Block 2)"""
        conv_layer = DepthwiseSeparableConv if self.use_lightweight else ConvBlock
        
        # Upsampling and fusion layers
        self.fusion_conv_1 = conv_layer(512 * 4, 256, kernel_size=3)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion_conv_2 = conv_layer(256 + 512, 256, kernel_size=3)
        self.fusion_conv_3 = conv_layer(256, 128, kernel_size=1, padding=0)
        self.fusion_conv_4 = conv_layer(128, 128, kernel_size=3)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion_conv_5 = conv_layer(128 + 256, 128, kernel_size=3)
        self.fusion_conv_6 = conv_layer(128, 64, kernel_size=1, padding=0)
        self.fusion_conv_7 = conv_layer(64, 64, kernel_size=3)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion_conv_8 = conv_layer(64 + 128, 64, kernel_size=3)
    
    def _build_classification_head(self):
        """Build the classification head (Block 3)"""
        conv_layer = DepthwiseSeparableConv if self.use_lightweight else ConvBlock
        
        # Final feature processing
        self.head_conv_1 = conv_layer(64, 32, kernel_size=1, padding=0)
        self.head_conv_2 = conv_layer(32, 32, kernel_size=3)
        self.head_conv_3 = conv_layer(32 * 2, 32, kernel_size=3)
        
        self.head_conv_4 = conv_layer(32, 16, kernel_size=1, padding=0)
        self.head_conv_5 = conv_layer(16, 16, kernel_size=3)
        self.head_conv_6 = conv_layer(16 * 2, 16, kernel_size=3)
        
        self.head_conv_7 = conv_layer(16, 8, kernel_size=1, padding=0)
        self.head_conv_8 = conv_layer(8, 8, kernel_size=3)
        self.head_conv_9 = conv_layer(8 * 2, 8, kernel_size=3)
        
        self.head_conv_10 = conv_layer(8, 4, kernel_size=1, padding=0)
        
        # Final classification layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.6),
            nn.Linear(64, self.num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate strategies"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _apply_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention using transformer"""
        b, c, h, w = x.size()
        
        # Prepare for transformer
        x_spatial = self.spatial_transform_prep(x)
        _, _, h_t, w_t = x_spatial.size()
        
        # Convert to sequence format
        x_seq = x_spatial.view(b, c, h_t * w_t).transpose(1, 2)
        x_seq = self.spatial_transformer(x_seq)
        
        # Convert back to feature map
        x_spatial = x_seq.transpose(1, 2).view(b, c, h_t, w_t)
        x_spatial = F.interpolate(x_spatial, size=(h, w), mode='bilinear', align_corners=False)
        
        return x_spatial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RGRNet
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Clear skip connections for new forward pass
        self.skip_connections.clear()
        
        # Block 1: Feature Extraction
        x = self.stem(x)
        self.skip_connections['stem'] = x
        
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        self.skip_connections['early_features'] = x
        
        x = self.conv_block_3(x)
        x = self.se_layer_1(x)
        self.skip_connections['mid_features'] = x
        
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        x = self.conv_block_7(x)
        x = self.se_layer_2(x)
        self.skip_connections['high_features'] = x
        
        x = self.conv_block_8(x)
        
        # Spatial Pyramid Pooling for multi-scale features
        x = self.spp(x)
        
        # Apply spatial attention
        x = self._apply_spatial_attention(x)
        x = self.se_layer_3(x)
        
        # Block 2: Multi-scale Feature Fusion
        x = self.fusion_conv_1(x)
        x = self.upsample_1(x)
        
        # Fuse with high-level features
        if 'high_features' in self.skip_connections:
            skip = self.skip_connections['high_features']
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.fusion_conv_2(x)
        x = self.fusion_conv_3(x)
        x = self.fusion_conv_4(x)
        x = self.upsample_2(x)
        
        # Fuse with mid-level features
        if 'mid_features' in self.skip_connections:
            skip = self.skip_connections['mid_features']
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.fusion_conv_5(x)
        x = self.fusion_conv_6(x)
        x = self.fusion_conv_7(x)
        x = self.upsample_3(x)
        
        # Fuse with early features
        if 'early_features' in self.skip_connections:
            skip = self.skip_connections['early_features']
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.fusion_conv_8(x)
        
        # Block 3: Classification Head
        x = self.head_conv_1(x)
        x_branch = self.head_conv_2(x)
        x = torch.cat([x, x_branch], dim=1)
        x = self.head_conv_3(x)
        
        x = self.head_conv_4(x)
        x_branch = self.head_conv_5(x)
        x = torch.cat([x, x_branch], dim=1)
        x = self.head_conv_6(x)
        
        x = self.head_conv_7(x)
        x_branch = self.head_conv_8(x)
        x = torch.cat([x, x_branch], dim=1)
        x = self.head_conv_9(x)
        
        x = self.head_conv_10(x)
        
        # Global average pooling and classification
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps at different stages
        """
        features = {}
        
        # Forward pass with feature extraction
        _ = self.forward(x)
        
        # Return stored skip connections as feature maps
        return self.skip_connections.copy()
