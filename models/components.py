"""
Common neural network components for RGRNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    
    Paper: "Squeeze-and-Excitation Networks" (CVPR 2018)
    Implements channel attention mechanism to enhance feature representation
    
    Args:
        channel (int): Number of input channels
        reduction (int): Reduction ratio for the bottleneck layer
    """
    
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE layer
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with channel attention applied
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TransformerLayer(nn.Module):
    """
    Lightweight Transformer layer for spatial attention
    
    Implements self-attention mechanism for capturing long-range dependencies
    in gesture recognition tasks
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (Optional[int]): Feed-forward dimension. If None, defaults to 4 * embed_dim
        dropout (float): Dropout rate
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 ff_dim: Optional[int] = None, dropout: float = 0.1):
        super(TransformerLayer, self).__init__()
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer layer
        
        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
            
        Returns:
            Output tensor with self-attention applied
        """
        # Self-attention block
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward block
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling module
    
    Paper: "Spatial Pyramid Pooling in Deep Convolutional Networks" (ECCV 2014)
    Captures multi-scale features by pooling at different scales
    
    Args:
        pool_sizes (List[int]): List of pooling kernel sizes
    """
    
    def __init__(self, pool_sizes: List[int] = [5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size//2)
            for size in pool_sizes
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPP module
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Concatenated multi-scale features
        """
        pyramid_features = [x]  # Original features
        
        for pool in self.pools:
            pyramid_features.append(pool(x))
        
        return torch.cat(pyramid_features, dim=1)


class ConvBlock(nn.Module):
    """
    Standard convolutional block with normalization and activation
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Padding size
        use_bn (bool): Whether to use batch normalization
        activation (str): Type of activation function
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True,
                 activation: str = 'relu'):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution for efficient computation
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Padding size
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
