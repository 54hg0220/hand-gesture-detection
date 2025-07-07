# 🖐️ RGRNet: Rapid Gesture Recognition Network

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

**A lightweight spatiotemporal CNN model for real-time hand gesture recognition**

[📄 Paper](https://link.springer.com/article/10.1007/s00521-022-08090-8) • [🎥 Demo Video](https://www.youtube.com/watch?v=mNivI2rsuzU) • [📚 Documentation](./docs/) • [🚀 Quick Start](#quick-start)

</div>

## 🌟 Overview

RGRNet is a lightweight spatiotemporal model developed using PyTorch to classify user hand gestures in real-time, enabling automatic feedback on whether users followed motor task instructions correctly. The model has been deployed in a web-based cognitive testing platform and evaluated in real-world studies.

### ✨ Key Features

- 🚀 **Real-time Performance**: Optimized for real-time gesture recognition
- 📱 **Multi-platform Deployment**: Server, mobile, and edge device support  
- 🧠 **Attention Mechanisms**: SE layers and transformer components for enhanced accuracy
- ⚡ **Lightweight Design**: Multiple model variants for different deployment scenarios
- 🎯 **Production Ready**: Professional training pipeline with comprehensive evaluation
- 🔧 **Configurable**: YAML-based configuration management
- 📊 **Comprehensive Metrics**: Detailed evaluation and visualization tools

## 📊 Model Performance

| Model Variant | Parameters | Model Size | Target FPS | Use Case |
|---------------|------------|------------|------------|----------|
| Standard      | ~15M       | ~60 MB     | 30-60      | Server   |
| Lightweight   | ~8M        | ~32 MB     | 60-120     | Mobile   |
| Efficient     | ~3M        | ~12 MB     | 120-240    | Edge/IoT |

## 📁 Project Structure

```
rgrnet/
├── 📁 models/              # Core model implementations
│   ├── components.py       # Reusable neural network components
│   ├── rgrnet.py          # Main RGRNet model
│   └── factory.py         # Model factory functions
├── 📁 training/           # Training infrastructure
│   └── trainer.py         # Professional training utilities
├── 📁 config/             # Configuration management
│   ├── config.py          # Configuration classes
│   └── configs/           # YAML configuration files
├── 📁 utils/              # Utility functions
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Visualization tools
├── 📁 data/               # Data processing
│   ├── dataset.py         # Dataset definitions
│   └── transforms.py      # Data transformations
├── 📁 scripts/            # Command-line scripts
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── infer.py           # Inference script
└── main.py                # Main entry point
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rgrnet.git
cd rgrnet

# Install dependencies
pip install -r requirements.txt

```

### 2. Basic Usage

```python
from models import create_rgrnet_for_gesture_recognition

# Create a model for gesture recognition
model = create_rgrnet_for_gesture_recognition(
    num_gestures=10,
    input_type='rgb',
    deployment_target='mobile'
)

# Load pre-trained weights (if available)
# model.load_state_dict(torch.load('path/to/weights.pth'))

# Inference
import torch
input_tensor = torch.randn(1, 3, 224, 224)
predictions = model(input_tensor)
```

### 3. Training Your Own Model

```bash
# Prepare your data
python scripts/prepare_data.py --data_dir ./data/gestures

# Train the model
python scripts/train.py --config configs/mobile_config.yaml

# Evaluate the trained model
python scripts/evaluate.py --checkpoint ./checkpoints/best.pth
```

### 4. Using the Model

```bash

# Evaluate a model
python main.py evaluate --checkpoint checkpoints/best.pth --data_dir data/test

# Run inference on a single image/video
python main.py infer --checkpoint checkpoints/best.pth --input path/to/image.jpg

#  Run inference on a real-time webcam
python scripts/infer.py --checkpoint checkpoints/best.pth --input webcam --realtime

# Training visulisation
plot_training_curves(history, save_path='training_curves.png')

# confusion matrix
plot_confusion_matrix_detailed(cm, class_names, save_path='confusion_matrix.png')

# feature visualisation
visualize_feature_maps(feature_maps, save_path='feature_maps.png')
```

## 📋 Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.


## 📖 Configuration

Create custom configurations using YAML files:

```yaml
# configs/my_config.yaml
model:
  name: 'rgrnet'
  variant: 'lightweight'
  num_classes: 10
  input_channels: 3
  image_size: 224

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: 'adamw'
  scheduler: 'cosine'

data:
  dataset_path: './data/gestures'
  augmentation: true
  normalize: true
```

## 🔬 Research & Citation

This work has been published in Neural Computing and Applications (Springer). If you use RGRNet in your research, please cite:

```bibtex
@article{rgrnet2022,
  title={Rapid Gesture Recognition Network for Real-time Hand Gesture Classification},
  author={[Your Name]},
  journal={Neural Computing and Applications},
  year={2022},
  publisher={Springer},
  url={https://link.springer.com/article/10.1007/s00521-022-08090-8}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Team](https://pytorch.org/) for the excellent deep learning framework
- [OpenCV](https://opencv.org/) for computer vision utilities
- [YOLOv5](https://github.com/ultralytics/yolov5) for the P6 model architecture that served as the foundation for RGRNet, with refactored detection and training components
- Research community for inspiration and feedback

## 📞 Contact

- 📧 Email: [huangguan0220@gmail.com]
- 📝 Paper: [Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-022-08090-8)
- 🎥 Demo: [YouTube](https://www.youtube.com/watch?v=mNivI2rsuzU)

---

