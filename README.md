# Brain Tumor MRI Classification using Deep Learning

## 🧠 Project Overview
An advanced deep learning model is designed for binary classification of brain MRI scans to detect tumors (benign vs. malignant). This model leverages a custom attention-enhanced ResNet50 architecture, incorporating state-of-the-art training techniques to achieve high accuracy and reliability.

The development process involved benchmarking performance across several architectures, including **ResNet18, ResNet50, EfficientNet, LSTM, 3D-CNN and Vision Transformer (ViT)**, to identify the optimal backbone for tumor detection. Furthermore, attention mechanisms like **Attention Gate** and **Self-Attention** were integrated to enhance feature refinement and improve the focus on critical tumor regions. A novel implementation of the **Attention-Based Multiple Instance Learning (ABMIL)** framework further boosts the model's interpretability and performance.

The model was trained and evaluated on the **[IAAA Challenge Dataset](https://www.kaggle.com/datasets/amirmohammadparvizi/newiaaa)**, demonstrating its robustness and adaptability to real-world medical imaging scenarios.

## 🌟 Key Features
- Custom attention mechanism for focused feature extraction
- ResNet50 backbone with attention gates
- Focal Loss for handling class imbalance
- Mixed precision training for improved performance
- Advanced data augmentation pipeline
- Comprehensive evaluation metrics and visualization

## 📊 Model Architecture
The model combines ResNet50 with custom attention gates:
- Base: Pretrained ResNet50
- Custom Components:
  - 3 Attention Gates (2048->1024, 2048->512, 2048->256)
  - Global Average Pooling
  - Custom classifier head (2048->512->1)
- Training Enhancements:
  - Focal Loss (alpha=0.25, gamma=2)
  - OneCycleLR scheduling
  - Mixed precision training with GradScaler

## 🛠 Requirements
```
torch>=1.7.0
torchvision>=0.8.0
albumentations>=0.5.2
opencv-python>=4.5.0
numpy>=1.19.2
pandas>=1.2.0
scikit-learn>=0.24.0
tqdm>=4.50.0
matplotlib>=3.3.0
imbalanced-learn>=0.8.0
```

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/awrsha/Classification-of-brain-diseases.git
cd Classification-of-brain-diseases

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python 0.7914.py --batch_size 32 --epochs 60 --lr 1e-3
```

## 📈 Performance Metrics
The model is evaluated using multiple metrics:
- ROC-AUC
- Precision-Recall curves
- F1 Score
- Confusion Matrix
- Threshold Analysis

## 🔧 Model Configuration
Key hyperparameters:
```python
{
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 60,
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2
}
```

## 🎯 Data Augmentation
Comprehensive augmentation pipeline including:
- Random rotation
- Flips
- Brightness/Contrast adjustments
- Coarse dropout
- Normalization
- Resize to 224x224

## 📊 Results Visualization
The project includes:
- ROC curves
- Precision-Recall curves
- Random sample predictions
- Confusion matrices
- Threshold analysis plots

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/AmirrHussain/Classification-of-brain-diseases#MIT-1-ov-file) file for details.

## 🙏 Acknowledgments
- The dataset providers
- The medical professionals who helped with data annotation
- The open-source community for the tools and libraries used

## 📧 Contact
For questions or feedback, please open an issue or contact [official.parvizi@gmail.com]
