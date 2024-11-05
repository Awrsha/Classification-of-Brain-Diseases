# Classification of brain diseases with PyTorch 🖼️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-%23EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project for image classification using PyTorch, OpenCV, and `albumentations` for data augmentation. This repository includes data preprocessing, model training, and evaluation workflows for image datasets, with a focus on efficient training with GPU support.

## Features ✨

- **Data Augmentation**: Utilizes `albumentations` for robust image augmentations.
- **Custom Dataset Loader**: Simplifies loading and preprocessing of custom image datasets.
- **GPU-Accelerated Training**: Supports CUDA for high-performance training on NVIDIA GPUs.
- **Metrics Calculation**: Calculates precision and recall metrics to evaluate model performance.
- **Resampling**: Balances classes using `RandomOverSampler` from `imblearn`.

## Installation 🛠️

Clone the repository and install dependencies:

```bash
git clone https://github.com/AmirrHussain/Classification-of-brain-diseases.git
cd Classification-of-brain-diseases
pip install -r requirements.txt
```

> **Note**: Ensure CUDA is available for GPU-accelerated training.

3. **Evaluate Model**: After training, evaluate the model on a test set to view metrics.

## Results 📊

Example results and performance metrics will appear here:

| Metric     | Value     |
|------------|-----------|
| Precision  | 0.79      |
| Recall     | 0.78      |

## Contributing 🤝

Contributions are welcome! Please open issues or pull requests as needed.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
