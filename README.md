🏴‍☠️ One Piece Character Classifier with NdLinear

This project is my submission for the Summer/Fall 2025 ML Research Internship at Ensemble, showcasing a practical and creative use of the open-source NdLinear layer for shrinking and accelerating neural networks.

# Project Overview

I built a deep learning model that classifies characters from the anime One Piece using images. The model is implemented in PyTorch and trained twice:

- Once with standard nn.Linear layers (Baseline)
- Once with NdLinear layers

The goal is to compare the two in terms of:

- Inference Time
- Parameter Count
- Prediction Accuracy

To make it interactive, I created a Gradio demo where users can upload an image and see side-by-side model predictions and performance.

# Model Architecture
```
Conv2d(3, 32, kernel_size=3)
ReLU + MaxPool
Conv2d(32, 64, kernel_size=3)
ReLU + MaxPool
Flatten
NdLinear / nn.Linear (64 * 16 * 16 -> 128)
NdLinear / nn.Linear (128 -> #classes)

```
Dataset: 18 One Piece characters (Luffy, Zoro, Nami, etc.) from a Kaggle dataset.
Image Size: 64x64
Dataset Link: - [One Piece Dataset (Kaggle)](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier)


# Gradio Demo

Upload a One Piece character image to get real-time predictions and a comparison chart.

The app displays:

- Predictions from both models
- Inference time
- Parameter count

A bar chart showing performance difference
```
pip install torch torchvision ndlinear gradio matplotlib
python app.py

```
