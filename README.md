# Human Activity Recognition with PyTorch

---

## Overview
This project implements a deep learning model to classify human activities from sensor data.  
The model is trained using PyTorch and distinguishes between multiple movement classes such as **walking, fast walking, and running**.

---

## Dataset
- Source: Accelerometer and gyroscope recordings (time-series sensor data).
- Preprocessed into feature arrays and corresponding labels (`walk`, `fast`, `run`).
- Split into training and testing sets using `scikit-learn`.

---

## Methodology
1. **Data Preprocessing**
   - Sensor sequences reshaped into suitable input format for PyTorch.
   - Labels encoded as integers (`walk → 0`, `fast → 1`, `run → 2`).
   - DataLoader used for efficient batch training.

2. **Model Architecture**
   - Implemented in PyTorch.
   - Composed of fully connected layers and sequence-based operations (RNN/LSTM).
   - Activation: ReLU.
   - Optimization: Adam with learning rate scheduler.

3. **Training**
   - Loss function: CrossEntropyLoss.
   - Metrics: Accuracy and Confusion Matrix.
   - Trained on Google Colab with GPU acceleration.

---

## Results
- The model achieves reliable classification performance on the test set.
- Example evaluation metrics:
  - Accuracy: ~XX% (replace with your results)
  - Confusion Matrix shows clear separation among the three activities.

---

## Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib
