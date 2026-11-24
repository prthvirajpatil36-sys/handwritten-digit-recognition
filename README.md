# Handwritten Digit Recognition Using PCA & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20MNIST-20BEFF)

A machine learning pipeline that classifies handwritten digits (`0`–`9`) by combining **Principal Component Analysis (PCA)** for dimensionality reduction with **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** classifiers. Compressing the feature space by **15.7x** (`784` → `50` features) preserves **98.12%** classification accuracy while significantly speeding up training and inference.

---

## Dataset & Setup
- **Source:** [Kaggle Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer)
- **Size:** `42,000` labeled `28 × 28` grayscale images (`784` pixel features per sample)
- **Preparation:** Download `train.csv` from Kaggle and place it in the project root directory before running the script.

---

## Methodology
1. **Preprocessing & Split:** Scaled pixel intensities from `[0, 255]` to `[0.0, 1.0]` and applied a stratified **80/20** train-test split (`33,600` training / `8,400` validation samples) to preserve class balance across all 10 digits.
2. **Dimensionality Reduction (PCA):** Reduced `784` raw pixel features to **50 principal components**, retaining **82.46%** of total dataset variance.
3. **Model Training & Evaluation:** Trained and benchmarked an RBF-kernel **SVM** against a distance-based **KNN** classifier on the reduced feature space.

---

## Results

| Model | Features Used | Variance Retained | Validation Accuracy |
| :--- | :---: | :---: | :---: |
| **SVM (RBF Kernel)** | 50 PCA components | 82.46% | **98.12%** |
| **KNN (`k=5`)** | 50 PCA components | 82.46% | **97.18%** |

### Sample Predictions
![Sample Digit Predictions](sample_predictions.png)

---

## Project Structure
```text
handwritten-digit-recognition-pca/
├── digit_recognition.py     # Data preprocessing, PCA, model training & evaluation
├── requirements.txt         # Project dependencies
├── sample_predictions.png   # Visualization of test samples with predicted labels
├── .gitignore               # Ignores dataset CSVs and local environment files
└── README.md                # Project documentation