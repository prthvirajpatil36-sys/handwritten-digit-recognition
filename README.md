# Handwritten Digit Recognition Using PCA and Machine Learning

A Python machine learning mini-project that classifies handwritten digits (`0`–`9`) by combining **Principal Component Analysis (PCA)** for dimensionality reduction with **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** classifiers.

## Dataset
- **Source:** [Kaggle Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer)
- **Details:** `42,000` labeled `28 x 28` grayscale images (`784` pixel features per image).
- **Setup:** Download `train.csv` from Kaggle and place it in the root directory before running the script.

## Workflow & Results
1. **Preprocessing:** Normalized pixel intensities from `[0, 255]` to `[0.0, 1.0]` and performed a stratified **80/20** train-test split (`33,600` training / `8,400` testing samples).
2. **Dimensionality Reduction (PCA):** Compressed `784` pixel features into **50 principal components**, retaining **82.46%** of total variance.
3. **Classification Performance:**
   - **SVM (RBF Kernel):** `98.12%` validation accuracy
   - **KNN (k=5):** `97.18%` validation accuracy

## Sample Output
![Sample Digit Predictions](sample_predictions.png)

## Project Structure
```text
handwritten-digit-recognition-pca/
├── digit_recognition.py
├── requirements.txt
├── sample_predictions.png
├── .gitignore
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python digit_recognition.py
```