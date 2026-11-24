
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load dataset and separate answers (y) from image pixels (X)
df = pd.read_csv("train.csv")
print(f"Dataset Loaded Successfully! Shape: {df.shape}")

X = df.drop("label", axis=1).values
y = df["label"].values

# Scale pixel values from [0, 255] to [0, 1]
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"Reduced Feature Shape: {X_train_pca.shape}")
print(f"Total Variance Retained by 50 PCA Components: {explained_var:.2f}%\n")

# Model 1: Support Vector Machine (SVM)
svm_model = SVC(kernel="rbf", C=5, gamma="scale", random_state=42)
svm_model.fit(X_train_pca, y_train)
svm_preds = svm_model.predict(X_test_pca)
svm_acc = accuracy_score(y_test, svm_preds) * 100

# Model 2: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_pca, y_train)
knn_preds = knn_model.predict(X_test_pca)
knn_acc = accuracy_score(y_test, knn_preds) * 100

print("--- Model Accuracy Comparison ---")
print(f"SVM (RBF Kernel) Accuracy : {svm_acc:.2f}%")
print(f"KNN (k=5) Accuracy        : {knn_acc:.2f}%\n")

print("--- Detailed Classification Report (SVM) ---")
print(classification_report(y_test, svm_preds))

plt.figure(figsize=(12, 3))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Actual: {y_test[i]}\nPred: {svm_preds[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()