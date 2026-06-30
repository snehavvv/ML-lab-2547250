import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# 1. Setup results folder and plot style
os.makedirs('results', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

print("=== Task 1: Data Preparation ===")
# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names  # 0 -> malignant, 1 -> benign

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_names)}")
print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
print(f"Target distribution counts:\n{df['target'].value_counts()}")

# Check for missing values and duplicates
missing_vals = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
print(f"Missing values: {missing_vals}")
print(f"Duplicate rows: {duplicate_rows}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Standard StandardScaler applied. Importance: KNN relies on distance calculations (e.g. Euclidean). Features with larger ranges would dominate the distance calculation if not scaled.")

print("\n=== Task 2: Train-Test Split Analysis ===")
splits = {
    "80:20": 0.20,
    "70:30": 0.30,
    "90:10": 0.10
}
split_results = {}

for name, test_size in splits.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    # Fit a baseline KNN model with K=5 for comparison
    knn_temp = KNeighborsClassifier(n_neighbors=5)
    knn_temp.fit(X_train, y_train)
    y_pred = knn_temp.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    split_results[name] = {
        'Train Samples': len(X_train),
        'Test Samples': len(X_test),
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

split_df = pd.DataFrame(split_results).T
print(split_df.to_string())

# Save split results to a text file
with open('results/split_analysis.txt', 'w') as f:
    f.write("=== Train-Test Split Comparison (K=5) ===\n")
    f.write(split_df.to_string())

print("\n=== Task 3: KNN Model with Heuristic K Selection ===")
# 3.1 Heuristic K Selection
# Use 80:20 split as our standard split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
n_samples = len(X_train)
heuristic_k = int(np.round(np.sqrt(n_samples)))
# Ensure K is odd to avoid ties
if heuristic_k % 2 == 0:
    heuristic_k += 1
print(f"Number of training samples (80:20): {n_samples}")
print(f"Heuristic K value (sqrt(n_train), rounded to odd): {heuristic_k}")

# 3.2 Experimenting with nearby values of K (K = 1 to 25)
k_values = list(range(1, 26))
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

# Plot accuracy vs K
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o', linestyle='--', color='royalblue')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='s', linestyle='-', color='darkorange')
plt.axvline(x=heuristic_k, color='red', linestyle=':', label=f'Heuristic K ({heuristic_k})')
plt.title('KNN: Accuracy vs. K Value (80:20 Split)', fontweight='bold')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig('results/accuracy_vs_k.png', dpi=300)
plt.close()
print("Saved Accuracy vs K plot to 'results/accuracy_vs_k.png'")

# Identify optimal K based on test performance
optimal_k_idx = np.argmax(test_accuracies)
optimal_k = k_values[optimal_k_idx]
print(f"Optimal K from train-test split trend: {optimal_k} with test accuracy {test_accuracies[optimal_k_idx]:.4f}")

# 3.3 Decision Boundary Mapping
print("\n=== Task 3.3: Plotting Decision Boundaries (PCA-reduced to 2D) ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(
    X_pca, y, test_size=0.20, random_state=42, stratify=y
)

# Plot decision boundaries for K = 1, 5, 10, 20
k_viz = [1, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

# Define grid for boundaries
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

for i, k in enumerate(k_viz):
    knn_viz = KNeighborsClassifier(n_neighbors=k)
    knn_viz.fit(X_pca_train, y_pca_train)
    
    # Predict on grid
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and scatter
    axes[i].contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    scatter = axes[i].scatter(X_pca_test[:, 0], X_pca_test[:, 1], c=y_pca_test, 
                              edgecolors='k', alpha=0.7, cmap=plt.cm.coolwarm, s=40)
    
    # Label classes based on breast cancer dataset (0 = Malignant, 1 = Benign)
    axes[i].set_title(f'Decision Boundary (K = {k})', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')
    
    # Calculate accuracy on 2D PCA representation
    acc_pca = accuracy_score(y_pca_test, knn_viz.predict(X_pca_test))
    axes[i].text(x_min + 0.5, y_max - 1, f"Test Acc: {acc_pca:.3f}", 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

# Add colorbar legend manually or using a handles/labels helper
handles, labels = scatter.legend_elements()
fig.legend(handles, ['Malignant (0)', 'Benign (1)'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
fig.suptitle('KNN Decision Boundaries (2D PCA Projection of Breast Cancer Dataset)', fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('results/decision_boundaries.png', dpi=300)
plt.close()
print("Saved Decision Boundaries plot to 'results/decision_boundaries.png'")

print("\n=== Task 4: Cross-Validation ===")
# Let's perform 5-fold and 10-fold cross validation for K in 1 to 25
cv_scores_5 = []
cv_scores_10 = []

for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    # Using full features for actual performance evaluation
    scores_5 = cross_val_score(knn_cv, X_scaled, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    scores_10 = cross_val_score(knn_cv, X_scaled, y, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    cv_scores_5.append(scores_5.mean())
    cv_scores_10.append(scores_10.mean())

# Plot CV Accuracies
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores_5, label='5-Fold Cross Validation Accuracy', marker='o', linestyle='-', color='teal')
plt.plot(k_values, cv_scores_10, label='10-Fold Cross Validation Accuracy', marker='s', linestyle='-', color='purple')
plt.axvline(x=heuristic_k, color='red', linestyle=':', label=f'Heuristic K ({heuristic_k})')
plt.title('KNN: Mean Cross-Validation Accuracy vs. K Value', fontweight='bold')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean Accuracy')
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig('results/cv_accuracy_vs_k.png', dpi=300)
plt.close()
print("Saved Cross-Validation Accuracy vs K plot to 'results/cv_accuracy_vs_k.png'")

best_k_cv_5 = k_values[np.argmax(cv_scores_5)]
best_k_cv_10 = k_values[np.argmax(cv_scores_10)]
print(f"Best K (5-fold CV): {best_k_cv_5} (Accuracy: {max(cv_scores_5):.4f})")
print(f"Best K (10-fold CV): {best_k_cv_10} (Accuracy: {max(cv_scores_10):.4f})")

# Final Selection
# Let's select the best performing K.
# Heuristic K = 21 is a bit high and limits model complexity, let's select K from cross validation
# Usually K=9 or K=11 is a sweet spot. Let's see the printout after running it.
# We will dynamically set final_k to the one that maximizes 10-fold CV accuracy.
final_k = best_k_cv_10
print(f"Final selected K for evaluation: {final_k}")

print("\n=== Task 5: Classification Evaluation ===")
# Train final model on 80:20 split using standard 30 features
final_knn = KNeighborsClassifier(n_neighbors=final_k)
final_knn.fit(X_train, y_train)

# Predictions
y_pred = final_knn.predict(X_test)
y_pred_proba = final_knn.predict_proba(X_test)[:, 1]

# Calculate classification metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Let's write the classification report to a file
report = classification_report(y_test, y_pred, target_names=cancer.target_names)
print("\nClassification Report:\n", report)

with open('results/classification_evaluation.txt', 'w') as f:
    f.write("=== Final KNN Classification Metrics ===\n")
    f.write(f"Selected K: {final_k}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.title(f'Confusion Matrix (KNN with K={final_k})', fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.close()
print("Saved Confusion Matrix to 'results/confusion_matrix.png'")

# Plot ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC Score: {roc_auc:.4f}")

with open('results/classification_evaluation.txt', 'a') as f:
    f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title(f'Receiver Operating Characteristic (ROC) Curve (K={final_k})', fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300)
plt.close()
print("Saved ROC Curve to 'results/roc_curve.png'")

print("\n=== Analysis completed successfully. All outputs generated in results/ directory. ===")
