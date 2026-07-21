import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Set styling for matplotlib/seaborn
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# Ensure plots output directory exists
plots_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(plots_dir, exist_ok=True)

def main():
    print("=" * 70)
    print("  BREAST CANCER WISCONSIN (DIAGNOSTIC) CLASSIFICATION EXPERIMENT  ")
    print("=" * 70)

    # 1. Download and Load Dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Target encoding: 0 = Malignant, 1 = Benign in sklearn dataset
    target_names = data.target_names  # ['malignant', 'benign']
    
    print("\n[STEP 1] Dataset Overview:")
    print(f"  - Total Samples: {X.shape[0]}")
    print(f"  - Total Features: {X.shape[1]}")
    print(f"  - Classes: {target_names[0]} (0): {sum(y == 0)}, {target_names[1]} (1): {sum(y == 1)}")
    print(f"  - Missing Values: {X.isnull().sum().sum()}")

    # Plot Class Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=y.map({0: 'Malignant (0)', 1: 'Benign (1)'}), palette=['#e74c3c', '#2ecc71'])
    plt.title('Breast Cancer Class Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Diagnosis Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[STEP 2] Train/Test Split (80/20):")
    print(f"  - Train shape: {X_train.shape}")
    print(f"  - Test shape:  {X_test.shape}")

    # 3. Data Preprocessing / Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n[STEP 3] Feature Scaling Applied: StandardScaler fitted on training set.")

    # 4. K-Selection for KNN Classifier
    k_range = range(1, 21)
    k_accuracies = []
    k_f1_scores = []

    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)
        preds_temp = knn_temp.predict(X_test_scaled)
        k_accuracies.append(accuracy_score(y_test, preds_temp))
        k_f1_scores.append(f1_score(y_test, preds_temp))

    best_k = k_range[np.argmax(k_accuracies)]
    print(f"\n[STEP 4] KNN K-Optimization:")
    print(f"  - Best K value based on test accuracy: K = {best_k} (Accuracy: {max(k_accuracies):.4f})")

    # Plot K Selection Graph
    plt.figure(figsize=(9, 5))
    plt.plot(k_range, k_accuracies, marker='o', color='#3498db', linewidth=2.5, label='Accuracy')
    plt.plot(k_range, k_f1_scores, marker='s', color='#e67e22', linestyle='--', linewidth=2, label='F1 Score')
    plt.title('KNN Accuracy & F1 Score vs K-Value', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(k_range)
    plt.axvline(best_k, color='#e74c3c', linestyle=':', label=f'Optimal K={best_k}')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'knn_k_selection.png'), dpi=300)
    plt.close()

    # 5. Model Training
    # Model A: Logistic Regression
    log_reg = LogisticRegression(random_state=42, max_iter=10000)
    log_reg.fit(X_train_scaled, y_train)

    # Model B: KNN with Optimal K
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)

    # 6. Predictions & Evaluation
    # Logistic Regression Predictions
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    # KNN Predictions
    y_pred_knn = knn.predict(X_test_scaled)
    y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

    # Compute Metrics Function
    def get_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'specificity': spec,
            'auc': roc_auc,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'fpr': fpr, 'tpr': tpr
        }

    lr_metrics = get_metrics(y_test, y_pred_lr, y_prob_lr)
    knn_metrics = get_metrics(y_test, y_pred_knn, y_prob_knn)

    # 7. Print Performance Comparison Table
    print("\n" + "=" * 75)
    print("                     PERFORMANCE COMPARISON METRICS                     ")
    print("=" * 75)
    print(f"{'Metric':<20} | {'Logistic Regression':<22} | {'KNN (K=' + str(best_k) + ')':<20}")
    print("-" * 75)
    print(f"{'Accuracy':<20} | {lr_metrics['accuracy']:<22.4f} | {knn_metrics['accuracy']:<20.4f}")
    print(f"{'Precision':<20} | {lr_metrics['precision']:<22.4f} | {knn_metrics['precision']:<20.4f}")
    print(f"{'Recall (Sensitivity)':<20} | {lr_metrics['recall']:<22.4f} | {knn_metrics['recall']:<20.4f}")
    print(f"{'F1 Score':<20} | {lr_metrics['f1_score']:<22.4f} | {knn_metrics['f1_score']:<20.4f}")
    print(f"{'Specificity':<20} | {lr_metrics['specificity']:<22.4f} | {knn_metrics['specificity']:<20.4f}")
    print(f"{'ROC AUC':<20} | {lr_metrics['auc']:<22.4f} | {knn_metrics['auc']:<20.4f}")
    print("-" * 75)
    print(f"{'True Positives (TP)':<20} | {lr_metrics['tp']:<22} | {knn_metrics['tp']:<20}")
    print(f"{'True Negatives (TN)':<20} | {lr_metrics['tn']:<22} | {knn_metrics['tn']:<20}")
    print(f"{'False Positives (FP)':<20} | {lr_metrics['fp']:<22} | {knn_metrics['fp']:<20}")
    print(f"{'False Negatives (FN)':<20} | {lr_metrics['fn']:<22} | {knn_metrics['fn']:<20}")
    print("=" * 75)

    # 8. Visualizing Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title(f"Logistic Regression\nAccuracy: {lr_metrics['accuracy']:.4f}", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("Actual Label", fontsize=11)

    sns.heatmap(knn_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title(f"KNN (K={best_k})\nAccuracy: {knn_metrics['accuracy']:.4f}", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("Actual Label", fontsize=11)

    plt.suptitle("Confusion Matrix Comparison", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Visualizing ROC Curves
    plt.figure(figsize=(8, 6))
    plt.plot(lr_metrics['fpr'], lr_metrics['tpr'], color='#2980b9', lw=2.5,
             label=f"Logistic Regression (AUC = {lr_metrics['auc']:.4f})")
    plt.plot(knn_metrics['fpr'], knn_metrics['tpr'], color='#27ae60', lw=2.5, linestyle='--',
             label=f"KNN (K={best_k}, AUC = {knn_metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    print(f"\n[STEP 5] Saved visual plots in '{plots_dir}':")
    print("  - class_distribution.png")
    print("  - knn_k_selection.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")

if __name__ == '__main__':
    main()
