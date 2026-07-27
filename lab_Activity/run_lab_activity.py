import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_lab():
    print("==================================================")
    print("      DECISION TREE LAB ACTIVITY EXPERIMENTS      ")
    print("==================================================\n")

    # ----------------------------------------------------
    # TASK 1: Dataset Exploration
    # ----------------------------------------------------
    print("--- TASK 1: DATASET EXPLORATION ---")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)
    
    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Feature names: {feature_names}")
    print(f"Target classes: {target_names}")
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    print("\nFirst 5 records:")
    print(df.head())
    
    print("\nClass distribution:")
    class_counts = df['species'].value_counts()
    print(class_counts)
    
    # ----------------------------------------------------
    # TASK 2: Data Preparation
    # ----------------------------------------------------
    print("\n--- TASK 2: DATA PREPARATION ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    ) # Note: let's check standard split without stratify or with stratify. Standard train_test_split(X, y, test_size=0.20, random_state=42)
    X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set shape: {X_train_ns.shape}")
    print(f"Testing set shape: {X_test_ns.shape}")

    # We will use X_train_ns, X_test_ns, y_train_ns, y_test_ns for standard non-stratified random_state=42 split as standard scikit-learn practice when unspecified
    X_tr, X_te, y_tr, y_te = X_train_ns, X_test_ns, y_train_ns, y_test_ns

    # ----------------------------------------------------
    # TASK 3: Building a Decision Tree Classifier (Default)
    # ----------------------------------------------------
    print("\n--- TASK 3: DEFAULT DECISION TREE CLASSIFIER ---")
    dt_default = DecisionTreeClassifier(random_state=42)
    dt_default.fit(X_tr, y_tr)
    y_pred_default = dt_default.predict(X_te)
    
    acc_default = accuracy_score(y_te, y_pred_default)
    cm_default = confusion_matrix(y_te, y_pred_default)
    cr_default = classification_report(y_te, y_pred_default, target_names=target_names)
    
    print(f"Accuracy Score: {acc_default:.4f}")
    print("\nConfusion Matrix:")
    print(cm_default)
    print("\nClassification Report:")
    print(cr_default)

    # ----------------------------------------------------
    # TASK 4: Visualizing the Decision Tree
    # ----------------------------------------------------
    print("\n--- TASK 4: TREE VISUALIZATION ---")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt_default, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=ax)
    plt.title("Decision Tree Classifier (Default parameters)")
    plt.tight_layout()
    plt.savefig("decision_tree_default.png", dpi=300)
    plt.close()
    
    depth_default = dt_default.get_depth()
    n_leaves_default = dt_default.get_n_leaves()
    print(f"Max depth of tree: {depth_default}")
    print(f"Number of leaf nodes: {n_leaves_default}")
    print(f"Root node feature: {feature_names[dt_default.tree_.feature[0]]}")
    print(f"Root node threshold: {dt_default.tree_.threshold[0]:.4f}")

    # ----------------------------------------------------
    # TASK 5: Comparing Gini Index vs Entropy
    # ----------------------------------------------------
    print("\n--- TASK 5: GINI VS ENTROPY COMPARISON ---")
    dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_tr, y_tr)
    dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42).fit(X_tr, y_tr)
    
    acc_tr_gini = accuracy_score(y_tr, dt_gini.predict(X_tr))
    acc_te_gini = accuracy_score(y_te, dt_gini.predict(X_te))
    acc_tr_ent = accuracy_score(y_tr, dt_entropy.predict(X_tr))
    acc_te_ent = accuracy_score(y_te, dt_entropy.predict(X_te))
    
    print(f"Gini - Train Acc: {acc_tr_gini:.4f}, Test Acc: {acc_te_gini:.4f}, Depth: {dt_gini.get_depth()}, Leaves: {dt_gini.get_n_leaves()}, Root Feature: {feature_names[dt_gini.tree_.feature[0]]}")
    print(f"Entropy - Train Acc: {acc_tr_ent:.4f}, Test Acc: {acc_te_ent:.4f}, Depth: {dt_entropy.get_depth()}, Leaves: {dt_entropy.get_n_leaves()}, Root Feature: {feature_names[dt_entropy.tree_.feature[0]]}")

    # Plot both trees for comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plot_tree(dt_gini, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=axes[0])
    axes[0].set_title("Decision Tree (Criterion = Gini)")
    plot_tree(dt_entropy, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=axes[1])
    axes[1].set_title("Decision Tree (Criterion = Entropy)")
    plt.tight_layout()
    plt.savefig("tree_gini_vs_entropy.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # TASK 6: Effect of max_depth
    # ----------------------------------------------------
    print("\n--- TASK 6: EFFECT OF MAX_DEPTH ---")
    max_depths = [1, 2, 3, 4, None]
    depth_results = []
    for md in max_depths:
        clf = DecisionTreeClassifier(max_depth=md, random_state=42).fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        actual_depth = clf.get_depth()
        depth_results.append({
            'Max Depth': str(md),
            'Training Accuracy': tr_acc,
            'Testing Accuracy': te_acc,
            'Tree Depth': actual_depth,
            'Leaves': clf.get_n_leaves()
        })
        print(f"max_depth={str(md):4s} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f} | Actual Depth: {actual_depth}")

    # ----------------------------------------------------
    # TASK 7: Effect of min_samples_split
    # ----------------------------------------------------
    print("\n--- TASK 7: EFFECT OF MIN_SAMPLES_SPLIT ---")
    splits = [2, 5, 10, 20]
    split_results = []
    for mss in splits:
        clf = DecisionTreeClassifier(min_samples_split=mss, random_state=42).fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        split_results.append({
            'min_samples_split': mss,
            'Training Accuracy': tr_acc,
            'Testing Accuracy': te_acc,
            'Tree Depth': clf.get_depth(),
            'Leaves': clf.get_n_leaves()
        })
        print(f"min_samples_split={mss:2d} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f} | Depth: {clf.get_depth()} | Leaves: {clf.get_n_leaves()}")

    # ----------------------------------------------------
    # TASK 8: Effect of min_samples_leaf
    # ----------------------------------------------------
    print("\n--- TASK 8: EFFECT OF MIN_SAMPLES_LEAF ---")
    leafs = [1, 2, 5, 10]
    leaf_results = []
    for msl in leafs:
        clf = DecisionTreeClassifier(min_samples_leaf=msl, random_state=42).fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        leaf_results.append({
            'min_samples_leaf': msl,
            'Training Accuracy': tr_acc,
            'Testing Accuracy': te_acc,
            'Tree Depth': clf.get_depth(),
            'Leaves': clf.get_n_leaves()
        })
        print(f"min_samples_leaf={msl:2d} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f} | Depth: {clf.get_depth()} | Leaves: {clf.get_n_leaves()}")

    # ----------------------------------------------------
    # TASK 9: Hyperparameter Tuning via GridSearchCV
    # ----------------------------------------------------
    print("\n--- TASK 9: HYPERPARAMETER TUNING ---")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid.fit(X_tr, y_tr)
    
    best_clf = grid.best_estimator_
    test_acc_best = accuracy_score(y_te, best_clf.predict(X_te))
    
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best 5-Fold Cross Validation Accuracy: {grid.best_score_:.4f}")
    print(f"Test Accuracy of Optimized Model: {test_acc_best:.4f}")
    print(f"Optimized Tree Depth: {best_clf.get_depth()}")
    print(f"Optimized Tree Leaves: {best_clf.get_n_leaves()}")

    # Save best tree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(best_clf, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=ax)
    plt.title("Optimized Decision Tree (GridSearchCV)")
    plt.tight_layout()
    plt.savefig("decision_tree_optimized.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_lab()
