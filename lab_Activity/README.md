# Machine Learning Lab Activity: Decision Tree Classifier

**Dataset:** Iris Dataset (`sklearn.datasets.load_iris()`)  
**Student Repo:** `ML-lab-2547250`  

---

## Contents of this Directory

- `Decision_Tree_Lab_Activity.ipynb`: Complete executed Jupyter Notebook containing all 10 tasks, output figures, metrics tables, grid search, and analytical responses.
- `run_lab_activity.py`: Python script executing all 10 tasks, metric computations, and plot generation.
- `decision_tree_default.png`: High-resolution visualization plot of the default Decision Tree Classifier.
- `tree_gini_vs_entropy.png`: Visual comparison plot between Gini Impurity and Entropy Decision Trees.
- `decision_tree_optimized.png`: High-resolution visualization plot of the GridSearch-optimized Decision Tree.

---

## Summary of Lab Activity Tasks

1. **Task 1: Dataset Exploration** — Evaluated dataset dimensions (150 samples, 4 features), feature names, species target classes, sample records, and class distribution.
2. **Task 2: Data Preparation** — 80/20 train-test split (`random_state=42`), rationale for train/test splitting.
3. **Task 3: Building a Decision Tree Classifier** — Default `DecisionTreeClassifier`, predictions, 100% test accuracy score, confusion matrix, and classification report.
4. **Task 4: Visualizing the Decision Tree** — Visual tree diagram via `plot_tree()`, identification of root node (`petal length (cm) <= 2.45`), internal nodes, 10 leaves, max depth (6), and feature selection rationale.
5. **Task 5: Comparing Gini Index and Entropy** — Comparison table evaluating accuracy, tree depth, leaf count, and root features under `criterion='gini'` vs `criterion='entropy'`.
6. **Task 6: Effect of Maximum Tree Depth** — Evaluated `max_depth` across `[1, 2, 3, 4, None]`, completed comparison table, identified underfitting (`depth=1`), best generalization (`depth=3`), and overfitting (`depth=None`).
7. **Task 7: Effect of min_samples_split** — Evaluated `min_samples_split` across `[2, 5, 10, 20]`, pre-pruning complexity reduction analysis.
8. **Task 8: Effect of min_samples_leaf** — Evaluated `min_samples_leaf` across `[1, 2, 5, 10]`, overfitting mitigation analysis.
9. **Task 9: Hyperparameter Tuning** — 5-Fold `GridSearchCV` optimization (`best_params: {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 2}`), cross-validation score (95.83%), and default vs optimized model comparison.
10. **Task 10: Analysis** — Conceptual answers regarding criterion, depth trade-offs, pre-pruning sample parameters, dominant hyperparameter, and final model recommendation.
