"""
Linear Regression using Gradient Descent Optimization Algorithm
Dataset: UCI Student Performance Dataset (student-mat.csv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure output directory for plots exists
os.makedirs("plots", exist_ok=True)

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ==========================================
# 1. Custom Linear Regression with Gradient Descent
# ==========================================
class LinearRegressionGD:
    """
    Linear Regression model optimized using Batch Gradient Descent.
    
    Formulas:
        Hypothesis: y_pred = X * w + b
        Cost (MSE): J(w, b) = (1 / (2 * m)) * sum((y_pred - y)^2)
        Gradients:
            dj_dw = (1 / m) * X.T * (y_pred - y)
            dj_db = (1 / m) * sum(y_pred - y)
        Update Rules:
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
    """
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.n_iterations):
            # Compute hypothesis / predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Error vector
            errors = y_pred - y
            
            # Compute Cost (Mean Squared Error Cost Function)
            cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
            self.cost_history.append(cost)
            
            # Compute Gradients
            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def load_and_preprocess_data(filepath="student-mat.csv"):
    print(f"--> Loading dataset from '{filepath}'...")
    # Load dataset (UCI Student Performance uses semicolon delimiter)
    df = pd.read_csv(filepath, sep=';')
    print(f"Dataset shape: {df.shape}")
    
    # Check missing values
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    
    # Target variable: G3 (Final grade)
    # Features X: drop G3
    X = df.drop(columns=['G3'])
    y = df['G3'].values
    
    # Encode categorical variables using One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"Features shape after One-Hot Encoding: {X_encoded.shape}")
    
    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Feature Scaling: Z-score Normalization (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_encoded.columns, scaler


# ==========================================
# 3. Experiments & Evaluation
# ==========================================
def run_experiments():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data("student-mat.csv")
    
    # ------------------------------------------
    # Experiment 1: Learning Rate Impact on Convergence
    # ------------------------------------------
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
    n_iters = 500
    lr_cost_histories = {}
    
    print("\n--> Experimenting with different Learning Rates...")
    plt.figure(figsize=(10, 6))
    for alpha in learning_rates:
        model = LinearRegressionGD(learning_rate=alpha, n_iterations=n_iters)
        model.fit(X_train, y_train)
        lr_cost_histories[alpha] = model.cost_history
        plt.plot(range(1, n_iters + 1), model.cost_history, label=f"α = {alpha}", linewidth=2)
    
    plt.title("Effect of Learning Rate on Gradient Descent Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Cost J(w, b) [Half MSE]", fontsize=12)
    plt.legend(title="Learning Rate (α)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/learning_rate_comparison.png")
    plt.close()
    print("Saved 'plots/learning_rate_comparison.png'")

    # ------------------------------------------
    # Experiment 2: Optimal Model Training
    # ------------------------------------------
    optimal_alpha = 0.05
    n_iterations = 1000
    print(f"\n--> Training Optimal Custom GD Model (alpha={optimal_alpha}, iterations={n_iterations})...")
    
    gd_model = LinearRegressionGD(learning_rate=optimal_alpha, n_iterations=n_iterations)
    gd_model.fit(X_train, y_train)
    
    # Plot Cost Convergence for optimal model
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_iterations + 1), gd_model.cost_history, color="#2b5c8f", linewidth=2.5)
    plt.title(f"Cost Function Convergence (α = {optimal_alpha})", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost J(w, b)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/cost_convergence.png")
    plt.close()
    print("Saved 'plots/cost_convergence.png'")
    
    # Evaluate Custom GD Model
    train_metrics_gd = gd_model.evaluate(X_train, y_train)
    test_metrics_gd = gd_model.evaluate(X_test, y_test)

    # ------------------------------------------
    # Experiment 3: Benchmark against Scikit-Learn LinearRegression
    # ------------------------------------------
    print("\n--> Training Scikit-Learn OLS Linear Regression Baseline...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    sklearn_preds_test = sklearn_model.predict(X_test)
    test_metrics_sk = {
        "MAE": mean_absolute_error(y_test, sklearn_preds_test),
        "MSE": mean_squared_error(y_test, sklearn_preds_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, sklearn_preds_test)),
        "R2": r2_score(y_test, sklearn_preds_test)
    }

    # Print Comparison Table
    print("\n=======================================================")
    print("              MODEL EVALUATION METRICS                 ")
    print("=======================================================")
    print(f"{'Metric':<10} | {'Custom GD (Test)':<18} | {'Scikit-Learn (Test)':<18}")
    print("-" * 55)
    for metric in ["MAE", "MSE", "RMSE", "R2"]:
        print(f"{metric:<10} | {test_metrics_gd[metric]:<18.4f} | {test_metrics_sk[metric]:<18.4f}")
    print("=======================================================\n")

    # ------------------------------------------
    # Visualizations: Predictions & Residuals
    # ------------------------------------------
    predictions_test = gd_model.predict(X_test)
    residuals = y_test - predictions_test
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions_test, alpha=0.7, color="#1f77b4", edgecolors="k", s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Prediction (y = x)")
    plt.title("Actual vs Predicted Final Grades (G3)", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Grade (G3)", fontsize=12)
    plt.ylabel("Predicted Grade (G3)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()
    print("Saved 'plots/actual_vs_predicted.png'")

    # Plot 2: Residuals Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=20)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Residuals Distribution (Errors = Actual - Predicted)", fontsize=14, fontweight='bold')
    plt.xlabel("Residual Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/residuals_distribution.png")
    plt.close()
    print("Saved 'plots/residuals_distribution.png'")

if __name__ == "__main__":
    run_experiments()
