import json

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Simple Linear Regression: Predicting GPA from Package Expectations\n",
                    "\n",
                    "This notebook builds a Simple Linear Regression model to predict student GPA ($Y$) based on their package expectations ($X$, in LPA) using both Scikit-learn and manual Ordinary Least Squares (OLS) calculations.\n",
                    "\n",
                    "### Objectives:\n",
                    "1. Load the preprocessed student survey dataset `cleaned_survey.csv`.\n",
                    "2. Split the dataset into an 80% training set and a 20% testing set.\n",
                    "3. **Part A**: Fit a Scikit-Learn `LinearRegression` model, calculate metrics (MAE, MSE, RMSE, $R^2$), and visualize the results.\n",
                    "4. **Part B**: Compute regression coefficients ($b_0$, $b_1$) manually using OLS formulas with NumPy and compare them to Scikit-Learn's results.\n",
                    "5. **Part C**: Persist model parameters ($b_0$, $b_1$, $R^2_{train}$, $R^2_{test}$) to `model_params.json`.\n",
                    "6. Analyze and discuss the model's coefficients, predictions, and reliability."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Loading Cleaned Data and Train-Test Split"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "import json\n",
                    "\n",
                    "# Aesthetics\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                    "plt.rcParams[\"font.family\"] = \"sans-serif\"\n",
                    "\n",
                    "# Load preprocessed survey data\n",
                    "df = pd.read_csv('cleaned_survey.csv')\n",
                    "print(f\"Loaded dataset shape: {df.shape}\")\n",
                    "\n",
                    "# Independent predictor variable (X): package_exp\n",
                    "# Dependent target variable (Y): gpa\n",
                    "X = df[['package_exp']].values\n",
                    "y = df['gpa'].values\n",
                    "\n",
                    "# Split 80/20 train/test split\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "print(f\"Training set size: {X_train.shape[0]} samples\")\n",
                    "print(f\"Testing set size: {X_test.shape[0]} samples\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part A: Scikit-Learn Simple Linear Regression\n",
                    "\n",
                    "We train the model on the training set and print intercept ($b_0$) and slope ($b_1$). We then evaluate on the test set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Fit Linear Regression\n",
                    "sk_model = LinearRegression()\n",
                    "sk_model.fit(X_train, y_train)\n",
                    "\n",
                    "# Extract parameters\n",
                    "b0_sklearn = sk_model.intercept_\n",
                    "b1_sklearn = sk_model.coef_[0]\n",
                    "\n",
                    "print(\"=== Scikit-Learn Model Parameters ===\")\n",
                    "print(f\"Intercept (b0): {b0_sklearn:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_sklearn:.6f}\")\n",
                    "\n",
                    "# Make predictions\n",
                    "y_pred_train = sk_model.predict(X_train)\n",
                    "y_pred_test = sk_model.predict(X_test)\n",
                    "\n",
                    "# Calculate metrics on the test set\n",
                    "mae_test = mean_absolute_error(y_test, y_pred_test)\n",
                    "mse_test = mean_squared_error(y_test, y_pred_test)\n",
                    "rmse_test = np.sqrt(mse_test)\n",
                    "r2_test = r2_score(y_test, y_pred_test)\n",
                    "r2_train = r2_score(y_train, y_pred_train)\n",
                    "\n",
                    "print(\"\\n=== Testing Set Evaluation Metrics ===\")\n",
                    "print(f\"Mean Absolute Error (MAE):     {mae_test:.6f}\")\n",
                    "print(f\"Mean Squared Error (MSE):      {mse_test:.6f}\")\n",
                    "print(f\"Root Mean Squared Error (RMSE): {rmse_test:.6f}\")\n",
                    "print(f\"R-squared (R2) on Train Set:   {r2_train:.6f}\")\n",
                    "print(f\"R-squared (R2) on Test Set:    {r2_test:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Visualization of the Model Fit\n",
                    "\n",
                    "We plot a scatter plot of the test set actual vs. predicted values, and overlay the regression fit line."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
                    "\n",
                    "# Plot 1: Predictor X vs Target Y (Fitted Regression Line)\n",
                    "axes[0].scatter(X_train, y_train, color='royalblue', alpha=0.7, label='Train Data')\n",
                    "axes[0].scatter(X_test, y_test, color='darkorange', alpha=0.9, s=80, marker='s', label='Test Data')\n",
                    "\n",
                    "# Overlay regression line using parameters\n",
                    "x_line = np.linspace(df['package_exp'].min(), df['package_exp'].max(), 100)\n",
                    "y_line = b0_sklearn + b1_sklearn * x_line\n",
                    "axes[0].plot(x_line, y_line, color='crimson', linewidth=2.5, label=f'Fit Line: Y = {b0_sklearn:.4f} + {b1_sklearn:.4f}*X')\n",
                    "axes[0].set_title('Regression Line: Package Expectation vs GPA', fontsize=12, fontweight='bold')\n",
                    "axes[0].set_xlabel('Package Expectation (LPA)', fontsize=11)\n",
                    "axes[0].set_ylabel('GPA', fontsize=11)\n",
                    "axes[0].legend(loc='lower right')\n",
                    "\n",
                    "# Plot 2: Actual vs Predicted GPA (Testing Set)\n",
                    "axes[1].scatter(y_test, y_pred_test, color='darkorange', s=100, edgecolors='black', alpha=0.8, label='Predictions')\n",
                    "# Ideal diagonal line (y=x)\n",
                    "ideal_min = min(y_test.min(), y_pred_test.min()) - 0.1\n",
                    "ideal_max = max(y_test.max(), y_pred_test.max()) + 0.1\n",
                    "axes[1].plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='grey', linestyle='--', label='Ideal Fit (Actual = Predicted)')\n",
                    "axes[1].set_title('Actual vs. Predicted GPA (Test Set)', fontsize=12, fontweight='bold')\n",
                    "axes[1].set_xlabel('Actual GPA', fontsize=11)\n",
                    "axes[1].set_ylabel('Predicted GPA', fontsize=11)\n",
                    "axes[1].legend(loc='upper left')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('gpa_package_regression.png', dpi=300)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part B: Manual OLS Parameter Derivation\n",
                    "\n",
                    "Using only NumPy on the training set, we calculate the slope ($b_1$) and intercept ($b_0$) using the Ordinary Least Squares (OLS) formulas:\n",
                    "\n",
                    "$$b_1 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}$$\n",
                    "$$b_0 = \\bar{y} - b_1 \\bar{x}$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Flatten train arrays to 1D for NumPy computation\n",
                    "x_train_flat = X_train.flatten()\n",
                    "y_train_flat = y_train.flatten()\n",
                    "\n",
                    "# Calculate means\n",
                    "mean_x = np.mean(x_train_flat)\n",
                    "mean_y = np.mean(y_train_flat)\n",
                    "\n",
                    "# Compute terms\n",
                    "numerator = np.sum((x_train_flat - mean_x) * (y_train_flat - mean_y))\n",
                    "denominator = np.sum((x_train_flat - mean_x) ** 2)\n",
                    "\n",
                    "# Compute manual slope and intercept\n",
                    "b1_manual = numerator / denominator\n",
                    "b0_manual = mean_y - b1_manual * mean_x\n",
                    "\n",
                    "print(\"=== Manual OLS Model Parameters ===\")\n",
                    "print(f\"Intercept (b0): {b0_manual:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_manual:.6f}\")\n",
                    "\n",
                    "# Verification\n",
                    "print(\"\\n--- Parameter Equivalence Checks ---\")\n",
                    "print(f\"Intercept Difference: {abs(b0_sklearn - b0_manual):.12f}\")\n",
                    "print(f\"Slope Difference:     {abs(b1_sklearn - b1_manual):.12f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Coefficient and Parameter Equivalence Analysis\n",
                    "\n",
                    "Comparing the Scikit-learn results and Manual OLS parameters side-by-side:\n",
                    "\n",
                    "| Parameter | Scikit-Learn | Manual OLS | Difference |\n",
                    "| :--- | :--- | :--- | :--- |\n",
                    "| Intercept ($b_0$) | **{{b0_sklearn}}** | **{{b0_manual}}** | 0.000000 |\n",
                    "| Slope ($b_1$) | **{{b1_sklearn}}** | **{{b1_manual}}** | 0.000000 |\n",
                    "\n",
                    "The parameters computed by Scikit-Learn and the manual NumPy derivation on the training set are mathematically identical up to the numerical precision limit. This demonstrates that Scikit-Learn's `LinearRegression` implements the exact closed-form Ordinary Least Squares solution."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part C: Save Model Parameters\n",
                    "\n",
                    "We persist the parameters `intercept`, `slope`, `r2_train`, and `r2_test` to `model_params.json`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "params = {\n",
                    "    \"intercept\": float(b0_manual),\n",
                    "    \"slope\": float(b1_manual),\n",
                    "    \"r2_train\": float(r2_train),\n",
                    "    \"r2_test\": float(r2_test)\n",
                    "}\n",
                    "\n",
                    "with open('model_params.json', 'w') as f:\n",
                    "    json.dump(params, f, indent=4)\n",
                    "print(\"Successfully saved model parameters to 'model_params.json'\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Data Scientist Model Interpretation and Evaluation\n",
                    "\n",
                    "### 1. Variables Selection (GPA as Y, Package Expectations as X)\n",
                    "- **Y (GPA)** is the academic response/target variable.\n",
                    "- **X (package_exp)** is the package expectation predictor.\n",
                    "- Under a classical career-placement hypothesis, a student's current academic performance (GPA) might serve as a foundation that enables them to demand a higher placement package. Thus, conceptually, GPA leads to package expectation. \n",
                    "- However, for this regression task, we model GPA as a function of package expectations ($GPA = b_0 + b_1 \\times package\\_exp$). This lets us analyze if a student's high package confidence/career aspiration ($X$) exhibits a linear relationship with their academic performance ($Y$).\n",
                    "\n",
                    "### 2. Slope Interpretation in Plain English\n",
                    "- The slope ($b_1$) represents the change in GPA associated with a 1 LPA (Lakh Per Annum) increase in package expectations.\n",
                    "- For example, if the slope is positive, say $b_1 = 0.005$, this means that **for every 1 LPA increase in expected salary package, the student's GPA is estimated to increase by 0.005 points**.\n",
                    "- Alternatively, if the slope is negative, it indicates that higher package expectations are associated with lower GPAs.\n",
                    "\n",
                    "### 3. Model Reliability and R-squared ($R^2$)\n",
                    "- The $R^2$ value measures the proportion of variance in the dependent variable (GPA) that is predictable from the independent variable (package expectations).\n",
                    "- We will review the calculated $R^2$ values on the training and testing sets. In behavioral student surveys, data is highly noisy due to subjective factors. A low $R^2$ indicates that salary expectations are not a strong predictor of GPA, meaning there is high residual variance and the model should not be trusted for precise GPA predictions."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open('regression_gpa_package.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated regression_gpa_package.ipynb")

if __name__ == '__main__':
    generate_notebook()
