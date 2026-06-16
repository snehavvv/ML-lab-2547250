import json

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lab 3: Simple Linear Regression on Student Survey Data\n",
                    "\n",
                    "This notebook demonstrates a complete end-to-end Machine Learning pipeline for Simple Linear Regression. \n",
                    "\n",
                    "### Objectives:\n",
                    "1. Load and clean raw survey data (handling missing values and outliers).\n",
                    "2. Perform Simple Linear Regression using `scikit-learn`.\n",
                    "3. Manually compute the regression parameters (slope and intercept) using Ordinary Least Squares (OLS) formulas.\n",
                    "4. Compare the coefficients, predictions, and evaluation metrics ($R^2$ and MSE) from both approaches.\n",
                    "5. Plot and visualize the data and regression line.\n",
                    "6. Save and reload the model parameters for future inference."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Ingesting Libraries and Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.metrics import mean_squared_error, r2_score\n",
                    "import joblib\n",
                    "import json\n",
                    "\n",
                    "# Set aesthetic style for plotting\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                    "plt.rcParams[\"font.family\"] = \"sans-serif\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the raw survey dataset\n",
                    "raw_df = pd.read_csv('student_survey_raw.csv')\n",
                    "\n",
                    "print(\"=== Raw Dataset Inspection ===\")\n",
                    "print(f\"Shape: {raw_df.shape}\")\n",
                    "print(\"\\nFirst 5 rows:\")\n",
                    "print(raw_df.head())\n",
                    "\n",
                    "print(\"\\nMissing values per column:\")\n",
                    "print(raw_df.isnull().sum())\n",
                    "\n",
                    "print(\"\\nDescriptive statistics of raw data:\")\n",
                    "print(raw_df.describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Data Preprocessing and Cleaning\n",
                    "\n",
                    "To clean the dataset, we will:\n",
                    "1. Drop rows with missing values (`NaN`) in our variables of interest (`Study_Hours` and `Exam_Score`).\n",
                    "2. Identify and filter out extreme outliers. For realistic student study hours, study hours per week should be between 0 and 60 hours. Exam scores must lie within $[0, 100]$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Drop missing values in the variables we want to model\n",
                    "clean_df = raw_df.dropna(subset=['Study_Hours', 'Exam_Score']).copy()\n",
                    "print(f\"Shape after dropping NaNs: {clean_df.shape}\")\n",
                    "\n",
                    "# 2. Filter outliers\n",
                    "# Study hours per week should realistically be between 0 and 60 hours\n",
                    "# Exam score should be between 0 and 100\n",
                    "clean_df = clean_df[\n",
                    "    (clean_df['Study_Hours'] >= 0) & \n",
                    "    (clean_df['Study_Hours'] <= 60) & \n",
                    "    (clean_df['Exam_Score'] >= 0) & \n",
                    "    (clean_df['Exam_Score'] <= 100)\n",
                    "]\n",
                    "\n",
                    "print(f\"Shape after filtering outliers: {clean_df.shape}\")\n",
                    "print(\"\\nSummary statistics of clean dataset:\")\n",
                    "print(clean_df.describe())\n",
                    "\n",
                    "# Save clean dataset\n",
                    "clean_df.to_csv('student_survey_clean.csv', index=False)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's visualize the distribution of `Study_Hours` and `Exam_Score` to ensure no major anomalies remain."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "sns.histplot(clean_df['Study_Hours'], kde=True, ax=axes[0], color='royalblue')\n",
                    "axes[0].set_title('Cleaned Study Hours Distribution')\n",
                    "axes[0].set_xlabel('Study Hours / Week')\n",
                    "\n",
                    "sns.histplot(clean_df['Exam_Score'], kde=True, ax=axes[1], color='teal')\n",
                    "axes[1].set_title('Cleaned Exam Score Distribution')\n",
                    "axes[1].set_xlabel('Exam Score')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Simple Linear Regression using Scikit-Learn\n",
                    "\n",
                    "Now we perform Simple Linear Regression with `Study_Hours` as our independent predictor variable ($X$) and `Exam_Score` as our dependent target variable ($Y$)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Format variables for scikit-learn (X needs to be 2D)\n",
                    "X = clean_df[['Study_Hours']].values\n",
                    "y = clean_df['Exam_Score'].values\n",
                    "\n",
                    "# Initialize and fit model\n",
                    "sk_model = LinearRegression()\n",
                    "sk_model.fit(X, y)\n",
                    "\n",
                    "# Extract parameters\n",
                    "beta_1_sklearn = sk_model.coef_[0]\n",
                    "beta_0_sklearn = sk_model.intercept_\n",
                    "\n",
                    "print(\"=== Scikit-Learn Model Parameters ===\")\n",
                    "print(f\"Intercept (beta_0): {beta_0_sklearn:.6f}\")\n",
                    "print(f\"Slope/Coefficient (beta_1): {beta_1_sklearn:.6f}\")\n",
                    "\n",
                    "# Compute predictions and metrics\n",
                    "y_pred_sklearn = sk_model.predict(X)\n",
                    "mse_sklearn = mean_squared_error(y, y_pred_sklearn)\n",
                    "r2_sklearn = r2_score(y, y_pred_sklearn)\n",
                    "\n",
                    "print(f\"Mean Squared Error (MSE): {mse_sklearn:.6f}\")\n",
                    "print(f\"R-squared (R^2): {r2_sklearn:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Simple Linear Regression using Manual OLS Formulas\n",
                    "\n",
                    "The Ordinary Least Squares (OLS) formulas to calculate the parameters for the best-fit line $y = \beta_0 + \beta_1 x$ are:\n",
                    "\n",
                    "1. **Slope ($\beta_1$):**\n",
                    "   $$\\beta_1 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}$$\n",
                    "\n",
                    "2. **Intercept ($\beta_0$):**\n",
                    "   $$\\beta_0 = \\bar{y} - \\beta_1 \\bar{x}$$\n",
                    "\n",
                    "where:\n",
                    "- $x_i$ and $y_i$ are the individual data points for `Study_Hours` and `Exam_Score` respectively.\n",
                    "- $\\bar{x}$ and $\\bar{y}$ are the mean values of $X$ and $Y$ respectively.\n",
                    "- $n$ is the number of samples."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Flatten X for simpler manual computation\n",
                    "x_manual = clean_df['Study_Hours'].values\n",
                    "y_manual = clean_df['Exam_Score'].values\n",
                    "\n",
                    "# 1. Calculate means\n",
                    "mean_x = np.mean(x_manual)\n",
                    "mean_y = np.mean(y_manual)\n",
                    "\n",
                    "# 2. Calculate terms for beta_1\n",
                    "numerator = np.sum((x_manual - mean_x) * (y_manual - mean_y))\n",
                    "denominator = np.sum((x_manual - mean_x) ** 2)\n",
                    "\n",
                    "# 3. Compute coefficients\n",
                    "beta_1_manual = numerator / denominator\n",
                    "beta_0_manual = mean_y - beta_1_manual * mean_x\n",
                    "\n",
                    "print(\"=== Manual OLS Model Parameters ===\")\n",
                    "print(f\"Intercept (beta_0): {beta_0_manual:.6f}\")\n",
                    "print(f\"Slope/Coefficient (beta_1): {beta_1_manual:.6f}\")\n",
                    "\n",
                    "# 4. Calculate manual predictions and evaluation metrics\n",
                    "y_pred_manual = beta_0_manual + beta_1_manual * x_manual\n",
                    "n = len(y_manual)\n",
                    "\n",
                    "# Mean Squared Error (MSE)\n",
                    "mse_manual = np.sum((y_manual - y_pred_manual) ** 2) / n\n",
                    "\n",
                    "# Total Sum of Squares (TSS) and Residual Sum of Squares (RSS)\n",
                    "tss = np.sum((y_manual - mean_y) ** 2)\n",
                    "rss = np.sum((y_manual - y_pred_manual) ** 2)\n",
                    "\n",
                    "# R-squared (R^2)\n",
                    "r2_manual = 1 - (rss / tss)\n",
                    "\n",
                    "print(f\"Mean Squared Error (MSE): {mse_manual:.6f}\")\n",
                    "print(f\"R-squared (R^2): {r2_manual:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Comparing and Verifying Both Approaches\n",
                    "\n",
                    "Let's compare the parameters and metrics side-by-side and assert their mathematical equivalence."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comparison_df = pd.DataFrame({\n",
                    "    'Parameter/Metric': ['Intercept (beta_0)', 'Slope (beta_1)', 'MSE', 'R-squared'],\n",
                    "    'Scikit-Learn': [beta_0_sklearn, beta_1_sklearn, mse_sklearn, r2_sklearn],\n",
                    "    'Manual OLS': [beta_0_manual, beta_1_manual, mse_manual, r2_manual]\n",
                    "})\n",
                    "comparison_df['Difference'] = np.abs(comparison_df['Scikit-Learn'] - comparison_df['Manual OLS'])\n",
                    "print(\"=== Method Comparison Table ===\")\n",
                    "print(comparison_df.to_string(index=False))\n",
                    "\n",
                    "# Numerical check using assertion (up to 10 decimal places)\n",
                    "np.testing.assert_almost_equal(beta_0_sklearn, beta_0_manual, decimal=10)\n",
                    "np.testing.assert_almost_equal(beta_1_sklearn, beta_1_manual, decimal=10)\n",
                    "np.testing.assert_almost_equal(mse_sklearn, mse_manual, decimal=10)\n",
                    "np.testing.assert_almost_equal(r2_sklearn, r2_manual, decimal=10)\n",
                    "\n",
                    "print(\"\\nVerification Successful! Scikit-learn and manual OLS formulas yield identical results.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Visualizing the Best Fit Regression Line\n",
                    "\n",
                    "We will draw a scatter plot of the cleaned data points along with the fitted regression line, shading the distance (residuals) between some actual values and predicted values to visualize the meaning of Ordinary Least Squares."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.scatterplot(x=clean_df['Study_Hours'], y=clean_df['Exam_Score'], color='#2b5c8f', s=80, alpha=0.8, label='Students (Survey Data)')\n",
                    "\n",
                    "# Plotting regression line\n",
                    "x_line = np.linspace(clean_df['Study_Hours'].min(), clean_df['Study_Hours'].max(), 100)\n",
                    "y_line = beta_0_sklearn + beta_1_sklearn * x_line\n",
                    "plt.plot(x_line, y_line, color='#d95f02', linewidth=3, label=f'Regression Line: Y = {beta_0_sklearn:.2f} + {beta_1_sklearn:.2f}*X')\n",
                    "\n",
                    "# Visualizing residual lines for 5 random samples\n",
                    "np.random.seed(15)\n",
                    "sample_indices = np.random.choice(clean_df.index, size=5, replace=False)\n",
                    "for idx in sample_indices:\n",
                    "    xi = clean_df.loc[idx, 'Study_Hours']\n",
                    "    yi = clean_df.loc[idx, 'Exam_Score']\n",
                    "    yp = beta_0_sklearn + beta_1_sklearn * xi\n",
                    "    plt.vlines(xi, ymin=min(yi, yp), ymax=max(yi, yp), colors='purple', linestyles='dashed', alpha=0.7)\n",
                    "plt.plot([], [], 'p--', color='purple', alpha=0.7, label='Residuals (Errors)')\n",
                    "\n",
                    "plt.title('Study Hours vs. Exam Score - Simple Linear Regression', fontsize=16, fontweight='bold', pad=15)\n",
                    "plt.xlabel('Study Hours per Week', fontsize=12, labelpad=10)\n",
                    "plt.ylabel('Exam Score (out of 100)', fontsize=12, labelpad=10)\n",
                    "plt.legend(loc='upper left', fontsize=11, frameon=True)\n",
                    "plt.tight_layout()\n",
                    "\n",
                    "# Save figure\n",
                    "plt.savefig('regression_plot.png', dpi=300)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Persisting Model Parameters and Verification\n",
                    "\n",
                    "We will save:\n",
                    "1. The `scikit-learn` model object using `joblib`.\n",
                    "2. The manual parameters $\\beta_0$ and $\\beta_1$ as a JSON config file."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Save Scikit-Learn model\n",
                    "joblib.dump(sk_model, 'student_model.joblib')\n",
                    "print(\"Scikit-learn model successfully saved to 'student_model.joblib'\")\n",
                    "\n",
                    "# 2. Save manual coefficients to JSON\n",
                    "params = {\n",
                    "    'model_name': 'Simple Linear Regression - Student Survey Study Hours vs Exam Score',\n",
                    "    'intercept_beta_0': float(beta_0_manual),\n",
                    "    'slope_beta_1': float(beta_1_manual),\n",
                    "    'metrics': {\n",
                    "        'mse': float(mse_manual),\n",
                    "        'r2': float(r2_manual)\n",
                    "    }\n",
                    "}\n",
                    "\n",
                    "with open('model_parameters.json', 'w') as f:\n",
                    "    json.dump(params, f, indent=4)\n",
                    "print(\"Model parameters successfully saved to 'model_parameters.json'\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Reloading and Testing Inference\n",
                    "\n",
                    "Let's test both saved models on a student who studies 18.5 hours per week."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "test_hours = 18.5\n",
                    "\n",
                    "# 1. Scikit-learn reload & predict\n",
                    "loaded_sk_model = joblib.load('student_model.joblib')\n",
                    "pred_sk = loaded_sk_model.predict([[test_hours]])[0]\n",
                    "\n",
                    "# 2. Manual JSON parameters reload & predict\n",
                    "with open('model_parameters.json', 'r') as f:\n",
                    "    loaded_params = json.load(f)\n",
                    "\n",
                    "b0 = loaded_params['intercept_beta_0']\n",
                    "b1 = loaded_params['slope_beta_1']\n",
                    "pred_manual = b0 + b1 * test_hours\n",
                    "\n",
                    "print(f\"Inference comparison for {test_hours} study hours:\")\n",
                    "print(f\"-> Scikit-learn model prediction: {pred_sk:.4f}%\")\n",
                    "print(f\"-> Manual parameter prediction:     {pred_manual:.4f}%\")\n",
                    "print(f\"Difference: {abs(pred_sk - pred_manual):.12f}\")"
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

    with open('regression_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated regression_analysis.ipynb")

if __name__ == '__main__':
    generate_notebook()
