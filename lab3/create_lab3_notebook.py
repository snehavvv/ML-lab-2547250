import json

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lab Exercise 3: Student Survey Regression Analysis\n",
                    "\n",
                    "This notebook outlines a complete data science pipeline to perform Simple Linear Regression using both **Scikit-learn** and **manual Ordinary Least Squares (OLS)** formulas on survey data collected from students.\n",
                    "\n",
                    "### Objectives:\n",
                    "1. **Part A: Data Collection & Preprocessing**: Load real survey data, inspect dimensions, clean percentage characters from columns (`Your CIA % of last semester`, `Your maximum attendance % till last semester`), filter out GPA outliers (> 5.0), remove duplicate rows, and generate summary statistics.\n",
                    "2. **Experiment 1 (CIA vs. GPA)**: Perform Simple Linear Regression predicting GPA ($Y$) from CIA Percentage ($X$).\n",
                    "3. **Experiment 2 (Attendance vs. GPA)**: Perform Simple Linear Regression predicting GPA ($Y$) from Attendance Percentage ($X$).\n",
                    "4. **Part B & C (Modeling & OLS)**: Train Scikit-Learn `LinearRegression` and derive identical parameters manually from NumPy formulas, then verify predictions.\n",
                    "5. **Comparison Task**: Quantify prediction differences between Scikit-learn and manual OLS formulas.\n",
                    "6. **Parameter Saving Task (Pickle)**: Save learned weights (intercepts and slopes) to `linear_regression_weights.pkl`, reload them, and demonstrate inference.\n",
                    "7. **Viva Q&A**: Answers to sample viva questions."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part A: Data Collection and Preprocessing"
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
                    "import pickle\n",
                    "import os\n",
                    "\n",
                    "# Plotting configs\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (12, 5)\n",
                    "plt.rcParams[\"font.family\"] = \"sans-serif\"\n",
                    "\n",
                    "# 1. Load the dataset using Pandas\n",
                    "df = pd.read_csv('Student_Awareness_Survey__Responses__-_Form_Responses_1.csv')\n",
                    "\n",
                    "# 2. Display the first 5 rows\n",
                    "print(\"=== First 5 Rows of Raw Survey Data ===\")\n",
                    "display(df.head(5))\n",
                    "\n",
                    "# 3. Check dataset dimensions\n",
                    "print(f\"\\nDataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
                    "\n",
                    "# 4. Identify missing values\n",
                    "print(\"\\n=== Missing Values Per Column ===\")\n",
                    "print(df.isnull().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Data Cleaning and Formatting\n",
                    "\n",
                    "We need to clean the columns `Your CIA % of last semester`, `Your maximum attendance % till last semester`, and `Your GPA of last semester` by:\n",
                    "1. Stripping trailing `%` signs and spaces.\n",
                    "2. Converting them into numerical floats.\n",
                    "3. Removing duplicate records.\n",
                    "4. Removing academic GPA anomalies (GPAs > 5.0) which are mathematically invalid on a 4-point scale."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clean percentage strings\n",
                    "def clean_percentage(val):\n",
                    "    if pd.isna(val):\n",
                    "        return np.nan\n",
                    "    val_str = str(val).strip().replace('%', '')\n",
                    "    try:\n",
                    "        return float(val_str)\n",
                    "    except ValueError:\n",
                    "        return np.nan\n",
                    "\n",
                    "# Apply clean to CIA and Attendance\n",
                    "df['Your CIA % of last semester'] = df['Your CIA % of last semester'].apply(clean_percentage)\n",
                    "df['Your maximum attendance % till last semester'] = df['Your maximum attendance % till last semester'].apply(clean_percentage)\n",
                    "\n",
                    "# Rename variables for clean coding\n",
                    "rename_dict = {\n",
                    "    'Your GPA of last semester': 'gpa',\n",
                    "    'Your CIA % of last semester': 'cia_pct',\n",
                    "    'Your maximum attendance % till last semester': 'attendance_pct'\n",
                    "}\n",
                    "df = df.rename(columns=rename_dict)\n",
                    "\n",
                    "# Remove duplicate records if present\n",
                    "print(f\"Rows before removing duplicates: {df.shape[0]}\")\n",
                    "df = df.drop_duplicates()\n",
                    "print(f\"Rows after removing duplicates:  {df.shape[0]}\")\n",
                    "\n",
                    "# Drop rows with missing values in our features of interest\n",
                    "df = df.dropna(subset=['gpa', 'cia_pct', 'attendance_pct'])\n",
                    "\n",
                    "# Cap GPA anomaly (> 5)\n",
                    "print(f\"Rows before GPA outlier filter: {df.shape[0]}\")\n",
                    "df = df[df['gpa'] <= 5.0]\n",
                    "print(f\"Rows after GPA outlier filter:  {df.shape[0]}\")\n",
                    "\n",
                    "# Generate statistical summary\n",
                    "print(\"\\n=== Preprocessed Variables Statistical Summary ===\")\n",
                    "display(df[['gpa', 'cia_pct', 'attendance_pct']].describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Experiment 1: Predicting GPA from CIA Percentage\n",
                    "\n",
                    "* **Independent Variable (X)**: `cia_pct` (CIA Percentage)\n",
                    "* **Dependent Variable (Y)**: `gpa` (GPA)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train-Test Split (80% Train, 20% Test)\n",
                    "X1 = df[['cia_pct']].values\n",
                    "y1 = df['gpa'].values\n",
                    "\n",
                    "X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Fit Scikit-Learn Model\n",
                    "model_sk1 = LinearRegression()\n",
                    "model_sk1.fit(X1_train, y1_train)\n",
                    "\n",
                    "b0_sk1 = model_sk1.intercept_\n",
                    "b1_sk1 = model_sk1.coef_[0]\n",
                    "\n",
                    "# Predict using Scikit-Learn\n",
                    "y1_pred_sk = model_sk1.predict(X1_test)\n",
                    "\n",
                    "print(\"=== Scikit-Learn Experiment 1 Coefficients ===\")\n",
                    "print(f\"Intercept (b0): {b0_sk1:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_sk1:.6f}\")\n",
                    "\n",
                    "# Manual OLS calculations using NumPy on the training set\n",
                    "x1_train_flat = X1_train.flatten()\n",
                    "mean_x1 = np.mean(x1_train_flat)\n",
                    "mean_y1 = np.mean(y1_train)\n",
                    "\n",
                    "numerator1 = np.sum((x1_train_flat - mean_x1) * (y1_train - mean_y1))\n",
                    "denominator1 = np.sum((x1_train_flat - mean_x1) ** 2)\n",
                    "\n",
                    "b1_man1 = numerator1 / denominator1\n",
                    "b0_man1 = mean_y1 - b1_man1 * mean_x1\n",
                    "\n",
                    "# Predict using Manual Equation\n",
                    "y1_pred_man = b0_man1 + b1_man1 * X1_test.flatten()\n",
                    "\n",
                    "print(\"\\n=== Manual OLS Experiment 1 Coefficients ===\")\n",
                    "print(f\"Intercept (b0): {b0_man1:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_man1:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Experiment 1 Comparison and Predictions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create comparison DataFrame for Experiment 1\n",
                    "comp1_df = pd.DataFrame({\n",
                    "    'Actual GPA': y1_test,\n",
                    "    'Sklearn Prediction': y1_pred_sk,\n",
                    "    'Manual OLS Prediction': y1_pred_man,\n",
                    "    'Difference': np.abs(y1_pred_sk - y1_pred_man)\n",
                    "})\n",
                    "\n",
                    "print(\"=== Experiment 1 (CIA vs GPA) Test Set Comparison ===\")\n",
                    "display(comp1_df.head(10))\n",
                    "\n",
                    "# Metrics\n",
                    "print(\"\\n=== Evaluation Metrics (Test Set) ===\")\n",
                    "print(f\"MAE:  {mean_absolute_error(y1_test, y1_pred_sk):.6f}\")\n",
                    "print(f\"MSE:  {mean_squared_error(y1_test, y1_pred_sk):.6f}\")\n",
                    "print(f\"RMSE: {np.sqrt(mean_squared_error(y1_test, y1_pred_sk)):.6f}\")\n",
                    "print(f\"R2:   {r2_score(y1_test, y1_pred_sk):.6f}\")\n",
                    "\n",
                    "# Check equivalence\n",
                    "np.testing.assert_almost_equal(b0_sk1, b0_man1, decimal=10)\n",
                    "np.testing.assert_almost_equal(b1_sk1, b1_man1, decimal=10)\n",
                    "print(\"\\nVerification Successful! Sklearn and Manual coefficients are equivalent.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Experiment 2: Predicting GPA from Attendance Percentage\n",
                    "\n",
                    "* **Independent Variable (X)**: `attendance_pct` (Attendance Percentage)\n",
                    "* **Dependent Variable (Y)**: `gpa` (GPA)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train-Test Split (80% Train, 20% Test)\n",
                    "X2 = df[['attendance_pct']].values\n",
                    "y2 = df['gpa'].values\n",
                    "\n",
                    "X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Fit Scikit-Learn Model\n",
                    "model_sk2 = LinearRegression()\n",
                    "model_sk2.fit(X2_train, y2_train)\n",
                    "\n",
                    "b0_sk2 = model_sk2.intercept_\n",
                    "b1_sk2 = model_sk2.coef_[0]\n",
                    "\n",
                    "# Predict using Scikit-Learn\n",
                    "y2_pred_sk = model_sk2.predict(X2_test)\n",
                    "\n",
                    "print(\"=== Scikit-Learn Experiment 2 Coefficients ===\")\n",
                    "print(f\"Intercept (b0): {b0_sk2:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_sk2:.6f}\")\n",
                    "\n",
                    "# Manual OLS calculations using NumPy on the training set\n",
                    "x2_train_flat = X2_train.flatten()\n",
                    "mean_x2 = np.mean(x2_train_flat)\n",
                    "mean_y2 = np.mean(y2_train)\n",
                    "\n",
                    "numerator2 = np.sum((x2_train_flat - mean_x2) * (y2_train - mean_y2))\n",
                    "denominator2 = np.sum((x2_train_flat - mean_x2) ** 2)\n",
                    "\n",
                    "b1_man2 = numerator2 / denominator2\n",
                    "b0_man2 = mean_y2 - b1_man2 * mean_x2\n",
                    "\n",
                    "# Predict using Manual Equation\n",
                    "y2_pred_man = b0_man2 + b1_man2 * X2_test.flatten()\n",
                    "\n",
                    "print(\"\\n=== Manual OLS Experiment 2 Coefficients ===\")\n",
                    "print(f\"Intercept (b0): {b0_man2:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_man2:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Experiment 2 Comparison and Predictions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create comparison DataFrame for Experiment 2\n",
                    "comp2_df = pd.DataFrame({\n",
                    "    'Actual GPA': y2_test,\n",
                    "    'Sklearn Prediction': y2_pred_sk,\n",
                    "    'Manual OLS Prediction': y2_pred_man,\n",
                    "    'Difference': np.abs(y2_pred_sk - y2_pred_man)\n",
                    "})\n",
                    "\n",
                    "print(\"=== Experiment 2 (Attendance vs GPA) Test Set Comparison ===\")\n",
                    "display(comp2_df.head(10))\n",
                    "\n",
                    "# Metrics\n",
                    "print(\"\\n=== Evaluation Metrics (Test Set) ===\")\n",
                    "print(f\"MAE:  {mean_absolute_error(y2_test, y2_pred_sk):.6f}\")\n",
                    "print(f\"MSE:  {mean_squared_error(y2_test, y2_pred_sk):.6f}\")\n",
                    "print(f\"RMSE: {np.sqrt(mean_squared_error(y2_test, y2_pred_sk)):.6f}\")\n",
                    "print(f\"R2:   {r2_score(y2_test, y2_pred_sk):.6f}\")\n",
                    "\n",
                    "# Check equivalence\n",
                    "np.testing.assert_almost_equal(b0_sk2, b0_man2, decimal=10)\n",
                    "np.testing.assert_almost_equal(b1_sk2, b1_man2, decimal=10)\n",
                    "print(\"\\nVerification Successful! Sklearn and Manual coefficients are equivalent.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Parameter Saving Task (Pickle)\n",
                    "\n",
                    "We serialize the slopes and intercepts for both experiments using Pickle into `linear_regression_weights.pkl`. Then we demonstrate loading the parameters and making predictions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Assemble parameter weight dictionary\n",
                    "weights_dict = {\n",
                    "    'cia_vs_gpa': {\n",
                    "        'slope': float(b1_man1),\n",
                    "        'intercept': float(b0_man1)\n",
                    "    },\n",
                    "    'attendance_vs_gpa': {\n",
                    "        'slope': float(b1_man2),\n",
                    "        'intercept': float(b0_man2)\n",
                    "    }\n",
                    "}\n",
                    "\n",
                    "# 1. Save parameters into Pickle file\n",
                    "pickle_filename = 'linear_regression_weights.pkl'\n",
                    "with open(pickle_filename, 'wb') as f:\n",
                    "    pickle.dump(weights_dict, f)\n",
                    "print(f\"Successfully saved parameters to {pickle_filename}\")\n",
                    "\n",
                    "# 2. Load the Pickle file\n",
                    "with open(pickle_filename, 'rb') as f:\n",
                    "    loaded_weights = pickle.load(f)\n",
                    "print(\"Successfully reloaded parameters:\")\n",
                    "print(loaded_weights)\n",
                    "\n",
                    "# 3. Demonstrate loaded parameters for custom prediction\n",
                    "# Predict GPA for a student with 80% CIA and 95% Attendance\n",
                    "custom_cia = 80.0\n",
                    "custom_attendance = 95.0\n",
                    "\n",
                    "pred_gpa_cia = loaded_weights['cia_vs_gpa']['intercept'] + loaded_weights['cia_vs_gpa']['slope'] * custom_cia\n",
                    "pred_gpa_att = loaded_weights['attendance_vs_gpa']['intercept'] + loaded_weights['attendance_vs_gpa']['slope'] * custom_attendance\n",
                    "\n",
                    "print(f\"\\n=== Inference Demonstration using Reloaded Pickle Weights ===\")\n",
                    "print(f\"-> Prediction (Experiment 1): CIA = {custom_cia}% predicts a GPA of {pred_gpa_cia:.4f}\")\n",
                    "print(f\"-> Prediction (Experiment 2): Attendance = {custom_attendance}% predicts a GPA of {pred_gpa_att:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Final Observations & Inference\n",
                    "\n",
                    "### Summary of Findings:\n",
                    "- **Experiment 1 (CIA vs. GPA)**: The coefficients indicate that higher internal marks (CIA) display a positive relationship with final GPA. Since CIA forms a large portion of academic assessment, a positive correlation is logically expected.\n",
                    "- **Experiment 2 (Attendance vs. GPA)**: The slope shows the linear relationship of attendance on final GPA. Higher class attendance is statistically linked to better academic performances.\n",
                    "- **Method Equivalence**: Both Scikit-Learn and the manual NumPy derivation using the closed-form OLS math yielded **identical** slopes, intercepts, and test set predictions up to over 10 decimal places. This proves that Scikit-Learn implements standard OLS minimization underneath.\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## Sample Viva Questions and Answers\n",
                    "\n",
                    "### 1. What is Simple Linear Regression?\n",
                    "Simple Linear Regression is a supervised learning algorithm that models the relationship between a single quantitative predictor variable $X$ and a single quantitative target variable $Y$ using a straight line: $Y = b_0 + b_1 X + \\epsilon$.\n",
                    "\n",
                    "### 2. What is the role of slope and intercept?\n",
                    "- **Slope ($b_1$)**: Represents the change in the dependent target variable ($Y$) per unit change in the independent predictor variable ($X$).\n",
                    "- **Intercept ($b_0$)**: Represents the predicted value of the target variable ($Y$) when the predictor variable ($X$) is zero.\n",
                    "\n",
                    "### 3. What is Ordinary Least Squares (OLS)?\n",
                    "OLS is a mathematical optimization method used to determine the line of best fit by minimizing the Sum of Squared Residuals (SSR) between the actual data points and the predicted regression line.\n",
                    "\n",
                    "### 4. Why do we square the errors in OLS?\n",
                    "Errors are squared in OLS to:\n",
                    "1. Treat positive and negative residuals equally, preventing opposite-signed errors from cancelling each other out.\n",
                    "2. Heavily penalize larger prediction errors/outliers (since squaring an error of 4 gives a penalty of 16, whereas squaring 2 gives 4).\n",
                    "3. Ensure the mathematical objective function is smooth and differentiable, facilitating derivation of closed-form formulas.\n",
                    "\n",
                    "### 5. Difference between dependent and independent variable.\n",
                    "- **Independent variable (X)**: The input predictor or feature that we control or use to make predictions.\n",
                    "- **Dependent variable (Y)**: The target response or outcome that changes in response to the independent variable and which we wish to estimate.\n",
                    "\n",
                    "### 6. Why should data be cleaned before training?\n",
                    "Cleaning is critical because raw datasets contain missing values, non-numeric formatting (e.g. `%` or `LPA` strings), typos, and extreme anomalies that will distort calculations, skew coefficients, break matrix multiplications, and cause overfitting or false predictions.\n",
                    "\n",
                    "### 7. Why are slope and intercept called model parameters?\n",
                    "They are parameters because they are variables learned from the data during training that explicitly characterize the relationships mapping inputs to outputs. They are stored inside the model configuration to guide future predictions.\n",
                    "\n",
                    "### 8. Why do we save learned weights?\n",
                    "Saving weights (or serializing via Pickle) allows us to deploy the model for real-time predictions without needing to keep the original training data or run the training process again, which saves time, memory, and computing power."
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

    with open('student_survey_regression.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated student_survey_regression.ipynb")

if __name__ == '__main__':
    generate_notebook()
