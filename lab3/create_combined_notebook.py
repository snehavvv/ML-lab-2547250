import json

def generate_combined_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Combined Student Survey: Data Preprocessing & Simple Linear Regression\n",
                    "\n",
                    "This combined notebook performs both the **Data Cleaning/Preprocessing** and the **Simple Linear Regression Modeling** in a single consolidated pipeline.\n",
                    "\n",
                    "### Notebook Sections:\n",
                    "1. **Data Load and Column Renaming**: Mapping Google Forms columns to clean handles.\n",
                    "2. **Data Preprocessing & Cleaning**: Standardizing expected package format, capping GPA anomalies, and median imputation.\n",
                    "3. **Train-Test Split**: Partitioning data 80% train and 20% test.\n",
                    "4. **Scikit-learn Simple Linear Regression**: Fitting the model to predict GPA from Package Expectations and calculating metrics.\n",
                    "5. **Manual OLS Derivation**: Computing regression parameters from scratch using NumPy formulas and verifying equivalence.\n",
                    "6. **Parameter Export & Visualization**: Exporting model coefficients and plotting actual vs. predicted values.\n",
                    "7. **Data Analyst Interpretations**: Justification of modeling and cleaning decisions."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 1: Load CSV and Rename Columns"
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
                    "# Set plots aesthetics\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (12, 5)\n",
                    "plt.rcParams[\"font.family\"] = \"sans-serif\"\n",
                    "\n",
                    "# 1. Load Raw CSV\n",
                    "raw_df = pd.read_csv('Student_Awareness_Survey__Responses__-_Form_Responses_1.csv')\n",
                    "print(f\"Raw dataset shape: {raw_df.shape}\")\n",
                    "\n",
                    "# 2. Define short aliases mapping\n",
                    "rename_dict = {\n",
                    "    'Your GPA of last semester': 'gpa',\n",
                    "    'What are your package expectations (LPA)': 'package_exp',\n",
                    "    'Rate your technical competencies': 'tech_rating',\n",
                    "    'Your CIA % of last semester': 'cia_pct',\n",
                    "    'Your maximum attendance % till last semester': 'attendance_pct',\n",
                    "    'What is the minimum salary of students placed through campus (In LPA..respond as a number)': 'min_sal',\n",
                    "    'What is the maximum salary of students placed through campus (In LPA..respond as a number)': 'max_sal',\n",
                    "    'What is the median salary of students placed through campus (In LPA..respond as a number)': 'median_sal'\n",
                    "}\n",
                    "\n",
                    "df = raw_df.rename(columns=rename_dict)\n",
                    "print(\"Renamed Columns:\", [c for c in df.columns if c in rename_dict.values()])\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 2: Data Preprocessing and Cleaning\n",
                    "\n",
                    "We apply cleaning rules:\n",
                    "1. Parse the strings in `package_exp` and convert entries containing absolute Rupees into LPA by dividing by 100,000.\n",
                    "2. Impute missing values in `package_exp` using the median value of expectations.\n",
                    "3. Filter out academic GPA anomalies (GPAs > 5.0) which violate scale domains."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clean package_exp strings and values\n",
                    "def clean_package_expression(val):\n",
                    "    if pd.isna(val):\n",
                    "        return np.nan\n",
                    "    val_str = str(val).strip().lower()\n",
                    "    # Extract digits and decimal point\n",
                    "    cleaned_str = ''.join([c for c in val_str if c.isdigit() or c == '.'])\n",
                    "    if not cleaned_str:\n",
                    "        return np.nan\n",
                    "    try:\n",
                    "        num = float(cleaned_str)\n",
                    "        # Scale absolute rupees entries\n",
                    "        if num >= 100000:\n",
                    "            num = num / 100000.0\n",
                    "        return num\n",
                    "    except ValueError:\n",
                    "        return np.nan\n",
                    "\n",
                    "df['package_exp'] = df['package_exp'].apply(clean_package_expression)\n",
                    "\n",
                    "# Median Imputation on expectations\n",
                    "median_package = df['package_exp'].median()\n",
                    "df['package_exp'] = df['package_exp'].fillna(median_package)\n",
                    "print(f\"Imputed package expectations with median: {median_package} LPA\")\n",
                    "\n",
                    "# Cap/Drop GPA Outliers\n",
                    "print(f\"Shape before dropping GPA anomaly: {df.shape}\")\n",
                    "df = df[df['gpa'] <= 5]\n",
                    "print(f\"Shape after dropping GPA anomaly: {df.shape}\")\n",
                    "\n",
                    "# Save cleaned dataset locally\n",
                    "df.to_csv('cleaned_survey.csv', index=False)\n",
                    "\n",
                    "# Show Null Counts of target columns\n",
                    "cols = ['gpa', 'package_exp', 'tech_rating', 'cia_pct', 'attendance_pct']\n",
                    "print(\"\\n--- Null Counts of Key Columns ---\")\n",
                    "print(df[cols].isnull().sum())\n",
                    "\n",
                    "print(\"\\n--- Cleaned Descriptive Statistics ---\")\n",
                    "print(df[cols].describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 3: Train-Test Split\n",
                    "\n",
                    "We split the cleaned dataset into an 80% training set and a 20% testing set to evaluate generalizability."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Predict GPA (Y) from expected salary package (X)\n",
                    "X = df[['package_exp']].values\n",
                    "y = df['gpa'].values\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "print(f\"Train sample count: {X_train.shape[0]}\")\n",
                    "print(f\"Test sample count:  {X_test.shape[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 4: Part A — Scikit-Learn Linear Regression\n",
                    "\n",
                    "We fit the Scikit-Learn `LinearRegression` model on the training set and display intercept ($b_0$), slope ($b_1$), and error metrics on the test set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sk_model = LinearRegression()\n",
                    "sk_model.fit(X_train, y_train)\n",
                    "\n",
                    "b0_sklearn = sk_model.intercept_\n",
                    "b1_sklearn = sk_model.coef_[0]\n",
                    "\n",
                    "print(\"=== Scikit-Learn Parameters ===\")\n",
                    "print(f\"Intercept (b0): {b0_sklearn:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_sklearn:.6f}\")\n",
                    "\n",
                    "# Inference\n",
                    "y_pred_train = sk_model.predict(X_train)\n",
                    "y_pred_test = sk_model.predict(X_test)\n",
                    "\n",
                    "# Test set evaluation metrics\n",
                    "mae = mean_absolute_error(y_test, y_pred_test)\n",
                    "mse = mean_squared_error(y_test, y_pred_test)\n",
                    "rmse = np.sqrt(mse)\n",
                    "r2_train = r2_score(y_train, y_pred_train)\n",
                    "r2_test = r2_score(y_test, y_pred_test)\n",
                    "\n",
                    "print(\"\\n=== Performance Metrics ===\")\n",
                    "print(f\"Mean Absolute Error (MAE):     {mae:.6f}\")\n",
                    "print(f\"Mean Squared Error (MSE):      {mse:.6f}\")\n",
                    "print(f\"Root Mean Squared Error (RMSE): {rmse:.6f}\")\n",
                    "print(f\"R-squared (R2) Train Set:      {r2_train:.6f}\")\n",
                    "print(f\"R-squared (R2) Test Set:       {r2_test:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 5: Part B — Manual OLS Derivation (NumPy)\n",
                    "\n",
                    "Using only NumPy on the training set, we calculate the OLS coefficients:\n",
                    "$$b_1 = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}$$\n",
                    "$$b_0 = \\bar{y} - b_1 \\bar{x}$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "x_train_flat = X_train.flatten()\n",
                    "y_train_flat = y_train\n",
                    "\n",
                    "# Compute Means\n",
                    "mean_x = np.mean(x_train_flat)\n",
                    "mean_y = np.mean(y_train_flat)\n",
                    "\n",
                    "# Coefficients calculations\n",
                    "num = np.sum((x_train_flat - mean_x) * (y_train_flat - mean_y))\n",
                    "den = np.sum((x_train_flat - mean_x) ** 2)\n",
                    "b1_manual = num / den\n",
                    "b0_manual = mean_y - b1_manual * mean_x\n",
                    "\n",
                    "print(\"=== Manual NumPy OLS Parameters ===\")\n",
                    "print(f\"Intercept (b0): {b0_manual:.6f}\")\n",
                    "print(f\"Slope (b1):     {b1_manual:.6f}\")\n",
                    "\n",
                    "# Confirm equivalence\n",
                    "print(\"\\n=== Verification Checks (Sklearn vs. Manual) ===\")\n",
                    "print(f\"Intercept Difference: {abs(b0_sklearn - b0_manual):.12f}\")\n",
                    "print(f\"Slope Difference:     {abs(b1_sklearn - b1_manual):.12f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 6: Part C — Parameter Export and Plotting\n",
                    "\n",
                    "We write model parameters to `model_params.json` and plot the fitted line overlaid on actual dataset splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Save coefficients\n",
                    "params = {\n",
                    "    \"intercept\": float(b0_manual),\n",
                    "    \"slope\": float(b1_manual),\n",
                    "    \"r2_train\": float(r2_train),\n",
                    "    \"r2_test\": float(r2_test)\n",
                    "}\n",
                    "with open('model_params.json', 'w') as f:\n",
                    "    json.dump(params, f, indent=4)\n",
                    "print(\"Saved parameters to 'model_params.json'\")\n",
                    "\n",
                    "# 2. Plotting regression\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
                    "\n",
                    "# Plot 1: Predictor vs Target (Fitted Line)\n",
                    "axes[0].scatter(X_train, y_train, color='royalblue', alpha=0.7, label='Train Data')\n",
                    "axes[0].scatter(X_test, y_test, color='darkorange', alpha=0.9, s=80, marker='s', label='Test Data')\n",
                    "\n",
                    "x_line = np.linspace(df['package_exp'].min(), df['package_exp'].max(), 100)\n",
                    "y_line = b0_sklearn + b1_sklearn * x_line\n",
                    "axes[0].plot(x_line, y_line, color='crimson', linewidth=2.5, label=f'Fit Line: Y = {b0_sklearn:.4f} + {b1_sklearn:.4f}*X')\n",
                    "axes[0].set_title('Combined regression plot: Expected Package vs. GPA', fontsize=12, fontweight='bold')\n",
                    "axes[0].set_xlabel('Package Expectation (LPA)', fontsize=11)\n",
                    "axes[0].set_ylabel('GPA', fontsize=11)\n",
                    "axes[0].legend(loc='lower right')\n",
                    "\n",
                    "# Plot 2: Actual vs Predicted GPA (Testing Set)\n",
                    "axes[1].scatter(y_test, y_pred_test, color='darkorange', s=100, edgecolors='black', alpha=0.8, label='Predicted')\n",
                    "ideal_min = min(y_test.min(), y_pred_test.min()) - 0.1\n",
                    "ideal_max = max(y_test.max(), y_pred_test.max()) + 0.1\n",
                    "axes[1].plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='grey', linestyle='--', label='Ideal prediction (y=x)')\n",
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
                    "## Section 7: Data Analyst / Data Scientist Interpretations\n",
                    "\n",
                    "### 1. Variables Selection (GPA as Y, Package Expectations as X)\n",
                    "- **Y (GPA)** is the academic response/target variable.\n",
                    "- **X (package_exp)** is the package expectation predictor.\n",
                    "- Conceptionally, a student's current academic performance (GPA) might serve as a foundation that enables them to demand a higher placement package. Thus, conceptually, GPA leads to package expectation. \n",
                    "- However, for this regression task, we model GPA as a function of package expectations ($GPA = b_0 + b_1 \\times package\\_exp$). This lets us analyze if a student's high package confidence/career aspiration ($X$) exhibits a linear relationship with their academic performance ($Y$).\n",
                    "\n",
                    "### 2. Slope Interpretation in Plain English\n",
                    "- The slope ($b_1 \\approx 0.007857$) represents the change in GPA associated with a 1 LPA (Lakh Per Annum) increase in package expectations.\n",
                    "- **In plain English, for every 1 LPA increase in expected salary package, the student's GPA is estimated to increase by 0.0079 points.** This represents a negligible impact.\n",
                    "\n",
                    "### 3. Model Reliability and R-squared ($R^2$)\n",
                    "- **$R^2$ Train**: $1.78\\%$\n",
                    "- **$R^2$ Test**: $-134.16\\%$\n",
                    "- The extremely low training $R^2$ indicates that expected package explains practically none of the variance in GPAs. \n",
                    "- The negative test $R^2$ shows that predicting using the regression line is worse than simply guessing the mean GPA of the test set. **Therefore, package expectations have no linear predictive value for GPAs, and the model is not strong enough to trust.**"
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

    with open('student_survey_regression_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated combined notebook student_survey_regression_analysis.ipynb")

if __name__ == '__main__':
    generate_combined_notebook()
