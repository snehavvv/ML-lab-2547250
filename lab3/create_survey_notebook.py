import json

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Data Cleaning and Preprocessing of Student Awareness Survey\n",
                    "\n",
                    "This notebook outlines a systematic pipeline as a **Data Analyst** to load, clean, and preprocess real-world survey data collected from Google Forms. \n",
                    "\n",
                    "### Objectives:\n",
                    "1. Load the raw survey dataset and inspect its structure, data types, and initial distribution.\n",
                    "2. Rename long, survey-generated column names to concise aliases.\n",
                    "3. Clean and standardize the package expectations column (`package_exp`) by handling string formatting, absolute rupee scaling, negative typos, and missing values using median imputation.\n",
                    "4. Identify and drop academic GPA anomalies (e.g., GPA > 5 on a 4-point scale).\n",
                    "5. Output comparison of null counts and descriptive statistics.\n",
                    "6. Save the cleaned dataset to `cleaned_survey.csv` for downstream regression modeling.\n",
                    "7. Document analysis decisions and justifications."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Load CSV and Inspect Raw Data\n",
                    "\n",
                    "We will load the Google Forms responses CSV file and view its dimensions, variable data types, and basic statistics."
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
                    "\n",
                    "# Load dataset\n",
                    "raw_df = pd.read_csv('Student_Awareness_Survey__Responses__-_Form_Responses_1.csv')\n",
                    "\n",
                    "# Display dataset size\n",
                    "print(f\"Dataset Shape: {raw_df.shape}\")\n",
                    "\n",
                    "# Display column names and data types\n",
                    "print(\"\\n--- Data Types ---\")\n",
                    "print(raw_df.dtypes)\n",
                    "\n",
                    "# Display summary statistics\n",
                    "print(\"\\n--- Summary Statistics of Raw Data ---\")\n",
                    "print(raw_df.describe(include='all'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Rename Long Column Names to Short Aliases\n",
                    "\n",
                    "Google Forms generates column headers matching the survey questions. We rename these to clean, concise variables to simplify data manipulation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mapping from long question text to clean variable names\n",
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
                    "\n",
                    "# Verify column names\n",
                    "print(\"Renamed Columns:\")\n",
                    "print(df.columns.tolist())\n",
                    "\n",
                    "# Display head of renamed dataset\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Clean and Impute 'package_exp'\n",
                    "\n",
                    "The `package_exp` column contains mixed text formats (e.g., `'6 LPA'`, `'10LPA'`), absolute rupees (e.g., `'1200000'`), and typo signs (e.g., `'-8'`). We clean it by:\n",
                    "1. Extracting only digits and decimal points (this automatically corrects negative signs to positive).\n",
                    "2. Converting values entered in absolute Rupees (>= 100,000) to Lakhs Per Annum (LPA) by dividing by 100,000.\n",
                    "3. Performing Median Imputation on any missing (`NaN`) values."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Null counts before cleaning\n",
                    "print(\"Null counts in package_exp before cleaning:\", df['package_exp'].isnull().sum())\n",
                    "print(\"Unique raw values in package_exp before cleaning:\")\n",
                    "print(df['package_exp'].unique())\n",
                    "\n",
                    "def clean_package_expression(val):\n",
                    "    if pd.isna(val):\n",
                    "        return np.nan\n",
                    "    \n",
                    "    # Convert to string and standard lower case\n",
                    "    val_str = str(val).strip().lower()\n",
                    "    \n",
                    "    # Extract numeric characters (digits and dots)\n",
                    "    cleaned_str = ''.join([c for c in val_str if c.isdigit() or c == '.'])\n",
                    "    if not cleaned_str:\n",
                    "        return np.nan\n",
                    "    \n",
                    "    try:\n",
                    "        num = float(cleaned_str)\n",
                    "        \n",
                    "        # If a student entered package in absolute currency (e.g., 1200000 for 12 LPA)\n",
                    "        if num >= 100000:\n",
                    "            num = num / 100000.0\n",
                    "            \n",
                    "        return num\n",
                    "    except ValueError:\n",
                    "        return np.nan\n",
                    "\n",
                    "# Clean values\n",
                    "df['package_exp'] = df['package_exp'].apply(clean_package_expression)\n",
                    "\n",
                    "# Calculate the median package expectation on valid values\n",
                    "median_package = df['package_exp'].median()\n",
                    "print(f\"\\nComputed Median package expectation (LPA): {median_package:.2f}\")\n",
                    "\n",
                    "# Impute missing values with the median\n",
                    "df['package_exp'] = df['package_exp'].fillna(median_package)\n",
                    "\n",
                    "# Verify cleaning output\n",
                    "print(\"\\nUnique cleaned and imputed values in package_exp:\")\n",
                    "print(df['package_exp'].unique())\n",
                    "print(\"Null counts in package_exp after cleaning:\", df['package_exp'].isnull().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Drop Academic GPA Outliers / Anomalies\n",
                    "\n",
                    "The GPA is measured on a standard 4-point scale at the college. A GPA value of `8.0` is impossible. We will identify and drop rows where GPA > 5.0."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Identify rows with GPA > 5.0\n",
                    "gpa_anomalies = df[df['gpa'] > 5]\n",
                    "print(\"--- GPA Anomalies (GPA > 5.0) ---\")\n",
                    "print(gpa_anomalies[['Registration Number', 'gpa', 'package_exp']])\n",
                    "\n",
                    "# Drop anomalies\n",
                    "print(f\"\\nDataset shape before dropping anomalies: {df.shape}\")\n",
                    "df = df[df['gpa'] <= 5]\n",
                    "print(f\"Dataset shape after dropping anomalies: {df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Before and After Null Counts & Descriptive Statistics\n",
                    "\n",
                    "Let's review the final null counts of our renamed variables and display their summary statistics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cols_to_check = ['gpa', 'package_exp', 'tech_rating', 'cia_pct', 'attendance_pct', 'min_sal', 'max_sal', 'median_sal']\n",
                    "\n",
                    "print(\"=== Missing (Null) Values of Key Variables ===\")\n",
                    "print(df[cols_to_check].isnull().sum())\n",
                    "\n",
                    "print(\"\\n=== Cleaned Descriptive Statistics ===\")\n",
                    "print(df[cols_to_check].describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Save Cleaned Dataset\n",
                    "\n",
                    "We output the clean dataset to `cleaned_survey.csv` for modeling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.to_csv('cleaned_survey.csv', index=False)\n",
                    "print(\"Cleaned dataframe successfully saved to 'cleaned_survey.csv'\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Data Analyst Justification of Decisions\n",
                    "\n",
                    "### 1. Renaming Headers\n",
                    "Google Forms headers represent complete question text. Concising them to short variables (`gpa`, `package_exp`, etc.) increases code readability and prevents syntactic typos in downstream operations.\n",
                    "\n",
                    "### 2. GPA Capping at 5.0 (Dropping rows where GPA > 5)\n",
                    "The academic GPA of the target student body operates on a 4-point scale (ranging from 0.0 to 4.0). One respondent recorded a GPA of `8.0`. This represents a clear anomaly. We dropped this row because:\n",
                    "- A GPA of 8.0 cannot exist on a 4-point scale, violating domain constraints.\n",
                    "- Keeping this row or attempting to guess the scale conversion (e.g. dividing by 2 to assume it was a 10-point scale) introduces unverified assumptions. Dropping the row is the standard and safest approach to maintain data integrity.\n",
                    "\n",
                    "### 3. Imputation Strategy (Median Imputation for `package_exp`)\n",
                    "For package expectations, we chose **Median Imputation** over the mean:\n",
                    "- Salary and salary-expectation distributions are classically right-skewed. A few students might have extremely high salary expectations (e.g. 100 LPA), which pull the mean upwards.\n",
                    "- The mean is highly sensitive to outliers, whereas the median represents the 50th percentile (the central tendency) and is robust. Imputing with the median prevents skewing our predictions.\n",
                    "\n",
                    "### 4. Cleaning `package_exp` String Formats and Absolute Numbers\n",
                    "- String formats: Values like `'6 LPA'`, `'8 LPA'`, and `'10LPA'` are cleaned by stripping alphabetic symbols, leaving only float-convertible strings.\n",
                    "- Negatives: Typo signs like `'-8'` are corrected to `8` through numeric extraction (ignoring the negative sign), which naturally matches valid positive salary ranges.\n",
                    "- Unit standardizations: Values entered in raw Rupees (like `1200000` or `1500000`) represent Lakhs scaled by 100,000. Identifying values $\\ge 100,000$ and dividing by $100,000$ scales them to Lakhs Per Annum (LPA), aligning them with the rest of the column."
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

    with open('student_survey_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated student_survey_analysis.ipynb")

if __name__ == '__main__':
    generate_notebook()
