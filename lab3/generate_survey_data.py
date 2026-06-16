import numpy as np
import pandas as pd

def generate_data(num_samples=120):
    np.random.seed(42)
    
    # 1. Independent variable: Study_Hours (normally distributed around 15 hours, clipped to 2-30)
    study_hours = np.random.normal(loc=15, scale=5, size=num_samples)
    study_hours = np.clip(study_hours, 2, 30)
    
    # 2. Dependent variable: Exam_Score = 30 + 2.2 * study_hours + noise (normally distributed with scale 4)
    noise = np.random.normal(loc=0, scale=4, size=num_samples)
    exam_score = 30 + 2.2 * study_hours + noise
    exam_score = np.clip(exam_score, 0, 100)
    
    # 3. Screen_Time_Hours (hours/day) - extra variable
    screen_time = np.random.uniform(1, 6, size=num_samples)
    
    df = pd.DataFrame({
        'Student_ID': [f'STU{i+1:03d}' for i in range(num_samples)],
        'Study_Hours': study_hours,
        'Screen_Time_Hours': screen_time,
        'Exam_Score': exam_score
    })
    
    # Introduce some missing values (NaNs) to demonstrate cleaning
    sh_missing_idx = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    es_missing_idx = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    
    df.loc[sh_missing_idx, 'Study_Hours'] = np.nan
    df.loc[es_missing_idx, 'Exam_Score'] = np.nan
    
    # Introduce outliers / anomalies:
    # Extreme outlier in Study_Hours
    df.loc[10, 'Study_Hours'] = 999.0
    # Negative outlier in Exam_Score
    df.loc[20, 'Exam_Score'] = -50.0
    # Extreme negative outlier in Study_Hours
    df.loc[30, 'Study_Hours'] = -5.0
    
    return df

if __name__ == '__main__':
    df = generate_data()
    df.to_csv('student_survey_raw.csv', index=False)
    print("Successfully generated student_survey_raw.csv with 120 samples (including NaNs and outliers).")
