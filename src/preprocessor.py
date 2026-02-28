import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from .config import Config

class StudentPreprocessor:
    """Responsible for cleaning, engineering components and encoding."""
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the raw data into features and target."""
        df = df.copy()
        
        # 1. Cleaning: Drop duplicates and handle missing values
        df = df.drop_duplicates()
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.median(numeric_only=True))
            
        # 2. Feature Engineering: Domain-specific metrics
        df = self._engineer_features(df)
        
        # 3. Categorical Encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = self.label_encoder.fit_transform(df[col])
            
        # 4. Target Definition (Pass/Fail)
        df[Config.TARGET_COLUMN] = (df['G3_mat'] >= 10).astype(int)
        
        # Define features (X) and target (y)
        X = df.drop(['G3_mat', 'G3_por', Config.TARGET_COLUMN], axis=1)
        y = df[Config.TARGET_COLUMN]
        
        return X, y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds derived attributes to help predictive modeling."""
        # Engagement Ratio: Study time relative to non-academic time
        df['engagement_ratio'] = df['studytime_mat'] / (df['freetime_mat'] + df['goout_mat'] + 0.1)
        
        # Consistency Index: Uniformity between Math and Portuguese performance
        df['consistency_index'] = 1 - (abs(df['G3_mat'] - df['G3_por']) / 20)
        
        # Mastery Trend: Growth from G1 to G2 for the primary subject (Math)
        # Using G2 since G3 is the target
        df['mastery_trend'] = df['G2_mat'] - df['G1_mat']
        
        # Absence Impact: Aggregated absenteeism across both disciplines
        df['absence_impact'] = df['absences_mat'] + df['absences_por']

        # NEW: Relative Engagement (Scale engagement by maximum observed)
        df['relative_studytime'] = df['studytime_mat'] / (df['studytime_mat'].max() + 0.1)
        
        # NEW: Interaction terms
        # Interaction between failures and absences (often highly predictive of risk)
        df['failure_absence_interaction'] = df['failures_mat'] * df['absences_mat']
        
        # Interaction between social life and mother's education (proxy for socioeconomic/social balance)
        df['social_edu_interaction'] = df['goout_mat'] * df['Medu']

        return df
