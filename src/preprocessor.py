import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from .config import Config

class StudentPreprocessor:
    """Consolidated preprocessor for both small-scale and large-scale educational datasets."""
    
    @staticmethod
    def process_uci(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Specific logic for UCI Student Performance dataset."""
        df = df.copy().drop_duplicates()
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.median(numeric_only=True))

        # Categorical Encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        # Feature Engineering: Assignment Specific Metrics
        # Engagement Ratio: Study time relative to leisure
        df['engagement_ratio'] = df['studytime_mat'] / (df['freetime_mat'] + df['goout_mat'] + 0.1)
        
        # Consistency Index: Across Math/Portuguese
        df['consistency_index'] = 1 - (abs(df['G3_mat'] - df['G3_por']) / 20)
        
        # Mastery Trend: G2 - G1
        df['mastery_trend'] = df['G2_mat'] - df['G1_mat']
        
        # Absence Impact
        df['absence_impact'] = df['absences_mat'] + df['absences_por']

        # Target Definition (Pass >= 10)
        df[Config.TARGET_COLUMN] = (df['G3_mat'] >= 10).astype(int)
        
        X = df.drop(['G3_mat', 'G3_por', Config.TARGET_COLUMN], axis=1)
        y = df[Config.TARGET_COLUMN]
        return X, y

    @staticmethod
    def process_bridge(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Specific logic for KDD Cup / Bridge to Algebra dataset with advanced feature engineering."""
        df = df.copy()
        
        # 1. Target Definition
        df[Config.TARGET_COLUMN] = pd.to_numeric(df['Correct First Attempt'], errors='coerce').fillna(0).astype(int)
        
        # 2. Advanced Feature Engineering
        # Engagement Ratio: Frequency of interactions per student
        df['engagement_ratio'] = df.groupby('Anon Student Id')['Anon Student Id'].transform('count')
        
        # Consistency Index: Stability in response time (lower variance = higher consistency)
        # Using 1 / (std + 1) to normalize
        df['consistency_index'] = 1 / (df.groupby('Anon Student Id')['Step Duration (sec)'].transform('std').fillna(0) + 1)
        
        # Mastery Trend: Cumulative success rate for each student
        df = df.sort_values(['Anon Student Id', 'Step Start Time'])
        df['mastery_trend'] = df.groupby('Anon Student Id')[Config.TARGET_COLUMN].transform(lambda x: x.expanding().mean())
        
        # Traditional Features
        df['student_ability'] = df.groupby('Anon Student Id')[Config.TARGET_COLUMN].transform('mean')
        df['problem_difficulty'] = df.groupby('Problem Name')[Config.TARGET_COLUMN].transform('mean')
        
        # 3. Categorical Handling
        df['Problem Hierarchy'] = df['Problem Hierarchy'].astype('category').cat.codes
        
        # 4. Feature Selection
        feature_cols = [
            'student_ability', 'problem_difficulty', 'Problem Hierarchy', 
            'Step Duration (sec)', 'engagement_ratio', 'consistency_index', 'mastery_trend'
        ]
        X = df[feature_cols].copy()
        X = X.fillna(X.median())
        
        y = df[Config.TARGET_COLUMN].reset_index(drop=True)
        return X, y
