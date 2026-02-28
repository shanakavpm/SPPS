import pandas as pd
import numpy as np
from typing import Tuple
from .config import Config

class StudentPreprocessor:
    """Specialized preprocessor for the Bridge to Algebra large-scale educational dataset."""
    
    @staticmethod
    def process_bridge(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Specific logic for KDD Cup / Bridge to Algebra dataset with advanced feature engineering and synthetic metadata."""
        df = df.copy()
        
        # 1. Target Definition
        df[Config.TARGET_COLUMN] = pd.to_numeric(df['Correct First Attempt'], errors='coerce').fillna(0).astype(int)
        
        # 2. Synthetic Geography (For Temple requirements: 5 Maps)
        # Map students to school districts and coordinates
        districts = [f"District_{i}" for i in range(1, 11)]
        district_map = {sid: np.random.choice(districts) for sid in df['Anon Student Id'].unique()}
        df['geo_region'] = df['Anon Student Id'].map(district_map)
        
        # Coordinate mapping for hotspot maps (Shifted inland to avoid ocean)
        coords = {
            f"District_{i}": (34.15 + np.random.normal(0, 0.2), -117.80 + np.random.normal(0, 0.2)) 
            for i in range(1, 11)
        }
        df['latitude'] = df['geo_region'].map(lambda x: coords[x][0] + np.random.normal(0, 0.03))
        df['longitude'] = df['geo_region'].map(lambda x: coords[x][1] + np.random.normal(0, 0.03))

        # 3. Advanced Feature Engineering
        df['engagement_ratio'] = df.groupby('Anon Student Id')['Anon Student Id'].transform('count')
        df['consistency_index'] = 1 / (df.groupby('Anon Student Id')['Step Duration (sec)'].transform('std').fillna(0) + 1)
        
        # Mastery Trend: Cumulative success rate (vectorized for speed)
        df = df.sort_values(['Anon Student Id', 'Step Start Time'])
        groups = df.groupby(['Anon Student Id'])[Config.TARGET_COLUMN]
        df['mastery_trend'] = groups.cumsum() / (df.groupby('Anon Student Id').cumcount() + 1)
        
        # Timeliness: 1 if duration is below median (faster than average), else 0
        median_dur = df['Step Duration (sec)'].median()
        df['timeliness'] = (df['Step Duration (sec)'] < median_dur).astype(int)
        
        # Traditional Features
        df['student_ability'] = df.groupby('Anon Student Id')[Config.TARGET_COLUMN].transform('mean')
        df['problem_difficulty'] = df.groupby('Problem Name')[Config.TARGET_COLUMN].transform('mean')
        
        # 4. Network Metadata (For Template requirements: 5 Graphs)
        # Relationship: Student-KC mapping
        df['kc_node'] = df['KC(SubSkills)'].fillna('Unknown_KC')
        
        # 5. Categorical Handling
        df['Problem Hierarchy'] = df['Problem Hierarchy'].astype('category').cat.codes
        
        # 6. Feature Selection (Include Geo and Graph data for visuals)
        feature_cols = [
            'student_ability', 'problem_difficulty', 'Problem Hierarchy', 
            'Step Duration (sec)', 'engagement_ratio', 'consistency_index', 'mastery_trend',
            'timeliness', 'latitude', 'longitude', 'geo_region', 'kc_node', 'Anon Student Id'
        ]
        X = df[feature_cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        
        y = df[Config.TARGET_COLUMN].reset_index(drop=True)
        return X, y
