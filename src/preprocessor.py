import pandas as pd
import numpy as np
import logging
from typing import Tuple
from .config import Config

logger = logging.getLogger(__name__)


class StudentPreprocessor:
    """
    Rigorous preprocessing pipeline for the Bridge to Algebra educational dataset.

    Pipeline Steps:
        1. Duplicate Removal      — eliminate exact duplicate rows
        2. Target Definition      — define binary classification target
        3. Missing Value Handling  — median imputation (numeric), 'Unknown' (categorical)
        4. Categorical Encoding   — convert text categories to numerical codes
        5. Feature Engineering     — derive engagement, consistency, mastery, and timeliness
        6. Feature Selection       — retain only model-relevant columns
        7. Final Imputation        — fill any residual NaN with column medians
    """

    @staticmethod
    def process_bridge(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preprocessing for KDD Cup / Bridge to Algebra dataset.

        Returns:
            X (pd.DataFrame): Feature matrix with engineered attributes.
            y (pd.Series):    Binary target (1 = Correct First Attempt, 0 = Incorrect).
        """
        df = df.copy()

        # ── Step 1: Remove Duplicate Entries ──────────────────────────────────
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        dropped = initial_rows - len(df)
        logger.info(f"Duplicate removal: {dropped} duplicates dropped ({initial_rows} → {len(df)} rows)")

        # ── Step 2: Target Variable Definition ────────────────────────────────
        # Binary classification: 1 = student got the step correct on first try
        df[Config.TARGET_COLUMN] = (
            pd.to_numeric(df['Correct First Attempt'], errors='coerce')
            .fillna(0)
            .astype(int)
        )

        # ── Step 3: Missing Value Handling ────────────────────────────────────
        # Numeric columns: impute with column median (robust to outliers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Categorical columns: fill with 'Unknown' sentinel
        df['KC(SubSkills)'] = df['KC(SubSkills)'].fillna('Unknown')
        df['Problem Name'] = df['Problem Name'].fillna('Unknown')
        df['Problem Hierarchy'] = df['Problem Hierarchy'].fillna('Unknown')

        logger.info(f"Missing values handled — numeric: median imputation, categorical: 'Unknown' fill")

        # ── Step 4: Categorical Variable Encoding ─────────────────────────────
        # Problem Hierarchy: ordinal encoding via pandas category codes
        df['Problem Hierarchy'] = df['Problem Hierarchy'].astype('category').cat.codes

        logger.info("Categorical variables encoded to numerical form")

        # ── Step 5: Feature Engineering ───────────────────────────────────────
        # 5a. Engagement Ratio — total number of interactions per student
        #     Higher engagement typically correlates with better performance
        df['engagement_ratio'] = df.groupby('Anon Student Id')['Anon Student Id'].transform('count')

        # 5b. Consistency Index — inverse of response-time variability
        #     A consistent student has low variance in step durations
        df['consistency_index'] = 1 / (
            df.groupby('Anon Student Id')['Step Duration (sec)'].transform('std').fillna(0) + 1
        )

        # 5c. Mastery Trend — expanding cumulative success rate over time
        #     Captures learning velocity: is the student improving?
        df = df.sort_values(['Anon Student Id', 'Step Start Time'])
        df['mastery_trend'] = (
            df.groupby('Anon Student Id')[Config.TARGET_COLUMN].cumsum()
            / (df.groupby('Anon Student Id').cumcount() + 1)
        )

        # 5d. Timeliness — binary flag for response speed
        #     1 = faster than median response time, 0 = slower
        median_dur = df['Step Duration (sec)'].median()
        df['timeliness'] = (df['Step Duration (sec)'] < median_dur).astype(int)

        # 5e. Student Ability — historical mean success rate per student
        df['student_ability'] = df.groupby('Anon Student Id')[Config.TARGET_COLUMN].transform('mean')

        # 5f. Problem Difficulty — baseline success rate across all students per problem
        df['problem_difficulty'] = df.groupby('Problem Name')[Config.TARGET_COLUMN].transform('mean')

        logger.info("Feature engineering complete: engagement_ratio, consistency_index, "
                     "mastery_trend, timeliness, student_ability, problem_difficulty")

        # ── Step 6: Feature Selection ─────────────────────────────────────────
        feature_cols = [
            'student_ability', 'problem_difficulty', 'Problem Hierarchy',
            'Step Duration (sec)', 'engagement_ratio', 'consistency_index',
            'mastery_trend', 'timeliness', 'Anon Student Id'
        ]
        X = df[feature_cols].copy()

        # ── Step 7: Final Imputation ──────────────────────────────────────────
        X = X.fillna(X.median(numeric_only=True))

        y = df[Config.TARGET_COLUMN].reset_index(drop=True)

        logger.info(f"Preprocessing complete — X shape: {X.shape}, y shape: {y.shape}")
        return X, y
