import pandas as pd
import numpy as np
import logging
import os
from .config import Config

logger = logging.getLogger(__name__)


class StudentDataLoader:
    """Handles loading the Bridge to Algebra educational dataset.
    
    If the actual KDD Cup dataset is not available, generates a realistic
    synthetic dataset with the same schema for demonstration.
    """

    REQUIRED_COLS = [
        'Anon Student Id', 'Problem Name', 'Problem Hierarchy',
        'Correct First Attempt', 'Step Duration (sec)', 'Step Start Time',
        'Step End Time', 'KC(SubSkills)'
    ]

    @staticmethod
    def load_bridge_sample(n_rows: int = 50000) -> pd.DataFrame:
        """Load a subset of the Bridge to Algebra dataset.
        
        Falls back to synthetic data if the real dataset is unavailable.
        """
        if os.path.exists(Config.BRIDGE_TRAIN):
            try:
                df = pd.read_csv(
                    Config.BRIDGE_TRAIN, sep='\t',
                    usecols=StudentDataLoader.REQUIRED_COLS, nrows=n_rows
                )
                logger.info(f"Bridge dataset loaded: {len(df)} records from file.")
                return df
            except Exception as e:
                logger.error(f"Failed to load Bridge data: {e}")
                raise
        else:
            logger.warning(
                f"Dataset file not found at {Config.BRIDGE_TRAIN}. "
                "Generating synthetic data with the same schema..."
            )
            return StudentDataLoader._generate_synthetic(n_rows)

    @staticmethod
    def _generate_synthetic(n_rows: int = 50000) -> pd.DataFrame:
        """
        Generate a realistic synthetic educational dataset matching the
        Bridge to Algebra schema.

        Simulates:
            - 500 unique students with varying ability levels
            - 200 unique problems with varying difficulty
            - 15 knowledge component (KC) categories
            - Realistic step durations and correctness patterns
        """
        np.random.seed(Config.RANDOM_STATE)

        n_students = 500
        n_problems = 200
        n_kcs = 15

        student_ids = [f"Student_{i:04d}" for i in range(n_students)]
        problem_names = [f"Problem_{i:03d}" for i in range(n_problems)]
        hierarchies = [f"Unit_{u}, Section_{s}" for u in range(1, 6) for s in range(1, 5)]
        kc_skills = [f"KC_Skill_{k}" for k in range(1, n_kcs + 1)]

        # Assign innate ability per student (some strong, some weak)
        student_ability = {s: np.clip(np.random.normal(0.6, 0.2), 0.1, 0.95) for s in student_ids}
        problem_difficulty = {p: np.clip(np.random.normal(0.5, 0.25), 0.05, 0.95) for p in problem_names}

        records = []
        base_time = pd.Timestamp("2008-09-01 08:00:00")

        for i in range(n_rows):
            sid = np.random.choice(student_ids)
            pid = np.random.choice(problem_names)
            ability = student_ability[sid]
            difficulty = problem_difficulty[pid]

            # Probability of correct = f(ability, difficulty)
            p_correct = np.clip(ability * (1 - difficulty * 0.5) + np.random.normal(0, 0.05), 0, 1)
            correct = int(np.random.random() < p_correct)

            # Step duration: faster students tend to be quicker; harder problems take longer
            base_duration = 20 + (1 - ability) * 30 + difficulty * 40
            duration = max(1, base_duration + np.random.normal(0, 15))

            start_time = base_time + pd.Timedelta(seconds=i * 5 + np.random.randint(0, 100))
            end_time = start_time + pd.Timedelta(seconds=duration)

            # Some KC entries are NaN (mimics real data)
            kc = np.random.choice(kc_skills) if np.random.random() > 0.15 else np.nan

            records.append({
                'Anon Student Id': sid,
                'Problem Name': pid,
                'Problem Hierarchy': np.random.choice(hierarchies),
                'Correct First Attempt': correct,
                'Step Duration (sec)': round(duration, 2),
                'Step Start Time': str(start_time),
                'Step End Time': str(end_time),
                'KC(SubSkills)': kc
            })

        df = pd.DataFrame(records)
        logger.info(f"Synthetic dataset generated: {len(df)} records, "
                     f"{n_students} students, {n_problems} problems")
        return df

    @staticmethod
    def load_bridge_generator(chunk_size: int = Config.CHUNK_SIZE):
        """Generator for processing the Bridge dataset in chunks (OOM prevention)."""
        try:
            for chunk in pd.read_csv(
                Config.BRIDGE_TRAIN, sep='\t',
                usecols=StudentDataLoader.REQUIRED_COLS,
                chunksize=chunk_size
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in data generator: {e}")
            raise
