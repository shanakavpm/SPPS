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
        Bridge to Algebra schema using vectorized NumPy operations.
        """
        np.random.seed(Config.RANDOM_STATE)

        n_students = 500
        n_problems = 200
        n_kcs = 15

        student_ids = np.array([f"Student_{i:04d}" for i in range(n_students)])
        problem_names = np.array([f"Problem_{i:03d}" for i in range(n_problems)])
        hierarchies = np.array([f"Unit_{u}, Section_{s}" for u in range(1, 6) for s in range(1, 4)])
        kc_skills = np.array([f"KC_Skill_{k}" for k in range(1, n_kcs + 1)])

        # Assign innate ability per student and difficulty per problem
        student_ability_vals = np.clip(np.random.normal(0.6, 0.2, n_students), 0.1, 0.95)
        problem_difficulty_vals = np.clip(np.random.normal(0.5, 0.25, n_problems), 0.05, 0.95)

        # Generate indices for random choices
        student_idx = np.random.randint(0, n_students, size=n_rows)
        problem_idx = np.random.randint(0, n_problems, size=n_rows)
        hierarchy_idx = np.random.randint(0, len(hierarchies), size=n_rows)
        kc_idx = np.random.randint(0, n_kcs, size=n_rows)

        # Map to actual values
        sids = student_ids[student_idx]
        pids = problem_names[problem_idx]
        abils = student_ability_vals[student_idx]
        diffs = problem_difficulty_vals[problem_idx]

        # Probability of correct = f(ability, difficulty)
        p_correct = np.clip(abils * (1 - diffs * 0.5) + np.random.normal(0, 0.05, size=n_rows), 0, 1)
        correct = (np.random.random(size=n_rows) < p_correct).astype(int)

        # Step duration calculation
        base_durations = 20 + (1 - abils) * 30 + diffs * 40
        durations = np.maximum(1, base_durations + np.random.normal(0, 15, size=n_rows))

        # Time generation
        base_time = np.datetime64("2008-09-01 08:00:00")
        offsets = (np.arange(n_rows) * 5 + np.random.randint(0, 100, size=n_rows)).astype('timedelta64[s]')
        start_times = base_time + offsets
        end_times = start_times + durations.astype('timedelta64[s]')

        # Knowledge Categories (with 15% NaN)
        kcs = kc_skills[kc_idx]
        kc_mask = np.random.random(size=n_rows) <= 0.15
        kcs_list = kcs.astype(object)
        kcs_list[kc_mask] = np.nan

        df = pd.DataFrame({
            'Anon Student Id': sids,
            'Problem Name': pids,
            'Problem Hierarchy': hierarchies[hierarchy_idx],
            'Correct First Attempt': correct,
            'Step Duration (sec)': np.round(durations, 2),
            'Step Start Time': start_times.astype(str),
            'Step End Time': end_times.astype(str),
            'KC(SubSkills)': kcs_list
        })

        logger.info(f"Synthetic dataset generated (vectorized): {len(df)} records")
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
