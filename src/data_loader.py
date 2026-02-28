import pandas as pd
import logging
from .config import Config

logger = logging.getLogger(__name__)

class StudentDataLoader:
    """Handles loading and selection of the Bridge to Algebra educational dataset."""
    
    @staticmethod
    def load_bridge_sample(n_rows: int = 500000) -> pd.DataFrame:
        """Loads a subset of the massive Bridge to Algebra dataset."""
        try:
            # Loading only essential columns to save memory
            cols = [
                'Anon Student Id', 'Problem Name', 'Problem Hierarchy', 
                'Correct First Attempt', 'Step Duration (sec)', 'Step Start Time', 
                'Step End Time', 'KC(SubSkills)'
            ]
            df = pd.read_csv(Config.BRIDGE_TRAIN, sep='\t', usecols=cols, nrows=n_rows)
            logger.info(f"Bridge Sample Loaded: {len(df)} records.")
            return df
        except Exception as e:
            logger.error(f"Failed to load Bridge data: {e}")
            raise

    @staticmethod
    def load_bridge_generator(chunk_size: int = Config.CHUNK_SIZE):
        """Generator for processing the Bridge dataset in chunks (OOM prevention)."""
        cols = [
            'Anon Student Id', 'Problem Name', 'Problem Hierarchy', 
            'Correct First Attempt', 'Step Duration (sec)', 'Step Start Time', 
            'Step End Time', 'KC(SubSkills)'
        ]
        try:
            for chunk in pd.read_csv(Config.BRIDGE_TRAIN, sep='\t', usecols=cols, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error in data generator: {e}")
            raise
