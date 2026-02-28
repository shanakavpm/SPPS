import pandas as pd
import logging
from .config import Config

logger = logging.getLogger(__name__)

class StudentDataLoader:
    """Handles loading and selection of various educational datasets."""
    
    @staticmethod
    def load_uci() -> pd.DataFrame:
        """Loads and merges the UCI Student Performance dataset (Math & Portuguese)."""
        try:
            mat_df = pd.read_csv(Config.UCI_MAT, sep=';')
            por_df = pd.read_csv(Config.UCI_POR, sep=';')
            
            merged_df = pd.merge(
                mat_df, por_df, 
                on=Config.UCI_MERGE_COLS, 
                suffixes=('_mat', '_por')
            )
            logger.info(f"UCI Dataset Loaded: {len(merged_df)} merged records.")
            return merged_df
        except Exception as e:
            logger.error(f"Failed to load UCI data: {e}")
            raise

    @staticmethod
    def load_bridge_sample(n_rows: int = 500000) -> pd.DataFrame:
        """Loads a subset of the massive Bridge to Algebra dataset."""
        try:
            # Loading only essential columns to save memory
            cols = ['Anon Student Id', 'Problem Name', 'Problem Hierarchy', 'Correct First Attempt', 'Step Duration (sec)', 'Step Start Time', 'Step End Time']
            df = pd.read_csv(Config.BRIDGE_TRAIN, sep='\t', usecols=cols, nrows=n_rows)
            logger.info(f"Bridge Sample Loaded: {len(df)} records.")
            return df
        except Exception as e:
            logger.error(f"Failed to load Bridge data: {e}")
            raise

    @staticmethod
    def load_bridge_generator(chunk_size: int = Config.CHUNK_SIZE):
        """Generator for processing the Bridge dataset in chunks (OOM prevention)."""
        cols = ['Anon Student Id', 'Problem Name', 'Problem Hierarchy', 'Correct First Attempt', 'Step Duration (sec)', 'Step Start Time', 'Step End Time']
        try:
            for chunk in pd.read_csv(Config.BRIDGE_TRAIN, sep='\t', usecols=cols, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error in data generator: {e}")
            raise
