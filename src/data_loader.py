import pandas as pd
import logging
from .config import Config

logger = logging.getLogger(__name__)

class StudentDataLoader:
    """Handles loading and merging of student datasets."""
    @staticmethod
    def load() -> pd.DataFrame:
        try:
            mat_df = pd.read_csv(Config.DATA_MAT, sep=';')
            por_df = pd.read_csv(Config.DATA_POR, sep=';')
            
            merged_df = pd.merge(
                mat_df, por_df, 
                on=Config.MERGE_COLUMNS, 
                suffixes=('_mat', '_por')
            )
            
            logger.info(f"Loaded {len(mat_df)} Math and {len(por_df)} Por records.")
            logger.info(f"Successfully merged {len(merged_df)} common records.")
            return merged_df
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
