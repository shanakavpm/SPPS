import os

class Config:
    """Centralized configuration for the analysis."""
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    REPORTS_DIR = os.path.join(os.getcwd(), 'reports')
    
    DATA_MAT = os.path.join(DATA_DIR, 'student-mat.csv')
    DATA_POR = os.path.join(DATA_DIR, 'student-por.csv')
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TARGET_COLUMN = 'target'
    
    MERGE_COLUMNS = [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", 
        "Mjob", "Fjob", "reason", "nursery", "internet"
    ]
