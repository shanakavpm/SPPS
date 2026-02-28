import os

class Config:
    """Centralized configuration following the Registry pattern for multiple datasets."""
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

    # Ensure directories exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Dataset 1: UCI Student Performance (Cortez et al., 2008)
    UCI_MAT = os.path.join(DATA_DIR, 'student-mat.csv')
    UCI_POR = os.path.join(DATA_DIR, 'student-por.csv')
    UCI_MERGE_COLS = [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", 
        "Mjob", "Fjob", "reason", "nursery", "internet"
    ]

    # Dataset 2: KDD Cup 2010 (Bridge to Algebra 2008-2009) - Large Scale
    BRIDGE_DIR = os.path.join(DATA_DIR, 'bridge_to_algebra_2008_2009')
    BRIDGE_TRAIN = os.path.join(BRIDGE_DIR, 'bridge_to_algebra_2008_2009_train.txt')
    
    # Global Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TARGET_COLUMN = 'target'
    CHUNK_SIZE = 100000  # For large file processing
    
    # XAI Settings
    SHAP_SAMPLE_SIZE = 1000
    LIME_TOP_FEATURES = 10
