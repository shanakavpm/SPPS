import os

class Config:
    """Centralized configuration for the 42-Hour Visualization Research Framework."""
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    REPORTS_DIR = FIGURES_DIR # Map to figures for backward compatibility

    # Ensure directories exist
    for d in [FIGURES_DIR, OUTPUTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Dataset 2: KDD Cup 2010 (Bridge to Algebra 2008-2009)
    BRIDGE_DIR = os.path.join(DATA_DIR, 'bridge_to_algebra_2008_2009')
    BRIDGE_TRAIN = os.path.join(BRIDGE_DIR, 'bridge_to_algebra_2008_2009_train.txt')
    
    # Global Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TARGET_COLUMN = 'target'
    CHUNK_SIZE = 100000
    
    # XAI Settings
    SHAP_SAMPLE_SIZE = 500
    LIME_TOP_FEATURES = 10
