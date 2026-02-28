import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import Config
from src.data_loader import StudentDataLoader
from src.preprocessor import StudentPreprocessor
from src.models import ModelEvaluator
from src.visualizer import XAIEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ml_framework(dataset_type: str = 'bridge'):
    """
    Unified Orchestrator for the Student Performance Prediction Framework (Part B).
    """
    logger.info(f"--- Starting Framework Execution: Dataset [{dataset_type.upper()}] ---")
    XAIEngine.setup_style()
    
    try:
        # 1. Data Collection Phase
        if dataset_type == 'uci':
            raw_df = StudentDataLoader.load_uci()
            X, y = StudentPreprocessor.process_uci(raw_df)
        else:
            # Load a manageable sample for the Bridge dataset
            raw_df = StudentDataLoader.load_bridge_sample(n_rows=50000)
            X, y = StudentPreprocessor.process_bridge(raw_df)

        # 2. Exploratory Visualization
        XAIEngine.generate_correlation_heatmap(X)

        # 3. Partitioning & Preprocessing Phase
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )

        # Handle class imbalance (Part B Requirement)
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Scaling continuous features
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        # 4. Model Analysis Phase (Comparative Study)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=Config.RANDOM_STATE),
            'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=Config.RANDOM_STATE),
            'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1, random_state=Config.RANDOM_STATE)
        }
        
        evaluator = ModelEvaluator(models)
        results = evaluator.run_eval(X_train_sc, X_test_sc, y_res, y_test)
        
        _print_performance_summary(results)

        # 5. Multimodal Diagnostics
        for name, m in results.items():
            safe_name = name.replace(" ", "_").lower()
            XAIEngine.generate_diagnostic_plots(m['model'], X_test_sc, y_test, safe_name)
            XAIEngine.generate_learning_curve(m['model'], X_train_sc, y_res, safe_name)

        # 6. XAI Integration Phase
        # We use Random Forest as the primary explainer for stability in SHAP/LIME
        best_model = results['Random Forest']['model']
        XAIEngine.run_explainability_suite(best_model, X_train_sc, X_test_sc, list(X.columns))
        
        # Global Analysis: Partial Dependence for key engineered features
        XAIEngine.generate_pdp(best_model, X_train_sc, ['student_ability', 'consistency_index'])

        logger.info("Pipeline completed. Check 'reports/' for all visual evidence.")

    except Exception as e:
        logger.error(f"Pipeline failure: {e}")
        raise

def _print_performance_summary(results):
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Acc':<8} | {'CV Mean':<8} | {'F1':<8} | {'AUC':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<25} | {m['accuracy']:.4f} | {m['cv_accuracy_mean']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_ml_framework(dataset_type='bridge')
