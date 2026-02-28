import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import Config
from src.data_loader import StudentDataLoader
from src.preprocessor import StudentPreprocessor
from src.models import ModelEvaluator
from src.visualizer import Visualizer

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    """Main execution function for the Student Performance Prediction System."""
    try:
        # 1. Load Data
        loader = StudentDataLoader()
        raw_df = loader.load()

        # 2. Preprocess and Engineer Features
        preprocessor = StudentPreprocessor()
        X, y = preprocessor.transform(raw_df)

        # 3. Handle Imbalance: SMOTE for robust binary classification
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)

        # 4. Partition Data: Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE
        )

        # 5. Standardize Features for Baseline Consistency
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        # 6. Comparative Model Training: LR vs RF vs XGBoost
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=Config.RANDOM_STATE),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=Config.RANDOM_STATE)
        }
        evaluator = ModelEvaluator(models)
        metrics = evaluator.run_eval(X_train_scaled, X_test_scaled, y_train, y_test)

        # 7. Print Performance Table
        _display_final_report(metrics)

        # 8. Explainability Integration (XAI)
        # Choosing Random Forest for robust SHAP integration and transparency
        Visualizer.setup_style()
        best_model = metrics['Random Forest']['model']
        Visualizer.save_shap_plots(best_model, X_test_scaled, list(X.columns))
        
        logger.info("Pipeline executed successfully. Outputs available in 'reports/'.")

    except Exception as e:
        logger.error(f"Critical error in pipeline: {e}")
        raise

def _display_final_report(metrics: dict):
    """Prints a professional markdown performance table to console."""
    print("\n### Performance Comparison Table")
    print("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    print("|-------|----------|-----------|--------|----------|---------|")
    for name, m in metrics.items():
        print(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |")

if __name__ == "__main__":
    run_pipeline()
