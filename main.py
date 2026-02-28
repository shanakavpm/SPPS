import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

        # 3. Partition Data First (to prevent leakage in CV/Scaling)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y
        )

        # 4. Handle Imbalance: SMOTE only on Training Data
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 5. Standardize Features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        # 6. Comparative Model Training: LR, RF, XGBoost, SVM, k-NN, LightGBM
        # Using class_weight='balanced' where possible as an alternative/addition to SMOTE
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=Config.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=Config.RANDOM_STATE),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=Config.RANDOM_STATE),
            'SVM': SVC(probability=True, class_weight='balanced', random_state=Config.RANDOM_STATE),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'LightGBM': LGBMClassifier(random_state=Config.RANDOM_STATE, verbose=-1)
        }
        
        evaluator = ModelEvaluator(models)
        metrics = evaluator.run_eval(X_train_scaled, X_test_scaled, y_train_res, y_test)

        # 7. Print Performance Table
        _display_final_report(metrics)

        # 8. Advanced Visualizations for the Best Model (e.g., Random Forest or LightGBM)
        Visualizer.setup_style()
        best_model_name = max(metrics, key=lambda x: metrics[x]['f1'])
        logger.info(f"Targeting visualizations for the best model: {best_model_name}")
        
        target_model_data = metrics[best_model_name]
        model = target_model_data['model']
        
        # Diagnostics
        y_pred = model.predict(X_test_scaled)
        y_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        Visualizer.save_confusion_matrix(y_test, y_pred, best_model_name)
        Visualizer.save_roc_curve(y_test, y_probs, best_model_name)
        Visualizer.save_feature_importance(model, list(X.columns), best_model_name)
        
        # XAI (SHAP & LIME)
        Visualizer.save_shap_plots(model, X_test_scaled, list(X.columns))
        Visualizer.save_lime_explanation(model, X_train_scaled, X_test_scaled, list(X.columns))
        
        # PDP (Top 3 features from RF)
        if hasattr(model, 'feature_importances_'):
            top_features = [X.columns[i] for i in np.argsort(model.feature_importances_)[-3:]]
            Visualizer.save_pdp_plot(model, X_train_scaled, top_features)

        logger.info("Pipeline executed successfully. Enhanced outputs available in 'reports/'.")

    except Exception as e:
        logger.error(f"Critical error in pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def _display_final_report(metrics: dict):
    """Prints a professional markdown performance table to console."""
    print("\n### Performance Comparison Table (with 5-Fold Cross-Validation)")
    print("| Model | Test Acc | CV Mean Acc | CV Std | Precision | Recall | F1-Score | ROC-AUC |")
    print("|-------|----------|-------------|--------|-----------|--------|----------|---------|")
    for name, m in metrics.items():
        print(f"| {name} | {m['accuracy']:.4f} | {m['cv_accuracy_mean']:.4f} | {m['cv_accuracy_std']:.3f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |")

if __name__ == "__main__":
    run_pipeline()
