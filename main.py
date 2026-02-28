import logging
import os
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

def run_ml_framework():
    """Main program to run the Student Performance analysis & visual reports."""
    logger.info("--- Starting Student Analytics Framework ---")
    XAIEngine.setup_style()
    
    try:
        # 1. Load Data (KDD Cup - Bridge to Algebra)
        logger.info("Loading student data...")
        raw_df = StudentDataLoader.load_bridge_sample(n_rows=50000)
        X_all, y = StudentPreprocessor.process_bridge(raw_df)

        # 2. Extract Features
        model_features = [
            'student_ability', 'problem_difficulty', 'Problem Hierarchy', 
            'Step Duration (sec)', 'engagement_ratio', 'consistency_index', 'mastery_trend', 'timeliness'
        ]
        X = X_all[model_features]
        
        # 3. Train/Test Split & Scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_res), columns=model_features)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=model_features)

        # 4. Multi-Model Evaluation
        models_suite = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=Config.RANDOM_STATE),
            'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=Config.RANDOM_STATE),
            'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1, random_state=Config.RANDOM_STATE)
        }
        
        evaluator = ModelEvaluator(models_suite)
        results = evaluator.run_eval(X_train_sc, X_test_sc, y_res, y_test)
        
        # Simple Performance Table
        print("\n" + "="*85)
        print(f"{'Model Name':<20} | {'Acc':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AUC':<8}")
        print("-" * 85)
        for name, m in results.items():
            print(f"{name:<20} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['roc_auc']:.4f}")
        print("="*85 + "\n")

        # 5. Visualizations
        df_viz = X_all.copy()
        df_viz[Config.TARGET_COLUMN] = y.values
        
        logger.info("Generating Analytics Reports (Normal, Geo, Network)...")
        XAIEngine.generate_normal_visuals(df_viz, Config.TARGET_COLUMN)
        XAIEngine.generate_designed_visuals(df_viz)
        
        best_model = results['Random Forest']['model']
        XAIEngine.generate_diagnostic_plots(best_model, X_test_sc, y_test, 'Random_Forest')
        XAIEngine.generate_learning_curve(best_model, X_train_sc, y_res, 'Random_Forest')

        XAIEngine.generate_geo_visuals(df_viz)
        XAIEngine.generate_network_visuals(X_all)

        # 6. Explainability
        logger.info("Running XAI explanations (SHAP/LIME)...")
        XAIEngine.run_explainability_suite(best_model, X_train_sc, X_test_sc, model_features)
        XAIEngine.generate_pdp(best_model, X_train_sc, model_features[:3])

        # 7. Deliverables & Audit Logs
        logger.info("Generating final project deliverables...")
        df_viz.to_csv(os.path.join(Config.OUTPUTS_DIR, 'cleaned_data.csv'), index=False)
        
        # Simple Figure Catalog for the paper
        catalog = pd.DataFrame([
            {'Visual': 'Statistical Suite', 'Filename': 'correlation_heatmap.png', 'Insight': 'Feature Associations'},
            {'Visual': 'Mastery Trend', 'Filename': 'slope_chart_mastery.png', 'Insight': 'Learning Velocity'},
            {'Visual': 'Spatial Risk', 'Filename': 'geo_static_points.png', 'Insight': 'District Disparities'},
            {'Visual': 'Curriculum Network', 'Filename': 'graph_overall.png', 'Insight': 'Skill Hubs'},
            {'Visual': 'XAI Global', 'Filename': 'shap_summary.png', 'Insight': 'Factor Attribution'}
        ])
        catalog.to_csv(os.path.join(Config.OUTPUTS_DIR, 'figure_index.csv'), index=False)

        # MANDATORY: Data Dictionary
        dict_text = """# Data Dictionary - Student Performance Prediction
| Column | Type | Description |
| :--- | :--- | :--- |
| Anon Student Id | String | Unique student identifier. |
| student_ability | Float | Historical success rate of the student (mean). |
| problem_difficulty | Float | Baseline success rate across all students for this step. |
| engagement_ratio | Float | Count of interactions as a proxy for engagement. |
| consistency_index | Float | Inverse of standard deviation of response times. |
| mastery_trend | Float | Expanding mean of Correct First Attempt (learning velocity). |
| timeliness | Binary | 1 if response is faster than median; otherwise 0. |
| Correct First Attempt | Integer | Target: 1 for correct, 0 for incorrect. |
| latitude/longitude | Float | Synthetic coordinates for spatial risk analysis. |
| geo_region | String | School district assignment for the student. |
"""
        with open(os.path.join(Config.OUTPUTS_DIR, 'data_dictionary.md'), 'w') as f:
            f.write(dict_text)

        # MANDATORY: Cleaning Log
        cleaning_log = pd.DataFrame([
            {'Step': 'Loading', 'Action': 'Chunked loading of 50,000 Bridge to Algebra rows'},
            {'Step': 'Imputation', 'Action': 'Median fill for numeric, "Unknown" for categories'},
            {'Step': 'Balancing', 'Action': 'Applied SMOTE to training set for class parity'},
            {'Step': 'Scaling', 'Action': 'StandardScaler applied to all model features'},
            {'Step': 'Engineering', 'Action': 'Calculated Mastery, Consistency, and Timeliness metrics'}
        ])
        cleaning_log.to_csv(os.path.join(Config.OUTPUTS_DIR, 'cleaning_log.csv'), index=False)
        
        logger.info("Project ready. All deliverables in figures/ and outputs/.")

    except Exception as e:
        logger.error(f"Execution Error: {e}")
        raise

if __name__ == "__main__":
    run_ml_framework()
