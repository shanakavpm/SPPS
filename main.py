"""
Student Performance Prediction System (SPPS) — Main Pipeline

Covers:
    Q4: Data Preprocessing (loading, cleaning, encoding, feature engineering, scaling)
    Q5: Data Analysis (Logistic Regression, Random Forest, XGBoost, LightGBM, LSTM, GRU)
    Q6: Explainable AI (SHAP global/local, LIME, PDP, feature importance, comparative report)
"""

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
from src.lstm_model import SequenceModelEvaluator
from src.visualizer import XAIEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline():
    """Main pipeline: preprocessing → model training → evaluation → XAI."""
    logger.info("=" * 60)
    logger.info("  Student Performance Prediction System (SPPS)")
    logger.info("=" * 60)

    XAIEngine.setup_style()

    try:
        # ══════════════════════════════════════════════════════════════
        # Q4: DATA PREPROCESSING
        # ══════════════════════════════════════════════════════════════
        logger.info("[Q4] Loading and preprocessing student data...")
        
        # Performance Enhancement: Check for preprocessed cache
        if os.path.exists(Config.CACHED_DATA):
            logger.info("Found cached preprocessed data. Loading from Parquet...")
            X_all = pd.read_parquet(Config.CACHED_DATA)
            y = X_all[Config.TARGET_COLUMN]
            X_all = X_all.drop(columns=[Config.TARGET_COLUMN])
        else:
            logger.info("No cache found or cache disabled. Processing raw data...")
            # Default to 100k for performance, can be increased for final runs
            raw_df = StudentDataLoader.load_bridge_sample(n_rows=100000)
            X_all, y = StudentPreprocessor.process_bridge(raw_df)
            
            # Save to cache for future runs
            cache_df = X_all.copy()
            cache_df[Config.TARGET_COLUMN] = y.values
            cache_df.to_parquet(Config.CACHED_DATA)
            logger.info(f"Preprocessed data cached to {Config.CACHED_DATA}")

        # Model features (excluding student ID which is only for grouping)
        model_features = [
            'student_ability', 'problem_difficulty', 'Problem Hierarchy',
            'Step Duration (sec)', 'engagement_ratio', 'consistency_index',
            'mastery_trend', 'timeliness'
        ]
        X = X_all[model_features]

        # Train/Test Split (stratified to preserve class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE, stratify=y
        )

        # SMOTE — oversample minority class in training set only
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # StandardScaler — normalize features to zero mean, unit variance
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_res), columns=model_features)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=model_features)

        logger.info("[Q4] Preprocessing complete.")

        # ══════════════════════════════════════════════════════════════
        # Q5: MULTI-MODEL EVALUATION
        # ══════════════════════════════════════════════════════════════
        logger.info("[Q5] Training traditional ML models...")

        models_suite = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=Config.RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, eval_metric='logloss', random_state=Config.RANDOM_STATE
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100, verbose=-1, random_state=Config.RANDOM_STATE
            ),
        }

        evaluator = ModelEvaluator(models_suite)
        ml_results = evaluator.run_eval(X_train_sc, X_test_sc, y_res, y_test)

        # Deep Learning models (LSTM / GRU) — sequential patterns
        logger.info("[Q5] Training deep learning models (LSTM, GRU)...")
        dl_results = SequenceModelEvaluator.evaluate(
            X_all, y, model_features,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )

        # ══════════════════════════════════════════════════════════════
        # Q5: VISUALIZATIONS
        # ══════════════════════════════════════════════════════════════
        logger.info("[Q5] Generating statistical and diagnostic visuals...")

        # Prepare visualization dataframe
        df_viz = X_all[model_features].copy()
        df_viz['Anon Student Id'] = X_all['Anon Student Id'].values
        df_viz[Config.TARGET_COLUMN] = y.values

        XAIEngine.generate_statistical_visuals(df_viz, Config.TARGET_COLUMN)
        XAIEngine.generate_mastery_slope_chart(df_viz)

        # Diagnostics for each traditional ML model
        for name, result in ml_results.items():
            safe = name.lower().replace(" ", "_")
            XAIEngine.generate_diagnostic_plots(result['model'], X_test_sc, y_test, name)

        # Learning curve for best traditional model
        best_ml_name = max(ml_results, key=lambda k: ml_results[k]['f1'])
        best_ml_model = ml_results[best_ml_name]['model']
        XAIEngine.generate_learning_curve(best_ml_model, X_train_sc, y_res, best_ml_name)

        # ══════════════════════════════════════════════════════════════
        # Q6: EXPLAINABLE AI (XAI)
        # ══════════════════════════════════════════════════════════════
        logger.info("[Q6] Running Explainable AI suite...")

        # Global + Local XAI on Logistic Regression (transparent baseline)
        XAIEngine.run_explainability_suite(
            ml_results['Logistic Regression']['model'],
            X_train_sc, X_test_sc, model_features,
            model_name="Logistic Regression"
        )

        # Global + Local XAI on XGBoost (black-box — supports SHAP TreeExplainer)
        blackbox_name = 'XGBoost'
        blackbox_model = ml_results[blackbox_name]['model']
        XAIEngine.run_explainability_suite(
            blackbox_model, X_train_sc, X_test_sc, model_features,
            model_name=blackbox_name
        )

        # Partial Dependence Plots (top 3 features) — using XGBoost
        XAIEngine.generate_pdp(blackbox_model, X_train_sc, model_features[:3])

        # Comparative analysis report (Q5 + Q6)
        XAIEngine.generate_comparative_report(
            ml_results, dl_results,
            output_path=os.path.join(Config.OUTPUTS_DIR, 'comparative_analysis.txt')
        )

        # ══════════════════════════════════════════════════════════════
        # DELIVERABLES
        # ══════════════════════════════════════════════════════════════
        logger.info("Generating project deliverables...")

        # Cleaned dataset
        df_viz.to_csv(os.path.join(Config.OUTPUTS_DIR, 'cleaned_data.csv'), index=False)

        # Data Dictionary
        dict_text = """# Data Dictionary — Student Performance Prediction System (SPPS)

| Column | Type | Description |
|:---|:---|:---|
| student_ability | Float | Historical mean success rate of the student. |
| problem_difficulty | Float | Baseline success rate across all students for this step/problem. |
| Problem Hierarchy | Integer | Ordinal-encoded problem category hierarchy. |
| Step Duration (sec) | Float | Time taken by the student to complete the step. |
| engagement_ratio | Float | Total count of interactions per student (engagement proxy). |
| consistency_index | Float | Inverse of standard deviation of response times (higher = more consistent). |
| mastery_trend | Float | Expanding cumulative mean of Correct First Attempt (learning velocity). |
| timeliness | Binary (0/1) | 1 if response time is faster than median; 0 otherwise. |
| target | Binary (0/1) | Target variable: 1 = Correct First Attempt, 0 = Incorrect. |
"""
        with open(os.path.join(Config.OUTPUTS_DIR, 'data_dictionary.md'), 'w', encoding='utf-8') as f:
            f.write(dict_text)

        # Cleaning Log
        cleaning_log = pd.DataFrame([
            {'Step': '1. Loading', 'Action': 'Loaded 500,000 rows from Bridge to Algebra (KDD Cup 2010)'},
            {'Step': '2. Duplicates', 'Action': 'Removed exact duplicate rows using drop_duplicates()'},
            {'Step': '3. Missing Values', 'Action': 'Median imputation for numeric columns; "Unknown" fill for categorical'},
            {'Step': '4. Encoding', 'Action': 'Problem Hierarchy converted to ordinal codes via .cat.codes'},
            {'Step': '5. Feature Engineering', 'Action': 'Derived: engagement_ratio, consistency_index, mastery_trend, timeliness, student_ability, problem_difficulty'},
            {'Step': '6. Balancing', 'Action': 'Applied SMOTE to training set for class balance'},
            {'Step': '7. Scaling', 'Action': 'StandardScaler applied to all model features (zero mean, unit variance)'},
        ])
        cleaning_log.to_csv(os.path.join(Config.OUTPUTS_DIR, 'cleaning_log.csv'), index=False)

        # Figure Index
        catalog = pd.DataFrame([
            {'Visual': 'Correlation Heatmap', 'Filename': 'correlation_heatmap.png', 'Purpose': 'Feature association analysis'},
            {'Visual': 'Target Distribution', 'Filename': 'target_distribution.png', 'Purpose': 'Class balance inspection'},
            {'Visual': 'Feature Box Plots', 'Filename': 'feature_boxplots.png', 'Purpose': 'Feature distribution & outliers'},
            {'Visual': 'Mastery Slope Chart', 'Filename': 'slope_chart_mastery.png', 'Purpose': 'Learning velocity visualization'},
            {'Visual': 'Confusion Matrices', 'Filename': 'cm_*.png', 'Purpose': 'Classification error analysis'},
            {'Visual': 'ROC Curves', 'Filename': 'roc_*.png', 'Purpose': 'Discrimination ability'},
            {'Visual': 'Learning Curve', 'Filename': 'lc_*.png', 'Purpose': 'Overfitting / underfitting detection'},
            {'Visual': 'SHAP Summary', 'Filename': 'shap_summary_*.png', 'Purpose': 'Global feature attribution (XAI)'},
            {'Visual': 'SHAP Force Plot', 'Filename': 'shap_force_*.png', 'Purpose': 'Individual student explanation (XAI)'},
            {'Visual': 'LIME Explanation', 'Filename': 'lime_local_*.png', 'Purpose': 'Local model approximation (XAI)'},
            {'Visual': 'PDP Analysis', 'Filename': 'pdp_analysis.png', 'Purpose': 'Marginal feature effects (XAI)'},
            {'Visual': 'Global Importance', 'Filename': 'global_importance_*.png', 'Purpose': 'Feature ranking per model (XAI)'},
        ])
        catalog.to_csv(os.path.join(Config.OUTPUTS_DIR, 'figure_index.csv'), index=False)

        logger.info("=" * 60)
        logger.info("  Pipeline complete. All outputs saved to figures/ and outputs/")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
