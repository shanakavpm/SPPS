import os
import logging
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, List
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
from .config import Config

logger = logging.getLogger(__name__)

class XAIEngine:
    """Core engine for Explainable AI (XAI) and Diagnostic Visualizations."""
    
    @staticmethod
    def setup_style():
        sns.set(style="whitegrid", palette="muted")
        plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})

    @staticmethod
    def generate_diagnostic_plots(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        """Generates Confusion Matrix and ROC Curve."""
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'cm_{model_name.lower()}.png'))
        plt.close()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend()
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'roc_{model_name.lower()}.png'))
        plt.close()

    @staticmethod
    def generate_learning_curve(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str):
        """Generates Learning Curve to diagnose over/underfitting."""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=Config.CV_FOLDS, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        plt.figure()
        plt.plot(train_sizes, train_mean, 'o-', label="Training score")
        plt.plot(train_sizes, test_mean, 's-', label="Cross-validation score")
        plt.title(f"Learning Curve: {model_name}")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'lc_{model_name.lower()}.png'))
        plt.close()

    @staticmethod
    def generate_correlation_heatmap(df: pd.DataFrame):
        """Generates a correlation heatmap for feature analysis."""
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'correlation_heatmap.png'))
        plt.close()

    @staticmethod
    def run_explainability_suite(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: List[str]):
        """Executes a full suite of XAI methods: SHAP, LIME, and Feature Importance."""
        logger.info("Starting XAI explanation suite...")
        
        # 1. Global: Feature Importance (Tree-based)
        if hasattr(model, 'feature_importances_'):
            imps = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
            plt.figure()
            imps.plot(kind='barh').invert_yaxis()
            plt.title("Top 15 Global Feature Importances")
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'global_importance.png'))
            plt.close()

        # 2. Global: SHAP Summary
        # Sample to prevent long computation on large datasets
        X_sample = X_test.sample(min(len(X_test), Config.SHAP_SAMPLE_SIZE))
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # Handle multi-class binary classifier indexing
            shap_viz = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            plt.figure()
            shap.summary_plot(shap_viz, X_sample, show=False)
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'shap_summary.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

        # 3. Local: LIME (First student in test set)
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=feature_names,
                class_names=['At Risk', 'Success'],
                mode='classification'
            )
            exp = lime_explainer.explain_instance(X_test.iloc[0], model.predict_proba, num_features=Config.LIME_TOP_FEATURES)
            fig = exp.as_pyplot_figure()
            plt.title("LIME Local Explanation: Student Case Study")
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'lime_local_case.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"LIME failed: {e}")

    @staticmethod
    def generate_pdp(model: Any, X: pd.DataFrame, features: List[str]):
        """Partial Dependence Plots for analyzing marginal effects of key features."""
        try:
            plt.figure()
            PartialDependenceDisplay.from_estimator(model, X, features)
            plt.suptitle("Partial Dependence Plots (PDP)")
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'pdp_analysis.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"PDP failed: {e}")
