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
from .config import Config

logger = logging.getLogger(__name__)

class Visualizer:
    """Helper for generating and saving XAI and diagnostic plots."""
    @staticmethod
    def setup_style():
        """Configure aesthetics for scientific visualization."""
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    @staticmethod
    def save_confusion_matrix(y_true, y_pred, model_name: str):
        """Generates and saves a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        path = os.path.join(Config.REPORTS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved Confusion Matrix to {path}")

    @staticmethod
    def save_roc_curve(y_true, y_probs, model_name: str):
        """Generates and saves an ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(Config.REPORTS_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved ROC Curve to {path}")

    @staticmethod
    def save_feature_importance(model: Any, feature_names: List[str], model_name: str):
        """Generates and saves a feature importance bar chart for tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 15 Feature Importances: {model_name}')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        path = os.path.join(Config.REPORTS_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved Feature Importance Plot to {path}")

    @staticmethod
    def save_shap_plots(model: Any, X_test: pd.DataFrame, feature_names: List[str]):
        """Generates global and local SHAP explanations."""
        # TreeExplainer is efficient for RF/XGBoost
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except Exception:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
        
        # Handle different SHAP value formats
        shap_viz_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        # 1. Global Summary Plot
        plt.figure()
        shap.summary_plot(shap_viz_vals, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        global_plot_path = os.path.join(Config.REPORTS_DIR, 'shap_summary_plot.png')
        plt.savefig(global_plot_path)
        plt.close()

        # 2. Local Waterfall Plot (First student)
        # Recalculate for waterfall if needed
        plt.figure()
        if hasattr(explainer, 'expected_value'):
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            # For summary_plot logic we used before
            shap.summary_plot(shap_viz_vals[0:1, :], X_test.iloc[0:1, :], feature_names=feature_names, show=False)
        plt.title("Local Explanation for Selected Student")
        plt.tight_layout()
        local_plot_path = os.path.join(Config.REPORTS_DIR, 'shap_local_plot.png')
        plt.savefig(local_plot_path)
        plt.close()
        logger.info("Saved SHAP Plots to Reports.")

    @staticmethod
    def save_lime_explanation(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: List[str]):
        """Generates and saves a LIME explanation for a single prediction."""
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['Fail', 'Pass'],
            mode='classification'
        )
        
        # Explain the first instance in the test set
        exp = explainer.explain_instance(X_test.iloc[0], model.predict_proba, num_features=10)
        
        plt.figure()
        exp.as_pyplot_figure()
        plt.title("LIME Explanation for Selected Student")
        plt.tight_layout()
        path = os.path.join(Config.REPORTS_DIR, 'lime_explanation.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved LIME Explanation to {path}")

    @staticmethod
    def save_pdp_plot(model: Any, X: pd.DataFrame, features: List[str]):
        """Generates and saves Partial Dependence Plots."""
        plt.figure(figsize=(12, 10))
        PartialDependenceDisplay.from_estimator(model, X, features)
        plt.suptitle("Partial Dependence Plots")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(Config.REPORTS_DIR, 'pdp_plots.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved PDP Plots to {path}")
