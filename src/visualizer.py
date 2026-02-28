import os
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any, List
from .config import Config

logger = logging.getLogger(__name__)

class Visualizer:
    """Helper for generating and saving XAI plots."""
    @staticmethod
    def setup_style():
        """Configure aesthetics for scientific visualization."""
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    @staticmethod
    def save_shap_plots(model: Any, X_test: pd.DataFrame, feature_names: List[str]):
        """Generates global and local SHAP explanations and saves to Reports dir."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class binary classifier indexing for SHAP
        shap_viz_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        # 1. Global Summary Plot
        plt.figure()
        shap.summary_plot(shap_viz_vals, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        global_plot_path = os.path.join(Config.REPORTS_DIR, 'shap_summary_plot.png')
        plt.savefig(global_plot_path)
        plt.close()
        logger.info(f"Saved Global SHAP Plot to {global_plot_path}")

        # 2. Local Risk Plot (First student in test set)
        plt.figure()
        shap.summary_plot(shap_viz_vals[0:1, :], X_test.iloc[0:1, :], feature_names=feature_names, show=False)
        plt.title("Local Explanation for Selected Student")
        plt.tight_layout()
        local_plot_path = os.path.join(Config.REPORTS_DIR, 'shap_local_plot.png')
        plt.savefig(local_plot_path)
        plt.close()
        logger.info(f"Saved Local SHAP Plot to {local_plot_path}")
