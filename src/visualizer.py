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
import folium
from folium.plugins import HeatMap, MarkerCluster
import networkx as nx
from .config import Config

logger = logging.getLogger(__name__)

class XAIEngine:
    """Enhanced engine for Statistical, Geographic, and Network Visualizations."""
    
    @staticmethod
    def setup_style():
        plt.switch_backend('Agg')
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})

    @staticmethod
    def generate_diagnostic_plots(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'cm_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend()
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'roc_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()

    @staticmethod
    def generate_learning_curve(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str):
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
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'lc_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()

    @staticmethod
    def generate_normal_visuals(df: pd.DataFrame, target_col: str):
        """Generates statistical distribution plots (Meets template requirements)."""
        # 3. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'correlation_heatmap.png'))
        plt.close()

        # 4. Histogram & KDE
        plt.figure()
        sns.histplot(df[target_col], kde=True, bins=20)
        plt.title(f"Distribution of {target_col}")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'target_distribution.png'))
        plt.close()

        # 5. ECDF
        plt.figure()
        sns.ecdfplot(data=df, x=target_col)
        plt.title(f"ECDF of {target_col}")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'target_ecdf.png'))
        plt.close()

        # 6. Small Multiples: Score by Region
        if 'geo_region' in df.columns:
            g = sns.FacetGrid(df, col="geo_region", col_wrap=5)
            g.map(sns.histplot, target_col)
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'small_multiples_regions.png'))
            plt.close()

    @staticmethod
    def generate_designed_visuals(df: pd.DataFrame):
        """Designed visuals: Slope Chart for mastery evolution."""
        # Compare early student ability vs current mastery trend
        top_students = df.nlargest(10, 'mastery_trend')['Anon Student Id'].tolist()
        subset = df[df['Anon Student Id'].isin(top_students)].groupby('Anon Student Id').agg({
            'student_ability': 'first',
            'mastery_trend': 'last'
        })
        
        plt.figure(figsize=(8, 10))
        for i, row in subset.iterrows():
            plt.plot([0, 1], [row['student_ability'], row['mastery_trend']], marker='o', label=str(i)[:8])
            plt.text(-0.1, row['student_ability'], f"{row['student_ability']:.2f}")
            plt.text(1.1, row['mastery_trend'], f"{row['mastery_trend']:.2f}")
        
        plt.xticks([0, 1], ['Prior Ability', 'Current Mastery'])
        plt.title("Mastery Evolution Slope Chart (Top 10 Students)")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'slope_chart_mastery.png'))
        plt.close()

    @staticmethod
    def generate_geo_visuals(df: pd.DataFrame):
        """Requirement B: 5 Maps (Static and Interactive)."""
        logger.info("Generating 5-Map Geographic Suite...")
        
        # 1. Static Point Map
        plt.figure(figsize=(10, 10))
        plt.scatter(df['longitude'], df['latitude'], s=5, alpha=0.5, c=df[Config.TARGET_COLUMN], cmap='RdYlGn')
        plt.title("Map 1: Student Success Hotspots (Static)")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'geo_static_points.png'))
        plt.close()

        # 2. Interactive Cluster Map (Folium)
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in df.sample(min(1000, len(df))).iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Mastery: {row['mastery_trend']:.2f}",
                icon=folium.Icon(color='green' if row[Config.TARGET_COLUMN] == 1 else 'red')
            ).add_to(marker_cluster)
        m.save(os.path.join(Config.OUTPUTS_DIR, 'geo_map_interactive.html'))

        # 3. Interactive Heatmap (Density)
        m2 = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)
        HeatMap(data=df[['latitude', 'longitude']].values, radius=15).add_to(m2)
        m2.save(os.path.join(Config.OUTPUTS_DIR, 'geo_heatmap_density.html'))

        # 4. Hexbin Hotspot Map
        plt.figure(figsize=(10, 8))
        plt.hexbin(df['longitude'], df['latitude'], gridsize=30, cmap='Purples')
        plt.colorbar(label='Student Density')
        plt.title("Map 4: Student Concentration (Hexbin)")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'geo_hexbin_density.png'))
        plt.close()

        # 5. Choropleth (Regional Averages)
        # Note: In a real project, we would use a GeoJSON. Here we simulate regional heat.
        region_vals = df.groupby('geo_region')['mastery_trend'].mean()
        plt.figure(figsize=(12, 6))
        region_vals.plot(kind='bar', color='skyblue')
        plt.title("Map 5: Regional Mastery Index (Aggregate Choropleth Surrogate)")
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'geo_choropleth_surrogate.png'))
        plt.close()



    @staticmethod
    def run_explainability_suite(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: List[str], model_name: str = "model", plot_global: bool = True):
        logger.info(f"Starting XAI explanation suite for {model_name}...")
        
        # Global Feature Importance
        safe_model_name = model_name.lower().replace(" ", "_")
        
        if plot_global:
            if hasattr(model, 'coef_'):
                # Logistic Regression absolute coefficients
                coefs = np.abs(model.coef_[0])
                imps = pd.Series(coefs, index=feature_names).sort_values(ascending=False).head(15)
                plt.figure()
                imps.plot(kind='barh').invert_yaxis()
                plt.title(f"{model_name} Feature Importance (Abs Coefficients)")
                plt.savefig(os.path.join(Config.REPORTS_DIR, f'global_importance_{safe_model_name}.png'))
                plt.close()
            elif hasattr(model, 'feature_importances_'):
                imps = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
                plt.figure()
                imps.plot(kind='barh').invert_yaxis()
                plt.title(f"Tree-based Global Feature Importances ({model_name})")
                plt.savefig(os.path.join(Config.REPORTS_DIR, f'global_importance_{safe_model_name}.png'))
                plt.close()

        # SHAP Summary
        X_sample = X_test.sample(min(len(X_test), Config.SHAP_SAMPLE_SIZE))
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_viz = shap_values[1] if isinstance(shap_values, list) else shap_values
            plt.figure()
            shap.summary_plot(shap_viz, X_sample, show=False)
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'shap_summary.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

        # LIME Case Study
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=feature_names,
                class_names=['At Risk', 'Success'],
                mode='classification'
            )
            exp = lime_explainer.explain_instance(X_test.iloc[0], model.predict_proba, num_features=Config.LIME_TOP_FEATURES)
            exp.as_pyplot_figure()
            plt.title("LIME Case Study: Student Success Prediction")
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'lime_local_case.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"LIME failed: {e}")

    @staticmethod
    def generate_pdp(model: Any, X: pd.DataFrame, features: List[str]):
        try:
            plt.figure(figsize=(15, 12))
            PartialDependenceDisplay.from_estimator(model, X, features)
            plt.suptitle("Partial Dependence Plots (PDP)")
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'pdp_analysis.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"PDP failed: {e}")
