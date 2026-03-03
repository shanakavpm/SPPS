"""
Visualization & Explainability Engine for Student Performance Prediction.

Covers:
    - Statistical distribution plots (Q5)
    - Model diagnostic plots: Confusion Matrix, ROC, Learning Curve (Q5)
    - Global XAI: Feature Importance, SHAP Summary, PDP (Q6)
    - Local XAI: SHAP Force Plot, LIME Case Study (Q6)
    - Comparative interpretability report (Q6)
"""

import os
import logging

import shap
import lime
import lime.lime_tabular
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, List, Dict

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve

from .config import Config

logger = logging.getLogger(__name__)


class XAIEngine:
    """Engine for statistical visualizations and explainable AI analysis."""

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def setup_style():
        plt.switch_backend('Agg')
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})

    # ──────────────────────────────────────────────────────────────────────
    # Q5 — Statistical Distribution Visuals
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def generate_statistical_visuals(df: pd.DataFrame, target_col: str):
        """Generate statistical distribution plots for data exploration."""

        # 1. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'correlation_heatmap.png'), dpi=150)
        plt.close()

        # 2. Target Distribution (Histogram + KDE)
        plt.figure()
        sns.histplot(df[target_col], kde=True, bins=20)
        plt.title(f"Distribution of {target_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'target_distribution.png'), dpi=150)
        plt.close()

        # 3. ECDF (Empirical Cumulative Distribution)
        plt.figure()
        sns.ecdfplot(data=df, x=target_col)
        plt.title(f"Empirical CDF of {target_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'target_ecdf.png'), dpi=150)
        plt.close()

        # 4. Feature Distributions — Box plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        plot_cols = [c for c in numeric_cols if c != target_col][:8]
        if plot_cols:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, col in zip(axes.flatten(), plot_cols):
                sns.boxplot(data=df, y=col, ax=ax)
                ax.set_title(col, fontsize=10)
            plt.suptitle("Feature Distributions (Box Plots)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'feature_boxplots.png'), dpi=150)
            plt.close()

        logger.info("Statistical visuals generated: heatmap, distribution, ECDF, boxplots")

    # ──────────────────────────────────────────────────────────────────────
    # Q5 — Mastery Trend Visualization
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def generate_mastery_slope_chart(df: pd.DataFrame):
        """Slope chart comparing early student ability vs. current mastery trend."""
        top_students = df.nlargest(10, 'mastery_trend')['Anon Student Id'].tolist()
        subset = df[df['Anon Student Id'].isin(top_students)].groupby('Anon Student Id').agg({
            'student_ability': 'first',
            'mastery_trend': 'last'
        })

        plt.figure(figsize=(8, 10))
        for i, row in subset.iterrows():
            plt.plot([0, 1], [row['student_ability'], row['mastery_trend']],
                     marker='o', linewidth=2, label=str(i)[:8])
            plt.text(-0.12, row['student_ability'], f"{row['student_ability']:.2f}", fontsize=9)
            plt.text(1.05, row['mastery_trend'], f"{row['mastery_trend']:.2f}", fontsize=9)

        plt.xticks([0, 1], ['Prior Ability', 'Current Mastery'])
        plt.title("Mastery Evolution Slope Chart (Top 10 Students)")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, 'slope_chart_mastery.png'), dpi=150)
        plt.close()
        logger.info("Mastery slope chart generated")

    # ──────────────────────────────────────────────────────────────────────
    # Q5 — Model Diagnostic Plots
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def generate_diagnostic_plots(model: Any, X_test: pd.DataFrame,
                                   y_test: pd.Series, model_name: str):
        """Generate confusion matrix and ROC curve for a trained model."""
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        safe_name = model_name.lower().replace(" ", "_")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Incorrect', 'Correct'],
                    yticklabels=['Incorrect', 'Correct'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'cm_{safe_name}.png'), dpi=150)
        plt.close()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR, f'roc_{safe_name}.png'), dpi=150)
        plt.close()

        logger.info(f"Diagnostic plots (CM + ROC) saved for {model_name}")

    @staticmethod
    def generate_learning_curve(model: Any, X: pd.DataFrame,
                                 y: pd.Series, model_name: str):
        """Plot training vs. cross-validation learning curve."""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=Config.CV_FOLDS, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )
        plt.figure()
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', label="Cross-validation score")
        plt.fill_between(train_sizes,
                         np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                         np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
        plt.fill_between(train_sizes,
                         np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                         np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1)
        plt.title(f"Learning Curve: {model_name}")
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy Score")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.REPORTS_DIR,
                    f'lc_{model_name.lower().replace(" ", "_")}.png'), dpi=150)
        plt.close()
        logger.info(f"Learning curve saved for {model_name}")

    # ──────────────────────────────────────────────────────────────────────
    # Q6 — Explainable AI Suite
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def run_explainability_suite(model: Any, X_train: pd.DataFrame,
                                  X_test: pd.DataFrame, feature_names: List[str],
                                  model_name: str = "model",
                                  plot_global: bool = True):
        """
        Full XAI suite: global importance, SHAP summary, SHAP force plot, and LIME.

        Args:
            plot_global: If True, generate global feature importance bar chart.
        """
        logger.info(f"Running XAI suite for {model_name}...")
        safe_name = model_name.lower().replace(" ", "_")

        # ── Global Feature Importance ─────────────────────────────────────
        if plot_global:
            if hasattr(model, 'coef_'):
                # Logistic Regression — absolute coefficient values
                coefs = np.abs(model.coef_[0])
                imps = pd.Series(coefs, index=feature_names).sort_values(ascending=False)
                plt.figure(figsize=(10, 6))
                imps.plot(kind='barh', color='steelblue')
                plt.gca().invert_yaxis()
                plt.title(f"{model_name} — Global Feature Importance (|Coefficients|)")
                plt.xlabel("Absolute Coefficient Value")
                plt.tight_layout()
                plt.savefig(os.path.join(Config.REPORTS_DIR,
                            f'global_importance_{safe_name}.png'), dpi=150)
                plt.close()
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models — Gini / gain-based importance
                imps = pd.Series(model.feature_importances_,
                                 index=feature_names).sort_values(ascending=False)
                plt.figure(figsize=(10, 6))
                imps.plot(kind='barh', color='forestgreen')
                plt.gca().invert_yaxis()
                plt.title(f"{model_name} — Global Feature Importance (Tree-based)")
                plt.xlabel("Importance Score")
                plt.tight_layout()
                plt.savefig(os.path.join(Config.REPORTS_DIR,
                            f'global_importance_{safe_name}.png'), dpi=150)
                plt.close()

        # ── SHAP Summary Plot (Global) ────────────────────────────────────
        X_sample = X_test.sample(min(len(X_test), Config.SHAP_SAMPLE_SIZE),
                                  random_state=Config.RANDOM_STATE)
        shap_values = None
        explainer = None
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_viz = shap_values[1] if isinstance(shap_values, list) else shap_values

            plt.figure()
            shap.summary_plot(shap_viz, X_sample, show=False)
            plt.title(f"SHAP Summary — {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(Config.REPORTS_DIR,
                        f'shap_summary_{safe_name}.png'), dpi=150)
            plt.close()
            logger.info(f"SHAP summary plot saved for {model_name}")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer failed for {model_name}: {e}")

        # ── SHAP Force Plot (Local — Individual Student) ──────────────────
        if explainer is not None and shap_values is not None:
            try:
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                ev = (explainer.expected_value[1]
                      if isinstance(explainer.expected_value, (list, np.ndarray))
                      else explainer.expected_value)

                # Force plot for the first test sample (individual student explanation)
                shap.force_plot(
                    ev, sv[0, :], X_sample.iloc[0],
                    matplotlib=True, show=False
                )
                plt.title(f"SHAP Force Plot — Individual Student ({model_name})", fontsize=11)
                plt.tight_layout()
                plt.savefig(os.path.join(Config.REPORTS_DIR,
                            f'shap_force_{safe_name}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP force plot (local) saved for {model_name}")
            except Exception as e:
                logger.warning(f"SHAP force plot failed for {model_name}: {e}")

        # ── LIME Local Explanation ────────────────────────────────────────
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=feature_names,
                class_names=['At Risk', 'Success'],
                mode='classification'
            )
            exp = lime_explainer.explain_instance(
                X_test.iloc[0].values,
                model.predict_proba,
                num_features=Config.LIME_TOP_FEATURES
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation — Individual Student ({model_name})", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.REPORTS_DIR,
                        f'lime_local_{safe_name}.png'), dpi=150)
            plt.close()
            logger.info(f"LIME local explanation saved for {model_name}")
        except Exception as e:
            logger.warning(f"LIME failed for {model_name}: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Q6 — Partial Dependence Plots
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def generate_pdp(model: Any, X: pd.DataFrame, features: List[str]):
        """Partial Dependence Plots showing marginal effect of features on predictions."""
        try:
            fig, ax = plt.subplots(figsize=(15, 12))
            PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
            plt.suptitle("Partial Dependence Plots (PDP)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.REPORTS_DIR, 'pdp_analysis.png'), dpi=150)
            plt.close()
            logger.info("Partial Dependence Plots saved")
        except Exception as e:
            logger.warning(f"PDP generation failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Q5+Q6 — Comparative Model Analysis Report
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def generate_comparative_report(
        ml_results: Dict[str, Dict[str, Any]],
        dl_results: Dict[str, Dict[str, Any]],
        output_path: str
    ):
        """
        Generate a written comparative analysis of all models.

        Covers:
            - Performance comparison across all metrics
            - Model interpretability: transparent vs. black-box
            - Educator trust and decision-making implications
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COMPARATIVE MODEL ANALYSIS REPORT")
        lines.append("Student Performance Prediction System (SPPS)")
        lines.append("=" * 80)

        # ── Section 1: Performance Comparison ─────────────────────────────
        lines.append("\n1. PERFORMANCE COMPARISON")
        lines.append("-" * 40)
        lines.append(f"{'Model':<22} | {'Acc':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7} | {'AUC':>7}")
        lines.append("-" * 72)

        all_results = {}
        for name, m in ml_results.items():
            all_results[name] = m
            lines.append(
                f"{name:<22} | {m['accuracy']:>7.4f} | {m['f1']:>7.4f} | "
                f"{m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['roc_auc']:>7.4f}"
            )

        if dl_results:
            lines.append("-" * 72)
            lines.append("  (Deep Learning — evaluated on per-student sequences)")
            for name, m in dl_results.items():
                all_results[name] = m
                lines.append(
                    f"{name:<22} | {m['accuracy']:>7.4f} | {m['f1']:>7.4f} | "
                    f"{m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['roc_auc']:>7.4f}"
                )

        # Find best model by F1
        best_name = max(all_results, key=lambda k: all_results[k]['f1'])
        lines.append(f"\nBest overall model (by F1): {best_name} "
                      f"(F1={all_results[best_name]['f1']:.4f})")

        # ── Section 2: Model Interpretability Comparison ──────────────────
        lines.append("\n\n2. INTERPRETABILITY COMPARISON — Transparent vs. Black-Box")
        lines.append("-" * 40)
        lines.append("""
Logistic Regression (Transparent Model):
  - Coefficients directly indicate the direction and magnitude of each feature's
    influence on the prediction. For example, a positive coefficient for
    'student_ability' means higher ability increases the probability of success.
  - Advantages: Full transparency, easy to audit, no additional XAI tools needed.
  - Limitation: Cannot capture non-linear relationships or feature interactions.

Random Forest / XGBoost / LightGBM (Black-Box Models):
  - Internal decision-making is opaque — individual trees and splits are not
    human-interpretable at scale.
  - Require post-hoc explanation methods:
      * SHAP summary plots → global feature attribution with direction
      * SHAP force plots   → per-student prediction breakdown
      * LIME               → local linear approximation for individual cases
      * PDP                → marginal effect of a feature on predictions
  - Advantages: Higher predictive accuracy, capture complex interactions.
  - Limitation: Explanations are approximations; may not fully capture model logic.

LSTM / GRU (Deep Learning — Black-Box Models):
  - Process sequential student interactions to capture temporal learning patterns.
  - Operate on a different granularity (per-student vs. per-step), making direct
    metric comparison nuanced.
  - Require specialized explainability methods (attention weights, gradient-based).
  - Advantages: Can model how mastery evolves over a student's learning journey.
  - Limitation: Computationally expensive, harder to explain, need more data.""")

        # ── Section 3: Educator Trust & Decision-Making ───────────────────
        lines.append("\n\n3. IMPACT ON EDUCATOR TRUST AND DECISION-MAKING")
        lines.append("-" * 40)
        lines.append("""
Explainable AI (XAI) integration is critical for real-world adoption of predictive
models in educational settings. Key impacts include:

  a) Building Trust:
     - Educators are more likely to trust predictions when they can see WHY a
       student is flagged as at-risk. SHAP force plots showing "low consistency_index
       and declining mastery_trend drove this prediction" are far more actionable
       than a raw probability score.

  b) Actionable Interventions:
     - Global explanations (SHAP summary, feature importance) reveal systemic issues.
       For instance, if 'timeliness' is a top predictor, institutions can redesign
       assignment deadlines or add time-management workshops.
     - Local explanations (LIME, SHAP force) enable personalized interventions.
       A tutor seeing that a specific student's risk is driven by 'consistency_index'
       can recommend regular study schedules rather than generic support.

  c) Fairness and Accountability:
     - Transparent models (Logistic Regression) serve as auditable baselines.
       If a black-box model produces results that diverge significantly from the
       transparent baseline, it warrants investigation for potential bias.

  d) Recommended Deployment Strategy:
     - Use Logistic Regression as the interpretable baseline for stakeholder reports.
     - Use XGBoost/RF with SHAP explanations for high-accuracy early warning systems.
     - Use LSTM/GRU for longitudinal tracking of student progress over time.
     - Always accompany predictions with explanations to maintain educator trust.""")

        lines.append("\n" + "=" * 80)

        report_text = "\n".join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # Also print to console (handle Windows encoding)
        try:
            print(report_text)
        except UnicodeEncodeError:
            print(report_text.encode('ascii', errors='replace').decode('ascii'))
        logger.info(f"Comparative analysis report saved to {output_path}")
