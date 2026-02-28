import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from .config import Config

class ModelEvaluator:
    """Strategy pattern for training, analyzing, and cross-validating multiple models."""
    def __init__(self, models: Dict[str, Any]):
        self.models = models

    def run_eval(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Trains each model and returns summarized performance metrics with cross-validation."""
        results = {}
        skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        for name, model in self.models.items():
            # Perform Cross-Validation on training set
            cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)
            
            # Train on full training set for test evaluation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'cv_accuracy_mean': np.mean(cv_results['test_accuracy']),
                'cv_accuracy_std': np.std(cv_results['test_accuracy']),
                'model': model
            }
        return results
