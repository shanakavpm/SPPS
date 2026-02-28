import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

class ModelEvaluator:
    """Strategy pattern for training and analyzing multiple models."""
    def __init__(self, models: Dict[str, Any]):
        self.models = models

    def run_eval(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Trains each model and returns summarized performance metrics."""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'model': model
            }
        return results
