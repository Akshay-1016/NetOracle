"""
Static Logistic Regression Baseline for Formal Comparison
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from typing import Dict

class BaselineModel:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)

    def evaluate(self, X_seq: np.ndarray, y_attack: np.ndarray) -> Dict[str, float]:
        X_static = X_seq[:, -1, :]
        y_target = (y_attack[:, 0] > 0.5).astype(int)
        
        split = int(len(X_static) * 0.8)
        X_train, X_test = X_static[:split], X_static[split:]
        y_train, y_test = y_target[:split], y_target[split:]
        
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        probs = self.clf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else preds
        
        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'f1_score': float(f1_score(y_test, preds, zero_division=0)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5
        }
