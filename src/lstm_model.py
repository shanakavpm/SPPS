"""
LSTM & GRU sequence models for student performance prediction.

Rationale:
    Educational interactions are inherently sequential — a student's mastery evolves
    over time. Recurrent models (LSTM/GRU) can capture these temporal dependencies
    that traditional ML models treat as independent rows.

Approach:
    1. Group step-level data by student, ordered chronologically.
    2. Truncate/pad each student's interaction sequence to a fixed length.
    3. Target: overall student performance (mean correctness > 0.5 → Pass).
    4. Train LSTM and GRU models with early stopping.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)

# Suppress TensorFlow info logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SequenceModelEvaluator:
    """Builds, trains, and evaluates LSTM and GRU on per-student sequences."""

    MAX_SEQ_LEN = 50   # Maximum number of steps per student sequence
    EPOCHS = 30
    BATCH_SIZE = 32

    @classmethod
    def prepare_sequences(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        student_col: str = 'Anon Student Id',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert step-level data into per-student padded sequences.

        Each student's chronologically-ordered interactions become one sample.
        Target: determined by adaptive threshold (median of student means)
        to ensure balanced classes regardless of dataset characteristics.
        """
        df = X.copy()
        df['_target'] = y.values

        student_means = []
        sequences = []

        for _, group in df.groupby(student_col):
            seq = group[feature_cols].values[-cls.MAX_SEQ_LEN:]
            sequences.append(seq)
            student_means.append(group['_target'].mean())

        # Adaptive threshold: use median of student means for balanced split
        threshold = np.median(student_means)
        labels = [int(m > threshold) for m in student_means]

        # Pad shorter sequences with zeros (pre-padding)
        X_seq = pad_sequences(
            sequences, maxlen=cls.MAX_SEQ_LEN,
            dtype='float32', padding='pre', truncating='pre', value=0.0
        )
        y_seq = np.array(labels)

        class_counts = np.bincount(y_seq)
        logger.info(f"Sequence data: {X_seq.shape[0]} students, "
                     f"seq_len={X_seq.shape[1]}, features={X_seq.shape[2]}, "
                     f"class balance: {class_counts}, threshold={threshold:.3f}")
        return X_seq, y_seq

    @staticmethod
    def _build_lstm(input_shape: tuple) -> Sequential:
        """Two-layer LSTM with dropout for regularization."""
        model = Sequential([
            Masking(mask_value=0.0, input_shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def _build_gru(input_shape: tuple) -> Sequential:
        """Two-layer GRU — lighter alternative to LSTM with comparable performance."""
        model = Sequential([
            Masking(mask_value=0.0, input_shape=input_shape),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @classmethod
    def evaluate(
        cls,
        X_all: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        End-to-end training and evaluation of LSTM and GRU models.

        Returns a dictionary of results keyed by model name, each containing
        accuracy, precision, recall, f1, roc_auc, and the trained model.
        """
        logger.info("Preparing student sequences for deep learning models...")

        # Scale features before sequencing
        scaler = StandardScaler()
        numeric_features = X_all[feature_cols].copy()
        numeric_features[feature_cols] = scaler.fit_transform(numeric_features)
        X_scaled = X_all.copy()
        X_scaled[feature_cols] = numeric_features[feature_cols]

        X_seq, y_seq = cls.prepare_sequences(X_scaled, y, feature_cols)

        # Stratify if both classes have enough samples
        stratify_arg = y_seq if min(np.bincount(y_seq)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )

        input_shape = (X_train.shape[1], X_train.shape[2])
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        results = {}

        for name, build_fn in [('LSTM', cls._build_lstm), ('GRU', cls._build_gru)]:
            logger.info(f"Training {name} model...")
            model = build_fn(input_shape)

            model.fit(
                X_train, y_train,
                epochs=cls.EPOCHS,
                batch_size=cls.BATCH_SIZE,
                validation_split=0.15,
                callbacks=[early_stop],
                verbose=0
            )

            y_prob = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_prob > 0.5).astype(int)

            # Handle ROC-AUC when only one class present in test set
            try:
                auc_val = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc_val = 0.0
                logger.warning(f"{name}: ROC-AUC undefined (single class in test set)")

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': auc_val,
                'model': model
            }
            logger.info(f"{name} — Acc: {results[name]['accuracy']:.4f}, "
                         f"F1: {results[name]['f1']:.4f}, "
                         f"AUC: {results[name]['roc_auc']:.4f}")

        return results
