"""Métricas de avaliação multiclasse e curvas de aprendizagem estratificadas (5%/95% a 95%/5%)."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas globais (macro, ponderadas e acurácia) sem warnings de divisão por zero."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def extract_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
    model_name: str,
) -> pd.DataFrame:
    """Extrai precisão, cobertura (recall), F1-score e suporte por classe em formato tabular."""
    names = list(class_names)
    report = classification_report(
        y_true,
        y_pred,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls_name in names:
        cls_stats = report[cls_name]
        rows.append(
            {
                "model": model_name,
                "class": cls_name,
                "precision": float(cls_stats["precision"]),
                "recall": float(cls_stats["recall"]),
                "f1_score": float(cls_stats["f1-score"]),
                "support": int(cls_stats["support"]),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
) -> pd.DataFrame:
    """Retorna matriz de confusão rotulada como DataFrame."""
    names = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(names)))
    return pd.DataFrame(cm, index=names, columns=names)


def compute_learning_curves_stratified(
    best_models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: np.ndarray,
    train_fractions: Iterable[float] | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula curvas de aprendizagem de (5%, 95%) a (95%, 5%) com passo de 5% via amostragem estratificada.

    Segue rigorosamente a especificação do projeto:
    - Pares (treino, teste) de (5%, 95%) até (95%, 5%) do conjunto original com passo de 5%;
    - Amostragem estratificada em cada partição;
    - Padronização ajustada exclusivamente no subconjunto de treino (via Pipeline clonado);
    - Cálculo de Precision, Recall, F1-Score (macro e weighted) e Accuracy tanto no conjunto
      de treinamento quanto no conjunto de teste.
    """
    if train_fractions is None:
        fractions = [round(0.05 * i, 2) for i in range(1, 20)]
    else:
        fractions = [round(float(f), 4) for f in train_fractions]

    records: list[dict[str, float | int | str]] = []

    for frac in fractions:
        test_frac = round(1.0 - frac, 4)
        train_pct = int(round(frac * 100))
        test_pct = int(round(test_frac * 100))

        split_cache = []
        for rep in range(n_repeats):
            seed = random_state + rep * 100 + train_pct
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                train_size=frac,
                stratify=y,
                random_state=seed,
            )
            split_cache.append((X_tr, X_te, y_tr, y_te))

        for model_key, estimator in best_models.items():
            rep_train_metrics: Dict[str, list[float]] = {}
            rep_test_metrics: Dict[str, list[float]] = {}
            n_train_samples = 0
            n_test_samples = 0

            for X_tr, X_te, y_tr, y_te in split_cache:
                n_train_samples = len(y_tr)
                n_test_samples = len(y_te)
                fitted = clone(estimator)
                fitted.fit(X_tr, y_tr)

                tr_scores = evaluate_predictions(y_tr, fitted.predict(X_tr))
                te_scores = evaluate_predictions(y_te, fitted.predict(X_te))

                for k, v in tr_scores.items():
                    rep_train_metrics.setdefault(k, []).append(v)
                for k, v in te_scores.items():
                    rep_test_metrics.setdefault(k, []).append(v)

            row: dict[str, float | int | str] = {
                "model": model_key,
                "train_fraction": frac,
                "test_fraction": test_frac,
                "train_pct": train_pct,
                "test_pct": test_pct,
                "split_label": f"{train_pct}%/{test_pct}%",
                "n_train": n_train_samples,
                "n_test": n_test_samples,
            }
            for metric_name in rep_train_metrics:
                row[f"train_{metric_name}_mean"] = float(
                    np.mean(rep_train_metrics[metric_name])
                )
                row[f"train_{metric_name}_std"] = float(
                    np.std(rep_train_metrics[metric_name])
                )
                row[f"test_{metric_name}_mean"] = float(
                    np.mean(rep_test_metrics[metric_name])
                )
                row[f"test_{metric_name}_std"] = float(
                    np.std(rep_test_metrics[metric_name])
                )
            records.append(row)

    return pd.DataFrame(records)
