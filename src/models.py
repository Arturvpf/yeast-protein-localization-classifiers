"""Definição dos 5 classificadores, espaços de busca e otimização via GridSearchCV sem vazamento de dados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "DecisionTree": "Árvore de Decisão",
    "GaussianNB": "Bayesiano Ingênuo",
    "LogisticRegression": "Regressão Logística",
    "KNN": "k-Vizinhos (Euclidiano)",
    "RandomForest": "Random Forest",
}


@dataclass
class TunedModelResult:
    """Resultado consolidado da otimização de hiperparâmetros de um classificador."""

    model_key: str
    display_name: str
    grid_search: GridSearchCV
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_cv_score: float


def build_cv_splitter(
    n_splits: int = 3,
    random_state: int = 42,
) -> StratifiedKFold:
    """Retorna validação cruzada estratificada compatível com a menor classe no treino (ERL=3)."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def build_baseline_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Constrói os 5 classificadores em sua configuração padrão (sem ajuste de hiperparâmetros)."""
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(
            random_state=random_state, max_iter=2000
        ),
        "KNN": KNeighborsClassifier(metric="euclidean"),
        "RandomForest": RandomForestClassifier(random_state=random_state),
    }


def build_tuned_search_spaces(
    random_state: int = 42,
) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Define Pipelines (StandardScaler + Classificador) e grades de hiperparâmetros para os 5 modelos."""
    return {
        "DecisionTree": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", DecisionTreeClassifier(random_state=random_state)),
                ]
            ),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 5, 7, 9, 12],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__ccp_alpha": [0.0, 0.003],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "GaussianNB": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", GaussianNB()),
                ]
            ),
            {
                "clf__var_smoothing": np.logspace(-9, 0.2, 40),
            },
        ),
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(random_state=random_state, max_iter=3000),
                    ),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
                "clf__solver": ["lbfgs", "newton-cg"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(metric="euclidean")),
                ]
            ),
            {
                "clf__n_neighbors": list(range(1, 32, 2)),
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "RandomForest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(random_state=random_state)),
                ]
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 8, 10, 12],
                "clf__min_samples_split": [2, 8],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def fit_and_tune_all_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scoring: str = "f1_macro",
    n_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Dict[str, TunedModelResult]:
    """Executa GridSearchCV estratificado para os 5 classificadores exclusivamente sobre X_train."""
    cv = build_cv_splitter(n_splits=n_splits, random_state=random_state)
    search_spaces = build_tuned_search_spaces(random_state=random_state)
    results: Dict[str, TunedModelResult] = {}

    for key, (pipeline, param_grid) in search_spaces.items():
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        gs.fit(X_train, y_train)
        clean_params = {
            k.replace("clf__", ""): (
                float(v) if isinstance(v, np.floating) else v
            )
            for k, v in gs.best_params_.items()
        }
        results[key] = TunedModelResult(
            model_key=key,
            display_name=MODEL_DISPLAY_NAMES[key],
            grid_search=gs,
            best_estimator=gs.best_estimator_,
            best_params=clean_params,
            best_cv_score=float(gs.best_score_),
        )

    return results


def compute_knn_validation_curve(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k_values: list[int] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Avalia o número de vizinhos k no conjunto de validação cruzada usando distância Euclidiana."""
    if k_values is None:
        k_values = list(range(1, 32, 2))
    cv = build_cv_splitter(n_splits=n_splits, random_state=random_state)
    scoring_dict = {
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "accuracy": "accuracy",
    }

    rows = []
    for weight_mode in ["uniform", "distance"]:
        for k in k_values:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=k,
                            metric="euclidean",
                            weights=weight_mode,
                        ),
                    ),
                ]
            )
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring_dict,
                return_train_score=False,
                n_jobs=1,
            )
            rows.append(
                {
                    "k": k,
                    "weights": weight_mode,
                    "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                    "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
                    "cv_precision_macro_mean": float(
                        np.mean(scores["test_precision_macro"])
                    ),
                    "cv_recall_macro_mean": float(
                        np.mean(scores["test_recall_macro"])
                    ),
                    "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
                }
            )
    return pd.DataFrame(rows)
