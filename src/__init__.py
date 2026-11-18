"""Yeast Protein Localization Classifiers — pipeline modular de Aprendizado de Máquina."""

from .data import (
    CLASS_DESCRIPTIONS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    TARGET_COLUMN,
    load_yeast_dataframe,
    prepare_features_and_target,
    split_train_test,
)
from .evaluation import (
    compute_learning_curves_stratified,
    evaluate_predictions,
    extract_per_class_metrics,
)
from .models import (
    MODEL_DISPLAY_NAMES,
    build_baseline_models,
    build_cv_splitter,
    build_tuned_search_spaces,
    fit_and_tune_all_models,
)

__all__ = [
    "CLASS_DESCRIPTIONS",
    "FEATURE_COLUMNS",
    "FEATURE_DESCRIPTIONS",
    "MODEL_DISPLAY_NAMES",
    "TARGET_COLUMN",
    "build_baseline_models",
    "build_cv_splitter",
    "build_tuned_search_spaces",
    "compute_learning_curves_stratified",
    "evaluate_predictions",
    "extract_per_class_metrics",
    "fit_and_tune_all_models",
    "load_yeast_dataframe",
    "prepare_features_and_target",
    "split_train_test",
]
