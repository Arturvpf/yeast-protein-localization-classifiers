"""Testes automatizados de integridade metodológica e reprodutibilidade do pipeline Yeast."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_project_root,
    load_yeast_dataframe,
    prepare_features_and_target,
    split_train_test,
)
from src.evaluation import (
    compute_learning_curves_stratified,
    evaluate_predictions,
    extract_per_class_metrics,
)
from src.models import (
    build_baseline_models,
    build_tuned_search_spaces,
    fit_and_tune_all_models,
)


def test_dataset_loading_from_file_and_zip() -> None:
    repo = get_project_root()
    df_file = load_yeast_dataframe(repo / "data" / "yeast.data")
    df_zip = load_yeast_dataframe(repo / "data" / "yeast-datasets.zip")

    assert df_file.shape == (1484, 9)
    assert df_zip.shape == (1484, 9)
    assert list(df_file.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert df_file.isna().sum().sum() == 0
    assert df_file[TARGET_COLUMN].nunique() == 10
    assert df_file.equals(df_zip)


def test_stratified_split_preserves_proportions_and_disjoint_indices() -> None:
    df = load_yeast_dataframe()
    X, y, label_encoder = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.30, random_state=42)

    assert len(X_train) == 1038
    assert len(X_test) == 446
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(np.unique(y_train)) == 10
    assert len(np.unique(y_test)) == 10
    assert len(label_encoder.classes_) == 10


def test_search_spaces_prevent_leakage_and_enforce_euclidean_knn() -> None:
    spaces = build_tuned_search_spaces(random_state=42)
    expected_keys = {
        "DecisionTree",
        "GaussianNB",
        "LogisticRegression",
        "KNN",
        "RandomForest",
    }
    assert set(spaces.keys()) == expected_keys

    for key, (pipe, _) in spaces.items():
        assert isinstance(pipe, Pipeline), f"{key} deve usar Pipeline para evitar vazamento"
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)

    knn_clf = spaces["KNN"][0].named_steps["clf"]
    assert knn_clf.metric == "euclidean"


def test_tuning_improves_naive_bayes_and_evaluates_all_metrics() -> None:
    df = load_yeast_dataframe()
    X, y, le = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.30, random_state=42)

    baselines = build_baseline_models(random_state=42)
    gnb_base = baselines["GaussianNB"].fit(X_train, y_train)
    base_metrics = evaluate_predictions(y_test, gnb_base.predict(X_test))

    tuned = fit_and_tune_all_models(X_train, y_train, n_splits=3, random_state=42, n_jobs=1)
    gnb_tuned = tuned["GaussianNB"].best_estimator
    tuned_metrics = evaluate_predictions(y_test, gnb_tuned.predict(X_test))

    # Sem ajuste de var_smoothing, GaussianNB colapsa (~14%); ajustado, supera 55%
    assert base_metrics["accuracy"] < 0.20
    assert tuned_metrics["accuracy"] > 0.55
    assert tuned_metrics["f1_macro"] > base_metrics["f1_macro"] + 0.15

    rf_metrics = evaluate_predictions(
        y_test, tuned["RandomForest"].best_estimator.predict(X_test)
    )
    assert rf_metrics["accuracy"] >= 0.59
    assert rf_metrics["f1_macro"] >= 0.59

    per_cls = extract_per_class_metrics(
        y_test,
        tuned["RandomForest"].best_estimator.predict(X_test),
        le.classes_,
        "RandomForest",
    )
    assert len(per_cls) == 10
    assert per_cls["support"].sum() == len(y_test)


def test_stratified_learning_curves_cover_5_to_95_percent() -> None:
    df = load_yeast_dataframe()
    X, y, _ = prepare_features_and_target(df)
    spaces = build_tuned_search_spaces(random_state=42)
    subset_models = {
        "KNN": spaces["KNN"][0],
        "GaussianNB": spaces["GaussianNB"][0],
    }

    lc_df = compute_learning_curves_stratified(
        subset_models,
        X,
        y,
        n_repeats=1,
        random_state=42,
    )

    # 19 proporções (5%, 10%, ..., 95%) x 2 modelos = 38 linhas
    assert len(lc_df) == 38
    expected_pcts = list(range(5, 100, 5))
    for model_key in subset_models:
        sub = lc_df[lc_df["model"] == model_key].sort_values("train_pct")
        assert sub["train_pct"].tolist() == expected_pcts
        assert (sub["train_pct"] + sub["test_pct"] == 100).all()
        assert (sub["n_train"] + sub["n_test"] == 1484).all()
        for metric in ["precision_macro", "recall_macro", "f1_macro", "accuracy"]:
            assert f"train_{metric}_mean" in sub.columns
            assert f"test_{metric}_mean" in sub.columns
            assert sub[f"test_{metric}_mean"].between(0.0, 1.0).all()
