"""Script principal para execução reprodutível de todo o pipeline experimental (Base Yeast)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import (
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
    MODEL_DISPLAY_NAMES,
    build_baseline_models,
    compute_knn_validation_curve,
    fit_and_tune_all_models,
)
from src.plotting import (
    plot_baseline_vs_tuned,
    plot_bivariate_boxplots,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_correlation_matrix,
    plot_knn_validation_curve,
    plot_learning_curve_for_metric,
    plot_learning_curves_summary_panel,
    plot_univariate_distributions,
)


def run_full_pipeline(
    output_dir: Path,
    lc_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Executa EDA, treinamento, otimização, avaliação no teste e curvas de aprendizagem."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("1. Carregando a base de dados Yeast...")
    df = load_yeast_dataframe()
    X, y, label_encoder = prepare_features_and_target(df)
    class_names = list(label_encoder.classes_)
    print(f"   Amostras: {len(df)} | Features: {X.shape[1]} | Classes: {len(class_names)}")

    print("2. Gerando figuras de Análise Exploratória de Dados (EDA)...")
    for fig in [
        plot_class_distribution(df, figures_dir / "01_class_distribution.png"),
        plot_univariate_distributions(df, figures_dir / "02_univariate_distributions.png"),
        plot_bivariate_boxplots(df, figures_dir / "03_bivariate_boxplots.png"),
        plot_correlation_matrix(df, figures_dir / "04_correlation_matrix.png"),
    ]:
        plt.close(fig)

    print("3. Dividindo dados de forma estratificada (70% treino+validação / 30% teste)...")
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.30, random_state=random_state
    )
    print(f"   Treino+Validação: {len(X_train)} | Teste (retido): {len(X_test)}")

    print("4. Avaliando modelos Baseline (configuração padrão sem ajuste)...")
    comparison_rows = []
    baselines = build_baseline_models(random_state=random_state)
    for key, clf in baselines.items():
        clf.fit(X_train, y_train)
        y_pred_base = clf.predict(X_test)
        scores_base = evaluate_predictions(y_test, y_pred_base)
        comparison_rows.append(
            {
                "model": key,
                "display_name": MODEL_DISPLAY_NAMES[key],
                "stage": "Baseline (Padrão)",
                "cv_f1_macro": None,
                **scores_base,
            }
        )

    print("5. Calculando curva de validação para escolha de k no k-NN (distância Euclidiana)...")
    knn_cv_df = compute_knn_validation_curve(
        X_train, y_train, n_splits=3, random_state=random_state
    )
    knn_cv_df.to_csv(output_dir / "knn_validation_curve.csv", index=False)
    plt.close(
        plot_knn_validation_curve(knn_cv_df, figures_dir / "05_knn_validation_curve.png")
    )

    print("6. Otimizando hiperparâmetros dos 5 classificadores via GridSearchCV estratificado...")
    tuned_results = fit_and_tune_all_models(
        X_train,
        y_train,
        scoring="f1_macro",
        n_splits=3,
        random_state=random_state,
        n_jobs=1,
    )

    best_estimators = {}
    predictions_tuned = {}
    per_class_frames = []
    hyperparams_payload = {}

    for key, res in tuned_results.items():
        best_estimators[key] = res.best_estimator
        y_pred = res.best_estimator.predict(X_test)
        predictions_tuned[key] = y_pred
        scores_tuned = evaluate_predictions(y_test, y_pred)

        comparison_rows.append(
            {
                "model": key,
                "display_name": res.display_name,
                "stage": "Otimizado (GridSearchCV)",
                "cv_f1_macro": res.best_cv_score,
                **scores_tuned,
            }
        )
        per_class_frames.append(
            extract_per_class_metrics(y_test, y_pred, class_names, key)
        )
        hyperparams_payload[key] = {
            "display_name": res.display_name,
            "best_cv_f1_macro": round(res.best_cv_score, 6),
            "best_params": res.best_params,
            "test_metrics": {k: round(v, 6) for k, v in scores_tuned.items()},
        }

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "baseline_vs_tuned_metrics.csv", index=False)

    per_class_df = pd.concat(per_class_frames, ignore_index=True)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    (output_dir / "best_hyperparameters.json").write_text(
        json.dumps(hyperparams_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plt.close(
        plot_baseline_vs_tuned(comparison_df, figures_dir / "06_baseline_vs_tuned.png")
    )
    plt.close(
        plot_confusion_matrices(
            predictions_tuned,
            y_test,
            class_names,
            figures_dir / "07_confusion_matrices.png",
        )
    )

    print(
        f"7. Calculando curvas de aprendizagem de (5%, 95%) a (95%, 5%) com passo de 5% "
        f"(n_repeats={lc_repeats})..."
    )
    lc_df = compute_learning_curves_stratified(
        best_estimators,
        X,
        y,
        n_repeats=lc_repeats,
        random_state=random_state,
    )
    lc_df.to_csv(output_dir / "learning_curves_metrics.csv", index=False)

    for fig in [
        plot_learning_curve_for_metric(
            lc_df,
            "precision_macro",
            "Precision (Macro)",
            figures_dir / "08_learning_curves_precision.png",
        ),
        plot_learning_curve_for_metric(
            lc_df,
            "recall_macro",
            "Recall / Cobertura (Macro)",
            figures_dir / "09_learning_curves_recall.png",
        ),
        plot_learning_curve_for_metric(
            lc_df,
            "f1_macro",
            "F1-Score (Macro)",
            figures_dir / "10_learning_curves_f1.png",
        ),
        plot_learning_curves_summary_panel(
            lc_df,
            figures_dir / "11_learning_curves_comparison.png",
        ),
    ]:
        plt.close(fig)

    tuned_only = comparison_df[
        comparison_df["stage"] == "Otimizado (GridSearchCV)"
    ].copy()
    print("\n=== Resumo Final no Conjunto de Teste (Modelos Otimizados) ===")
    summary_cols = [
        "display_name",
        "cv_f1_macro",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]
    print(tuned_only[summary_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArtefatos e figuras salvos em: {output_dir.resolve()}")

    return {
        "comparison": comparison_df,
        "per_class": per_class_df,
        "learning_curves": lc_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de classificação de localização de proteínas (UCI Yeast)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Diretório de saída para tabelas CSV/JSON e gráficos PNG (padrão: results/).",
    )
    parser.add_argument(
        "--lc-repeats",
        type=int,
        default=5,
        help="Número de repetições estratificadas por proporção nas curvas de aprendizagem (padrão: 5).",
    )
    args = parser.parse_args()
    run_full_pipeline(output_dir=args.output_dir, lc_repeats=args.lc_repeats)


if __name__ == "__main__":
    main()
