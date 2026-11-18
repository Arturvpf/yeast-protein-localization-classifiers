"""Visualizações estatísticas, diagnósticos de validação e curvas de aprendizagem em alta resolução."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .data import FEATURE_COLUMNS, TARGET_COLUMN
from .models import MODEL_DISPLAY_NAMES

PALETTE_MODELS: dict[str, str] = {
    "DecisionTree": "#d95f02",
    "GaussianNB": "#7570b3",
    "LogisticRegression": "#1f78b4",
    "KNN": "#e7298a",
    "RandomForest": "#1b9e77",
}


def configure_plot_style() -> None:
    """Aplica tema visual consistente e legível para relatórios científicos."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titleweight"] = "bold"


def _save_if_requested(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")


def plot_class_distribution(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota a distribuição desbalanceada das 10 classes de localização celular."""
    configure_plot_style()
    counts = df[TARGET_COLUMN].value_counts()
    pcts = counts / len(df) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=sns.color_palette("viridis", len(counts)),
        edgecolor="#222222",
        linewidth=0.7,
    )
    for bar, cnt, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 8,
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_title("Distribuição das Classes de Localização Celular (Base Yeast, N = 1.484)")
    ax.set_xlabel("Classe de Localização (class_protein)")
    ax.set_ylabel("Número de Proteínas")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_univariate_distributions(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota histogramas com densidade (KDE) para os 8 atributos preditivos."""
    configure_plot_style()
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURE_COLUMNS):
        ax = axes_flat[idx]
        unique_vals = df[feat].nunique()
        use_kde = unique_vals > 5
        sns.histplot(
            df[feat],
            bins=min(30, max(5, unique_vals)),
            kde=use_kde,
            color="#2b6cb0",
            edgecolor="white",
            alpha=0.8,
            ax=ax,
        )
        mean_val = df[feat].mean()
        median_val = df[feat].median()
        ax.axvline(mean_val, color="#c53030", linestyle="--", linewidth=1.3, label=f"Média={mean_val:.2f}")
        ax.axvline(median_val, color="#2f855a", linestyle=":", linewidth=1.3, label=f"Mediana={median_val:.2f}")
        ax.set_title(f"Atributo: {feat}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Análise Univariada — Distribuição das 8 Features Preditoras", fontsize=15, y=1.01)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_bivariate_boxplots(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota boxplots de todas as 8 features preditoras estratificadas pelas 10 classes."""
    configure_plot_style()
    class_order = df[TARGET_COLUMN].value_counts().index.tolist()
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 14))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURE_COLUMNS):
        ax = axes_flat[idx]
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feat,
            order=class_order,
            hue=TARGET_COLUMN,
            hue_order=class_order,
            palette="tab10",
            legend=False,
            fliersize=2.0,
            linewidth=0.9,
            ax=ax,
        )
        ax.set_title(f"Distribuição de '{feat}' por Classe")
        ax.set_xlabel("Classe")
        ax.set_ylabel(feat)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Análise Bivariada — Relação entre Atributos Preditivos e Localização Celular", fontsize=15, y=1.01)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota matriz de correlação de Pearson entre os 8 atributos preditivos."""
    configure_plot_style()
    corr = df[FEATURE_COLUMNS].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.82, "label": "Correlação de Pearson (r)"},
        ax=ax,
    )
    ax.set_title("Matriz de Correlação entre Atributos Preditivos (Base Yeast)")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_knn_validation_curve(
    knn_cv_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota curva de validação cruzada para escolha do hiperparâmetro k no k-NN Euclidiano."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    styles = {
        "uniform": ("o-", "#1f78b4", "Pesos Uniformes (uniform)"),
        "distance": ("s-", "#e7298a", "Ponderado pela Distância (distance)"),
    }

    for w_mode, (fmt, color, label) in styles.items():
        sub = knn_cv_df[knn_cv_df["weights"] == w_mode].sort_values("k")
        axes[0].plot(sub["k"], sub["cv_f1_macro_mean"], fmt, color=color, label=label, linewidth=2)
        axes[0].fill_between(
            sub["k"],
            sub["cv_f1_macro_mean"] - sub["cv_f1_macro_std"],
            sub["cv_f1_macro_mean"] + sub["cv_f1_macro_std"],
            alpha=0.14,
            color=color,
        )
        axes[1].plot(sub["k"], sub["cv_accuracy_mean"], fmt, color=color, label=label, linewidth=2)

    best_idx = knn_cv_df["cv_f1_macro_mean"].idxmax()
    best_row = knn_cv_df.loc[best_idx]
    axes[0].axvline(best_row["k"], color="#222222", linestyle="--", linewidth=1.2, label=f"Melhor k = {int(best_row['k'])}")
    axes[1].axvline(best_row["k"], color="#222222", linestyle="--", linewidth=1.2, label=f"Melhor k = {int(best_row['k'])}")

    axes[0].set_title("Seleção de k via Validação Cruzada (F1-Macro)")
    axes[0].set_xlabel("Número de Vizinhos (k) — Distância Euclidiana")
    axes[0].set_ylabel("F1-Score Macro (Validação Cruzada)")
    axes[0].set_xticks(sorted(knn_cv_df["k"].unique()))
    axes[0].legend(fontsize=9)

    axes[1].set_title("Acurácia na Validação Cruzada vs. Número de Vizinhos (k)")
    axes[1].set_xlabel("Número de Vizinhos (k) — Distância Euclidiana")
    axes[1].set_ylabel("Acurácia Média (Validação Cruzada)")
    axes[1].set_xticks(sorted(knn_cv_df["k"].unique()))
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_baseline_vs_tuned(
    comparison_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compara modelos Baseline (padrão) vs. Otimizados (Pipeline + GridSearchCV) no conjunto de teste."""
    configure_plot_style()
    metrics = [
        ("f1_macro", "F1-Score (Macro)"),
        ("precision_macro", "Precision (Macro)"),
        ("recall_macro", "Recall (Macro)"),
        ("accuracy", "Acurácia Global"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharey=True)
    axes_flat = axes.flatten()

    for ax, (col, title) in zip(axes_flat, metrics):
        sns.barplot(
            data=comparison_df,
            x="display_name",
            y=col,
            hue="stage",
            palette={"Baseline (Padrão)": "#a0aec0", "Otimizado (GridSearchCV)": "#2b6cb0"},
            edgecolor="#1a202c",
            linewidth=0.6,
            ax=ax,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Score no Conjunto de Teste (30%)")
        ax.set_ylim(0.0, 0.80)
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="Configuração", fontsize=8.5, title_fontsize=9)

    fig.suptitle("Impacto da Otimização de Hiperparâmetros e Padronização no Conjunto de Teste", fontsize=14.5, y=1.01)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_confusion_matrices(
    predictions_by_model: Mapping[str, np.ndarray],
    y_test: np.ndarray,
    class_names: Iterable[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota matrizes de confusão dos 5 classificadores otimizados no conjunto de teste."""
    configure_plot_style()
    names = list(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    axes_flat = axes.flatten()

    for idx, (model_key, y_pred) in enumerate(predictions_by_model.items()):
        ax = axes_flat[idx]
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(names)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
            cbar=False,
            linewidths=0.4,
            linecolor="#e2e8f0",
            ax=ax,
        )
        disp = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        ax.set_title(f"{disp}")
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")

    for extra_idx in range(len(predictions_by_model), len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    fig.suptitle("Matrizes de Confusão no Conjunto de Teste (30%, N = 446)", fontsize=15, y=1.01)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_learning_curve_for_metric(
    lc_df: pd.DataFrame,
    metric_key: str,
    metric_title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota curvas de aprendizagem (Treino vs. Teste de 5%/95% a 95%/5%) para os 5 classificadores."""
    configure_plot_style()
    model_keys = list(MODEL_DISPLAY_NAMES.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharey=True)
    axes_flat = axes.flatten()

    tr_mean_col = f"train_{metric_key}_mean"
    tr_std_col = f"train_{metric_key}_std"
    te_mean_col = f"test_{metric_key}_mean"
    te_std_col = f"test_{metric_key}_std"

    for idx, m_key in enumerate(model_keys):
        ax = axes_flat[idx]
        sub = lc_df[lc_df["model"] == m_key].sort_values("train_pct")
        x = sub["train_pct"].to_numpy()

        tr_mean = sub[tr_mean_col].to_numpy()
        tr_std = sub[tr_std_col].to_numpy()
        te_mean = sub[te_mean_col].to_numpy()
        te_std = sub[te_std_col].to_numpy()

        ax.plot(x, tr_mean, "o--", color="#c53030", linewidth=1.8, markersize=4.5, label="Treino (p%)")
        ax.fill_between(x, np.clip(tr_mean - tr_std, 0, 1), np.clip(tr_mean + tr_std, 0, 1), color="#c53030", alpha=0.12)

        ax.plot(x, te_mean, "s-", color="#2b6cb0", linewidth=2.1, markersize=4.8, label="Teste (100-p%)")
        ax.fill_between(x, np.clip(te_mean - te_std, 0, 1), np.clip(te_mean + te_std, 0, 1), color="#2b6cb0", alpha=0.16)

        ax.set_title(MODEL_DISPLAY_NAMES[m_key])
        ax.set_xlabel("% Treino (complementar: % Teste = 100 - % Treino)")
        ax.set_ylabel(metric_title)
        ax.set_xticks(range(5, 100, 10))
        ax.set_ylim(0.05, 1.03)
        ax.legend(loc="lower right", fontsize=9)

    # Sexto painel: comparação direta da curva de TESTE entre os 5 classificadores
    ax_cmp = axes_flat[5]
    for m_key in model_keys:
        sub = lc_df[lc_df["model"] == m_key].sort_values("train_pct")
        ax_cmp.plot(
            sub["train_pct"],
            sub[te_mean_col],
            "o-",
            color=PALETTE_MODELS[m_key],
            linewidth=2.0,
            markersize=4.2,
            label=MODEL_DISPLAY_NAMES[m_key],
        )
    ax_cmp.set_title(f"Comparativo no Teste — {metric_title}")
    ax_cmp.set_xlabel("% Treino (5%/95% até 95%/5%, passo 5%)")
    ax_cmp.set_ylabel(f"{metric_title} (Teste)")
    ax_cmp.set_xticks(range(5, 100, 10))
    ax_cmp.set_ylim(0.05, 1.03)
    ax_cmp.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        f"Curvas de Aprendizagem ({metric_title}) — Partições Estratificadas de (5%, 95%) a (95%, 5%)",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_learning_curves_summary_panel(
    lc_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Painel comparativo sintetizando Precision, Recall, F1-Score e Acurácia no conjunto de teste."""
    configure_plot_style()
    metrics = [
        ("precision_macro", "Precision (Macro)"),
        ("recall_macro", "Recall / Cobertura (Macro)"),
        ("f1_macro", "F1-Score (Macro)"),
        ("accuracy", "Acurácia Global"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharey=False)
    axes_flat = axes.flatten()

    for ax, (m_col, m_label) in zip(axes_flat, metrics):
        for m_key, disp_name in MODEL_DISPLAY_NAMES.items():
            sub = lc_df[lc_df["model"] == m_key].sort_values("train_pct")
            ax.plot(
                sub["train_pct"],
                sub[f"test_{m_col}_mean"],
                "o-",
                color=PALETTE_MODELS[m_key],
                linewidth=2.0,
                markersize=4.2,
                label=disp_name,
            )
        ax.set_title(f"Desempenho no Teste — {m_label}")
        ax.set_xlabel("Proporção de Treinamento (% do Dataset Original)")
        ax.set_ylabel(m_label)
        ax.set_xticks(range(5, 100, 10))
        ax.set_ylim(0.15, 0.75)
        ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(
        "Comparação Global das Curvas de Aprendizagem no Conjunto de Teste (5%/95% a 95%/5%)",
        fontsize=14.5,
        y=1.01,
    )
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig
