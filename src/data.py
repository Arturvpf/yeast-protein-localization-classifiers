"""Carregamento, validação e divisão estratificada da base UCI Yeast."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS: list[str] = [
    "mcg",
    "gvh",
    "alm",
    "mit",
    "erl",
    "pox",
    "vac",
    "nuc",
]

TARGET_COLUMN: str = "class_protein"

RAW_COLUMNS: list[str] = ["sequence_name", *FEATURE_COLUMNS, TARGET_COLUMN]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "mcg": "Método de McGeoch para reconhecimento de sequência sinal",
    "gvh": "Método de von Heijne para reconhecimento de sequência sinal",
    "alm": "Pontuação do programa ALOM de predição de região transmembranar",
    "mit": "Análise discriminante do conteúdo de aminoácidos N-terminal (mitocondrial)",
    "erl": "Presença do motivo 'HDEL' de retenção no lúmen do RE (binário: 0.5 ou 1.0)",
    "pox": "Sinal de direcionamento peroxissomal no C-terminal",
    "vac": "Análise discriminante de proteínas vacuolares e extracelulares",
    "nuc": "Análise discriminante de sinais de localização nuclear",
}

CLASS_DESCRIPTIONS: dict[str, str] = {
    "CYT": "Citosólica / Citoesquelética",
    "NUC": "Nuclear",
    "MIT": "Mitocondrial",
    "ME3": "Membrana (sem sinal N-terminal)",
    "ME2": "Membrana (sinal não clivado)",
    "ME1": "Membrana (sinal clivado)",
    "EXC": "Extracelular",
    "VAC": "Vacuolar",
    "POX": "Peroxissomal",
    "ERL": "Lúmen do Retículo Endoplasmático",
}


def get_project_root() -> Path:
    """Retorna o diretório raiz do repositório."""
    return Path(__file__).resolve().parent.parent


def load_yeast_dataframe(
    path: str | Path | None = None,
    drop_sequence_name: bool = True,
) -> pd.DataFrame:
    """Carrega a base Yeast a partir de arquivo local, arquivo ZIP ou caminho explícito.

    Funciona de forma transparente tanto localmente (repositório clonado ou notebook)
    quanto no Google Colab, sem exigir edição manual de caminhos.
    """
    repo_root = get_project_root()
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))

    candidates.extend(
        [
            repo_root / "data" / "yeast.data",
            Path("data/yeast.data"),
            Path("../data/yeast.data"),
            Path("yeast.data"),
            Path("/content/drive/MyDrive/Projeto_AM/Datasets/yeast.data"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix.lower() == ".zip":
                df = _read_from_zip(candidate)
            else:
                df = pd.read_csv(
                    candidate,
                    sep=r"\s+",
                    header=None,
                    names=RAW_COLUMNS,
                )
            break
    else:
        zip_candidates = [
            repo_root / "data" / "yeast-datasets.zip",
            Path("data/yeast-datasets.zip"),
            Path("../data/yeast-datasets.zip"),
        ]
        for zpath in zip_candidates:
            if zpath.is_file():
                df = _read_from_zip(zpath)
                break
        else:
            raise FileNotFoundError(
                "Não foi possível localizar 'yeast.data' nem 'yeast-datasets.zip' em 'data/'."
            )

    if drop_sequence_name and "sequence_name" in df.columns:
        df = df.drop(columns=["sequence_name"])

    return df


def _read_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extrai e lê 'yeast.data' diretamente de um arquivo ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        data_members = [m for m in zf.namelist() if m.endswith("yeast.data")]
        if not data_members:
            raise FileNotFoundError(
                f"Arquivo 'yeast.data' não encontrado dentro de {zip_path}."
            )
        raw_bytes = zf.read(data_members[0])
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Separa matriz de atributos preditivos X e vetor alvo codificado y."""
    X = df[FEATURE_COLUMNS].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    return X, y, label_encoder


def split_train_test(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Realiza divisão aleatória e estratificada (70% treino+validação / 30% teste)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
