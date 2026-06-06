"""Baseline TF-IDF + LogisticRegression/SVM com K-Fold e ablação 2×2."""

import logging
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5
NOMES_CLASSES = ["Irregular", "Regular com Ressalva", "Regular"]

TFIDF_PARAMS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
LOGREG_PARAMS = dict(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=RANDOM_STATE,
    class_weight="balanced",
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def construir_pipeline(modelo: str = "logreg") -> Pipeline:
    """Retorna Pipeline(TfidfVectorizer → classificador)."""
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    if modelo == "logreg":
        clf = LogisticRegression(**LOGREG_PARAMS)
    elif modelo == "svm":
        # LinearSVC não tem predict_proba nativo; use CalibratedClassifierCV para LIME/threshold
        clf = LinearSVC(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        warnings.warn(
            "LinearSVC não suporta predict_proba diretamente. "
            "Para LIME e otimização de threshold, envolva em CalibratedClassifierCV.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(f"modelo deve ser 'logreg' ou 'svm', recebido: {modelo!r}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# IC 95% via t-Student (sem scipy obrigatório)
# ---------------------------------------------------------------------------


def _ic95(scores: list[float]) -> tuple[float, float]:
    """Intervalo de confiança 95% via t-Student para n amostras pequenas."""
    try:
        from scipy.stats import t as t_dist

        n = len(scores)
        mean = np.mean(scores)
        se = np.std(scores, ddof=1) / sqrt(n)
        lower, upper = t_dist.interval(0.95, df=n - 1, loc=mean, scale=se)
        return float(lower), float(upper)
    except ImportError:
        # Fallback manual: t crítico para 4 graus de liberdade (n=5) ≈ 2.776
        t_crit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
        n = len(scores)
        mean = np.mean(scores)
        se = np.std(scores, ddof=1) / sqrt(n)
        t_crit = t_crit_table.get(n - 1, 2.0)
        return float(mean - t_crit * se), float(mean + t_crit * se)


# ---------------------------------------------------------------------------
# K-Fold
# ---------------------------------------------------------------------------


def treinar_baseline_kfold(
    X: pd.Series,
    y: pd.Series,
    modelo: str = "logreg",
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Treina com StratifiedKFold e retorna métricas consolidadas.

    Retorna dict com: fold_scores, mean_f1, std_f1, ci_95,
    per_class_f1 (média por classe), confusion_matrix (soma dos folds).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X_arr = X.values
    y_arr = y.values

    fold_f1: list[float] = []
    fold_acc: list[float] = []
    per_class_f1_acum: dict[str, list[float]] = {c: [] for c in NOMES_CLASSES}
    cm_total = np.zeros((len(NOMES_CLASSES), len(NOMES_CLASSES)), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        pipeline = construir_pipeline(modelo)
        pipeline.fit(X_arr[train_idx], y_arr[train_idx])
        y_pred = pipeline.predict(X_arr[val_idx])
        y_true = y_arr[val_idx]

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        fold_f1.append(f1)
        fold_acc.append(acc)

        per_class = f1_score(y_true, y_pred, average=None, labels=NOMES_CLASSES, zero_division=0)
        for i, cls in enumerate(NOMES_CLASSES):
            per_class_f1_acum[cls].append(float(per_class[i]))

        cm = confusion_matrix(y_true, y_pred, labels=NOMES_CLASSES)
        cm_total += cm

        logger.info("Fold %d/%d — F1-macro: %.4f | Acurácia: %.4f", fold_idx, n_splits, f1, acc)

    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1, ddof=1))
    ci_lower, ci_upper = _ic95(fold_f1)

    logger.info(
        "Resultado final — F1-macro: %.4f ± %.4f | IC95: [%.4f, %.4f]",
        mean_f1,
        std_f1,
        ci_lower,
        ci_upper,
    )

    return {
        "fold_scores": fold_f1,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_95": (ci_lower, ci_upper),
        "mean_acc": float(np.mean(fold_acc)),
        "per_class_f1": {cls: float(np.mean(v)) for cls, v in per_class_f1_acum.items()},
        "confusion_matrix": cm_total,
    }


# ---------------------------------------------------------------------------
# Ablação 2×2 (eixo TF-IDF)
# ---------------------------------------------------------------------------


def avaliar_ablacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa TF-IDF + LogReg em SUMARIO e VOTO.

    Requer colunas SUMARIO_TFIDF, VOTO_TFIDF e LABEL no DataFrame.
    """
    from preprocessamento.limpeza import aplicar_limpeza

    resultados = []
    for campo_orig, col_limpa in [("SUMARIO", "SUMARIO_TFIDF"), ("VOTO", "VOTO_TFIDF")]:
        if col_limpa not in df.columns:
            logger.info("Aplicando limpeza TF-IDF na coluna %s...", campo_orig)
            df[col_limpa] = aplicar_limpeza(df, campo=campo_orig, modo="tfidf")

        logger.info("=== Ablação: TF-IDF + LogReg | Campo: %s ===", campo_orig)
        res = treinar_baseline_kfold(df[col_limpa], df["LABEL"])

        resultados.append(
            {
                "Modelo": "TF-IDF + LogisticRegression",
                "Campo": campo_orig,
                "F1-macro": round(res["mean_f1"], 4),
                "F1-macro std": round(res["std_f1"], 4),
                "CI_95_lower": round(res["ci_95"][0], 4),
                "CI_95_upper": round(res["ci_95"][1], 4),
                "Acurácia": round(res["mean_acc"], 4),
                "F1_Irregular": round(res["per_class_f1"].get("Irregular", 0), 4),
                "F1_RegularRessalva": round(res["per_class_f1"].get("Regular com Ressalva", 0), 4),
                "F1_Regular": round(res["per_class_f1"].get("Regular", 0), 4),
            }
        )

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------


def plotar_resultados(resultados: dict, titulo: str = "", output_dir: Path | None = None) -> None:
    """Plota F1-macro por fold com média e IC95."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib não disponível — pulando plot")
        return

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "resultados" / "figuras"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = resultados["fold_scores"]
    mean_f1 = resultados["mean_f1"]
    ci_lower, ci_upper = resultados["ci_95"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(scores) + 1), scores, color="steelblue", alpha=0.8)
    ax.axhline(mean_f1, color="red", linestyle="--", label=f"Média={mean_f1:.4f}")
    ax.axhspan(ci_lower, ci_upper, alpha=0.15, color="red", label=f"IC95=[{ci_lower:.4f},{ci_upper:.4f}]")
    ax.set_xlabel("Fold")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"Baseline K-Fold {titulo}")
    ax.set_xticks(range(1, len(scores) + 1))
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()

    nome_arquivo = f"baseline_{titulo.replace(' ', '_').replace('/', '_')}.png" if titulo else "baseline.png"
    fig.savefig(output_dir / nome_arquivo, dpi=150)
    plt.close(fig)
    logger.info("Figura salva: %s", output_dir / nome_arquivo)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    caminho_train = data_processed / "train.parquet"
    if not caminho_train.exists():
        print("Execute primeiro filtrar_tematico.py e limpeza.py para gerar os splits.")
        sys.exit(1)

    train_df = pd.read_parquet(caminho_train)
    resultados_ablacao = avaliar_ablacao(train_df)
    print("\n=== Ablação TF-IDF ===")
    print(resultados_ablacao.to_string(index=False))
