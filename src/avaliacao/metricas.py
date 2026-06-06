"""Métricas, comparação de modelos, threshold ótimo e LIME."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

NOMES_CLASSES = ["Irregular", "Regular com Ressalva", "Regular"]
RESULTADOS_DIR = Path(__file__).resolve().parents[2] / "resultados"


# ---------------------------------------------------------------------------
# Relatório de classificação
# ---------------------------------------------------------------------------


def gerar_relatorio_classificacao(
    y_true: list,
    y_pred: list,
    nomes_classes: list[str] = NOMES_CLASSES,
    titulo: str = "",
) -> dict:
    """Retorna dict com métricas completas; alerta se F1=0 em alguma classe."""
    relatorio = classification_report(
        y_true,
        y_pred,
        target_names=nomes_classes,
        output_dict=True,
        zero_division=0,
    )
    if titulo:
        logger.info("=== %s ===", titulo)
    logger.info(
        classification_report(y_true, y_pred, target_names=nomes_classes, zero_division=0)
    )

    # Alerta de colapso de classe
    for cls in nomes_classes:
        if cls in relatorio and relatorio[cls].get("f1-score", 1) == 0:
            warnings.warn(
                f"F1=0 para classe '{cls}' — possível colapso do modelo. "
                "Verificar class_weight e distribuição do conjunto de treino.",
                RuntimeWarning,
                stacklevel=2,
            )
    return relatorio


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------


def plotar_matriz_confusao(
    y_true: list,
    y_pred: list,
    nomes_classes: list[str] = NOMES_CLASSES,
    titulo: str = "",
    output_dir: Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn não disponíveis — pulando matriz de confusão")
        return

    if output_dir is None:
        output_dir = RESULTADOS_DIR / "figuras"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=nomes_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title_suffix in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Absoluta", "Normalizada (recall)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=nomes_classes,
            yticklabels=nomes_classes,
            ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {title_suffix}" + (f" ({titulo})" if titulo else ""))

    fig.tight_layout()
    nome = f"cm_{titulo.replace(' ', '_').replace('/', '_')}.png" if titulo else "cm.png"
    fig.savefig(output_dir / nome, dpi=150)
    plt.close(fig)
    logger.info("Matriz de confusão salva: %s", output_dir / nome)


# ---------------------------------------------------------------------------
# Comparação de modelos
# ---------------------------------------------------------------------------


def comparar_modelos(resultados_dict: dict[str, dict]) -> pd.DataFrame:
    """
    Recebe dict {nome_modelo: {mean_f1, std_f1, ci_95, mean_acc}} e
    retorna DataFrame ordenado por F1-macro decrescente.
    """
    linhas = []
    for nome, res in resultados_dict.items():
        ci = res.get("ci_95", (None, None))
        linhas.append(
            {
                "Modelo/Campo": nome,
                "F1-macro": round(res.get("mean_f1", 0), 4),
                "± std": round(res.get("std_f1", 0), 4),
                "IC95 [low, high]": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "—",
                "Acurácia": round(res.get("mean_acc", 0), 4),
            }
        )

    df = pd.DataFrame(linhas).sort_values("F1-macro", ascending=False).reset_index(drop=True)
    logger.info("\n=== Comparação de Modelos ===\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Otimização de threshold (custo assimétrico)
# ---------------------------------------------------------------------------


def otimizar_threshold(
    y_true: list[int],
    proba_irregular: np.ndarray,
    custo_fn: float = 10.0,
    custo_fp: float = 1.0,
    output_dir: Path | None = None,
) -> tuple[float, dict]:
    """
    Encontra threshold ótimo minimizando custo(FN × custo_fn + FP × custo_fp).

    y_true: 0=Irregular, 1/2=não-Irregular (binário para esta análise).
    proba_irregular: probabilidade predita da classe Irregular.
    Retorna (threshold_ótimo, dict_métricas).
    """
    y_true_bin = np.array([1 if v == 0 else 0 for v in y_true])  # 1=Irregular
    thresholds = np.linspace(0.05, 0.95, 181)
    custos = []

    for thr in thresholds:
        y_pred_bin = (proba_irregular >= thr).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        # cm[1,0] = FN (Irregular predito como não-Irregular)
        # cm[0,1] = FP (não-Irregular predito como Irregular)
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        custos.append(custo_fn * fn + custo_fp * fp)

    custos_arr = np.array(custos)
    idx_otimo = int(np.argmin(custos_arr))
    threshold_otimo = float(thresholds[idx_otimo])

    y_pred_otimo = (proba_irregular >= threshold_otimo).astype(int)
    metricas = {
        "threshold_otimo": threshold_otimo,
        "custo_total": float(custos_arr[idx_otimo]),
        "f1_irregular": float(f1_score(y_true_bin, y_pred_otimo, pos_label=1, zero_division=0)),
        "recall_irregular": float(recall_score(y_true_bin, y_pred_otimo, pos_label=1, zero_division=0)),
        "precision_irregular": float(precision_score(y_true_bin, y_pred_otimo, pos_label=1, zero_division=0)),
    }
    logger.info(
        "Threshold ótimo: %.2f | Custo: %.1f | F1-Irregular: %.4f | Recall: %.4f",
        threshold_otimo,
        metricas["custo_total"],
        metricas["f1_irregular"],
        metricas["recall_irregular"],
    )

    # Plot curva de custo
    _plotar_curva_threshold(thresholds, custos_arr, threshold_otimo, output_dir)

    return threshold_otimo, metricas


def _plotar_curva_threshold(
    thresholds: np.ndarray,
    custos: np.ndarray,
    threshold_otimo: float,
    output_dir: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if output_dir is None:
        output_dir = RESULTADOS_DIR / "figuras"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, custos, color="steelblue")
    ax.axvline(threshold_otimo, color="red", linestyle="--", label=f"Ótimo={threshold_otimo:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(f"Custo (FN×10 + FP×1)")
    ax.set_title("Curva de Custo por Threshold — Detecção de Irregular")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "threshold_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def explicar_com_lime(
    modelo,
    textos: list[str],
    labels_verdadeiros: list[int],
    tokenizer=None,
    nomes_classes: list[str] = NOMES_CLASSES,
    n_samples: int = 5,
    output_dir: Path | None = None,
) -> None:
    """
    Gera explicações LIME para os primeiros n_samples textos.

    Para modelos sklearn (baseline): modelo deve ter predict_proba.
    Para Transformer: passar tokenizer; predictor faz inferência em batch.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("lime não disponível — pulando LIME")
        return

    if output_dir is None:
        output_dir = RESULTADOS_DIR / "figuras"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    explainer = LimeTextExplainer(class_names=nomes_classes)

    if tokenizer is not None:
        predictor = _construir_predictor_transformer(modelo, tokenizer)
    else:
        predictor = modelo.predict_proba

    for i, (texto, label) in enumerate(zip(textos[:n_samples], labels_verdadeiros[:n_samples])):
        try:
            exp = explainer.explain_instance(
                texto,
                predictor,
                num_features=10,
                labels=list(range(len(nomes_classes))),
            )
            caminho_html = Path(output_dir) / f"lime_{i}.html"
            exp.save_to_file(str(caminho_html))
            logger.info(
                "LIME amostra %d/%d (label=%s) → %s",
                i + 1,
                n_samples,
                nomes_classes[label] if label < len(nomes_classes) else label,
                caminho_html,
            )
        except Exception as exc:
            logger.warning("LIME falhou para amostra %d: %s", i, exc)


def _construir_predictor_transformer(modelo, tokenizer, batch_size: int = 32):
    """Retorna função predictor compatível com LIME para modelos HuggingFace."""
    def predictor(textos: list[str]) -> np.ndarray:
        import torch
        modelo.eval()
        all_probs = []
        for start in range(0, len(textos), batch_size):
            batch = textos[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            device = next(modelo.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = modelo(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    return predictor


# ---------------------------------------------------------------------------
# Serialização de resultados
# ---------------------------------------------------------------------------


def salvar_metricas_json(metricas: dict, caminho: Path = RESULTADOS_DIR / "metricas.json") -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    def _converter(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Tipo não serializável: {type(obj)}")

    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(metricas, fh, indent=2, ensure_ascii=False, default=_converter)
    logger.info("Métricas salvas: %s", caminho)
