"""Fine-tuning LegalBert-pt com LoRA, class weights, early stopping e K-Fold."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"
MODEL_FALLBACK = "neuralmind/bert-base-portuguese-cased"

RANDOM_STATE = 42
N_SPLITS = 5
NOMES_CLASSES = ["Irregular", "Regular com Ressalva", "Regular"]
LABEL2ID = {c: i for i, c in enumerate(NOMES_CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(NOMES_CLASSES)}

TRAIN_PARAMS = dict(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.10,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    seed=RANDOM_STATE,
    fp16=True,
    report_to="none",
)

LORA_CONFIG_PARAMS = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------


def _carregar_modelo(num_labels: int = 3) -> tuple:
    """Carrega tokenizer e modelo LegalBert-pt. Fallback para BERTimbau."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for nome in [MODEL_NAME, MODEL_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(nome)
            model = AutoModelForSequenceClassification.from_pretrained(
                nome,
                num_labels=num_labels,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            logger.info("Modelo carregado: %s", nome)
            return tokenizer, model
        except Exception as exc:
            logger.warning("Falha ao carregar %s: %s. Tentando fallback...", nome, exc)
    raise RuntimeError("Nenhum modelo base disponível.")


def _aplicar_lora(model):
    """Aplica LoRA ao modelo para K-Fold eficiente (~300K parâmetros treináveis)."""
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        **LORA_CONFIG_PARAMS,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def _calcular_class_weights(y: list[int]):
    """Calcula pesos balanceados inversamente proporcionais à frequência de classe."""
    import torch

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(y))
    # Garante ordem correta para todas as 3 classes
    weight_tensor = torch.ones(len(NOMES_CLASSES))
    for cls_idx, weight in zip(classes, weights):
        weight_tensor[cls_idx] = float(weight)
    return weight_tensor


# ---------------------------------------------------------------------------
# Dataset HuggingFace
# ---------------------------------------------------------------------------


def _criar_dataset(textos: list[str], labels: list[int], tokenizer):
    """Tokeniza e cria datasets.Dataset com coluna 'labels'."""
    from datasets import Dataset

    encoding = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoding["labels"] = labels
    return Dataset.from_dict(encoding)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def _construir_compute_metrics():
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1_macro": float(f1_score(label_ids, preds, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(label_ids, preds)),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# WeightedTrainer (cross-entropy com pesos de classe)
# ---------------------------------------------------------------------------


def _construir_weighted_loss_trainer(class_weights):
    """Subclasse dinâmica de Trainer com cross-entropy ponderada (D-03)."""
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights.to(logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ---------------------------------------------------------------------------
# Treino completo (hold-out)
# ---------------------------------------------------------------------------


def treinar_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    campo: str = "VOTO",
    output_dir: str | Path = "resultados/modelos/transformer",
    usar_lora: bool = False,
) -> dict:
    """
    Fine-tuning LegalBert-pt com class weights e early stopping.

    Retorna dict com métricas do conjunto de validação.
    """
    from transformers import EarlyStoppingCallback, TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_textos = train_df[campo].fillna("").tolist()
    val_textos = val_df[campo].fillna("").tolist()
    train_labels = [LABEL2ID[l] for l in train_df["LABEL"]]
    val_labels = [LABEL2ID[l] for l in val_df["LABEL"]]

    tokenizer, model = _carregar_modelo()
    if usar_lora:
        model = _aplicar_lora(model)

    class_weights = _calcular_class_weights(train_labels)
    train_dataset = _criar_dataset(train_textos, train_labels, tokenizer)
    val_dataset = _criar_dataset(val_textos, val_labels, tokenizer)

    args = TrainingArguments(output_dir=str(output_dir), **TRAIN_PARAMS)
    WeightedTrainer = _construir_weighted_loss_trainer(class_weights)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_construir_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metricas = trainer.evaluate()
    logger.info("Val F1-macro: %.4f | Acurácia: %.4f", metricas.get("eval_f1_macro", 0), metricas.get("eval_accuracy", 0))
    return metricas


# ---------------------------------------------------------------------------
# K-Fold com LoRA
# ---------------------------------------------------------------------------


def kfold_com_lora(
    df: pd.DataFrame,
    campo: str = "VOTO",
    n_splits: int = N_SPLITS,
    output_base: str | Path = "resultados/modelos",
) -> dict:
    """
    LoRA K-Fold CV com isolamento de checkpoint por fold.

    Libera VRAM explicitamente entre folds para evitar OOM no T4.
    """
    from math import sqrt

    labels_arr = df["LABEL"].values
    textos_arr = df[campo].fillna("").values

    # Valida amostras mínimas por classe
    from collections import Counter
    contagens = Counter(labels_arr)
    for cls, cnt in contagens.items():
        if cnt < n_splits:
            raise ValueError(
                f"Classe '{cls}' tem apenas {cnt} amostras — insuficiente para {n_splits} folds. "
                "Reduza n_splits ou expanda o corpus."
            )

    label_ids_arr = np.array([LABEL2ID[l] for l in labels_arr])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_f1: list[float] = []
    fold_acc: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(textos_arr, label_ids_arr), start=1):
        import torch
        from transformers import EarlyStoppingCallback, TrainingArguments

        fold_output = Path(output_base) / f"lora_fold_{fold_idx}"
        fold_output.mkdir(parents=True, exist_ok=True)

        tokenizer, model = _carregar_modelo()
        model = _aplicar_lora(model)

        train_textos = textos_arr[train_idx].tolist()
        val_textos = textos_arr[val_idx].tolist()
        train_labels = label_ids_arr[train_idx].tolist()
        val_labels = label_ids_arr[val_idx].tolist()

        class_weights = _calcular_class_weights(train_labels)
        train_dataset = _criar_dataset(train_textos, train_labels, tokenizer)
        val_dataset = _criar_dataset(val_textos, val_labels, tokenizer)

        params = {**TRAIN_PARAMS, "output_dir": str(fold_output)}
        args = TrainingArguments(**params)
        WeightedTrainer = _construir_weighted_loss_trainer(class_weights)

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_construir_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        metricas_fold = trainer.evaluate()
        f1 = metricas_fold.get("eval_f1_macro", 0.0)
        acc = metricas_fold.get("eval_accuracy", 0.0)
        fold_f1.append(float(f1))
        fold_acc.append(float(acc))
        logger.info("Fold %d/%d — F1-macro: %.4f | Acurácia: %.4f", fold_idx, n_splits, f1, acc)

        # Libera VRAM
        del model, trainer
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass

    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1, ddof=1))
    se = std_f1 / sqrt(n_splits)

    try:
        from scipy.stats import t as t_dist
        ci_lower, ci_upper = t_dist.interval(0.95, df=n_splits - 1, loc=mean_f1, scale=se)
    except ImportError:
        t_crit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
        t_crit = t_crit_table.get(n_splits - 1, 2.0)
        ci_lower = mean_f1 - t_crit * se
        ci_upper = mean_f1 + t_crit * se

    logger.info(
        "LoRA K-Fold %s — F1-macro: %.4f ± %.4f | IC95: [%.4f, %.4f]",
        campo,
        mean_f1,
        std_f1,
        ci_lower,
        ci_upper,
    )

    return {
        "campo": campo,
        "fold_scores": fold_f1,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_95": (float(ci_lower), float(ci_upper)),
        "mean_acc": float(np.mean(fold_acc)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    if not (data_processed / "train.parquet").exists():
        print("Execute o pipeline de pré-processamento antes do treino do Transformer.")
        sys.exit(1)

    df_train = pd.read_parquet(data_processed / "train.parquet")
    df_val = pd.read_parquet(data_processed / "val.parquet")

    # Fine-tuning no campo VOTO (principal — D-06)
    metricas = treinar_transformer(df_train, df_val, campo="VOTO")
    print("\nMétricas val:", metricas)
