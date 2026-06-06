"""Limpeza de texto e divisão estratificada de dados."""

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
HEAD_TOKENS = 128
TAIL_TOKENS = 382  # 128 + 382 = 510 + 2 especiais = 512
MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"

DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

_STOPWORDS_PT: set[str] = set()


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------


def _carregar_stopwords() -> set[str]:
    global _STOPWORDS_PT
    if _STOPWORDS_PT:
        return _STOPWORDS_PT
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        _STOPWORDS_PT = set(stopwords.words("portuguese"))
    except Exception as exc:
        logger.warning("NLTK stopwords não disponíveis: %s", exc)
        _STOPWORDS_PT = set()
    return _STOPWORDS_PT


# ---------------------------------------------------------------------------
# Limpeza de texto
# ---------------------------------------------------------------------------


def limpar_coluna(texto: str | None, modo: str = "tfidf") -> str:
    """
    Limpa texto para TF-IDF (lowercase + stopwords) ou BERT (preserva case).

    modo='tfidf': lowercase, remove pontuação e stopwords, colapsa espaços.
    modo='bert':  normaliza unicode, remove excesso de quebras, preserva case.
    """
    if texto is None or (isinstance(texto, float)):
        return ""

    texto = str(texto).strip()

    # Normalização unicode comum a ambos os modos
    texto = unicodedata.normalize("NFKC", texto)

    # Remove caracteres de controle (mantém \n para BERT)
    if modo == "tfidf":
        texto = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", texto)
        texto = texto.lower()
        # Remove pontuação e dígitos isolados
        texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
        # Remove stopwords
        sw = _carregar_stopwords()
        if sw:
            tokens = texto.split()
            tokens = [t for t in tokens if t not in sw]
            texto = " ".join(tokens)
        # Colapsa espaços múltiplos
        texto = re.sub(r"\s+", " ", texto).strip()
    else:  # bert
        texto = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", texto)
        # Colapsa mais de 2 quebras de linha consecutivas
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        # Colapsa espaços múltiplos (mantém \n)
        texto = re.sub(r"[ \t]+", " ", texto).strip()

    return texto


# ---------------------------------------------------------------------------
# Head+tail para BERT (D-01)
# ---------------------------------------------------------------------------


def truncar_head_tail(texto: str, tokenizer) -> str:
    """
    Trunca texto para 510 tokens usando estratégia head+tail (D-01).

    Retorna string decodificada pronta para re-tokenização no treino.
    """
    ids = tokenizer.encode(texto, add_special_tokens=False)
    budget = HEAD_TOKENS + TAIL_TOKENS  # 510
    if len(ids) <= budget:
        return texto
    head = ids[:HEAD_TOKENS]
    tail = ids[-TAIL_TOKENS:]
    return tokenizer.decode(head + tail, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Aplicação em DataFrame
# ---------------------------------------------------------------------------


def aplicar_limpeza(
    df: pd.DataFrame,
    campo: str = "SUMARIO",
    modo: str = "tfidf",
) -> pd.Series:
    """Retorna Series com texto limpo. Para BERT aplica também head+tail."""
    try:
        from tqdm import tqdm as tqdm_cls

        tqdm_cls.pandas(desc=f"Limpeza {modo}/{campo}")
        serie = df[campo].progress_apply(lambda t: limpar_coluna(t, modo))
    except ImportError:
        serie = df[campo].apply(lambda t: limpar_coluna(t, modo))

    if modo == "bert":
        from transformers import AutoTokenizer

        logger.info("Carregando tokenizer %s para head+tail...", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        try:
            from tqdm import tqdm as tqdm_cls

            tqdm_cls.pandas(desc=f"Head+tail {campo}")
            serie = serie.progress_apply(lambda t: truncar_head_tail(t, tokenizer))
        except ImportError:
            serie = serie.apply(lambda t: truncar_head_tail(t, tokenizer))

    return serie


# ---------------------------------------------------------------------------
# Split estratificado
# ---------------------------------------------------------------------------


def dividir_dados(
    df: pd.DataFrame,
    col_label: str = "LABEL",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split estratificado 70 / 15 / 15 com seed=42.

    Raises ValueError se alguma classe tiver amostras insuficientes para o split.
    """
    contagens = df[col_label].value_counts()
    n_splits_minimo = 3  # precisa de ao menos 3 por classe para ter ≥1 em cada split
    classes_pequenas = contagens[contagens < n_splits_minimo]
    if not classes_pequenas.empty:
        raise ValueError(
            f"Classes com amostras insuficientes para split estratificado "
            f"(mínimo {n_splits_minimo}): {classes_pequenas.to_dict()}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[col_label],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[col_label],
        random_state=RANDOM_STATE,
    )

    for nome, split in [("treino", train_df), ("val", val_df), ("teste", test_df)]:
        logger.info(
            "Split %s: %d amostras | %s",
            nome,
            len(split),
            split[col_label].value_counts().to_dict(),
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def salvar_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path = DATA_PROCESSED,
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for nome, split in [("train", train), ("val", val), ("test", test)]:
        destino = data_dir / f"{nome}.parquet"
        split.to_parquet(destino, index=False, engine="pyarrow")
        logger.info("Salvo: %s (%d linhas)", destino, len(split))
