"""Filtro temático, extração de label e persistência em parquet."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COLUNAS_USADAS = ["NUMACORDAO", "SITUACAO", "SUMARIO", "VOTO", "ACORDAO", "ASSUNTO"]

TERMOS_TEMATICOS = [
    "saúde",
    "sus",
    "fnde",
    "merenda",
    "educação",
    "ministério da saúde",
    "secretaria de saúde",
    "secretaria de educação",
]

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"


# ---------------------------------------------------------------------------
# Inspeção
# ---------------------------------------------------------------------------


def inspecionar_colunas(caminho_csv: str | Path) -> pd.DataFrame:
    """Lê apenas o cabeçalho do CSV e retorna DataFrame com nome e índice de cada coluna."""
    caminho_csv = Path(caminho_csv)
    df_header = pd.read_csv(
        caminho_csv,
        sep="|",
        encoding="utf-8-sig",
        nrows=0,
    )
    # Normaliza nomes (remove espaços e BOM residual)
    df_header.columns = df_header.columns.str.strip()

    info = pd.DataFrame(
        {"coluna": df_header.columns, "indice": range(len(df_header.columns))}
    )
    logger.info("CSV: %d colunas", len(info))
    logger.info("Colunas de interesse presentes: %s", [c for c in COLUNAS_USADAS if c in info["coluna"].values])
    ausentes = [c for c in COLUNAS_USADAS if c not in info["coluna"].values]
    if ausentes:
        logger.warning("Colunas não encontradas: %s", ausentes)
    return info


# ---------------------------------------------------------------------------
# Extração de label
# ---------------------------------------------------------------------------


def _extrair_label(situacao: str | None) -> str | None:
    """Mapeia o valor de SITUACAO para uma das 3 classes ou None.

    Formato antigo do CSV TCU: SITUACAO continha o desfecho diretamente
    ("IRREGULAR", "REGULAR COM RESSALVA", "REGULAR").
    No formato atual (2020+) SITUACAO = "OFICIALIZADO" (status de publicação).
    """
    if pd.isna(situacao) or situacao is None:
        return None
    s = str(situacao).lower().strip()
    if s in ("", "oficializado"):
        return None
    if "irregular" in s:
        return "Irregular"
    if "ressalva" in s:
        return "Regular com Ressalva"
    if s.startswith("regular"):
        return "Regular"
    return None


def _extrair_label_from_row(row: pd.Series) -> str | None:
    """Extrai o desfecho tentando SITUACAO, SUMARIO e ACORDAO (nessa ordem).

    No formato atual do TCU (2020+), SITUACAO = "OFICIALIZADO" para todos os
    registros. O desfecho real aparece como keyword estruturada no SUMARIO
    ("CONTAS IRREGULARES", "CONTAS REGULARES COM RESSALVA", "CONTAS REGULARES")
    ou no dispositivo do ACORDAO ("julgar irregulares...").
    """
    # 1. Formato antigo
    label = _extrair_label(row.get("SITUACAO", ""))
    if label is not None:
        return label

    # 2. SUMARIO — keywords estruturados no cabeçalho da ementa
    sumario = str(row.get("SUMARIO", "") or "").lower()
    if re.search(r"\bcontas?\s+irregulares?\b", sumario):
        return "Irregular"
    if re.search(r"\bcontas?\s+regulares?\s+com\s+ressalva\b", sumario):
        return "Regular com Ressalva"
    if re.search(r"\bcontas?\s+regulares?\b", sumario):
        return "Regular"

    # 3. ACORDAO — dispositivo da decisão
    acordao = str(row.get("ACORDAO", "") or "").lower()
    if re.search(r"\bjulgar?\s+irregulares?\b", acordao):
        return "Irregular"
    if re.search(r"\bjulgar?\s+regulares?\s+com\s+ressalva\b", acordao):
        return "Regular com Ressalva"
    if re.search(r"\bjulgar?\s+regulares?\b", acordao):
        return "Regular"

    return None


# ---------------------------------------------------------------------------
# Filtro temático
# ---------------------------------------------------------------------------


def _contem_termos(texto: str | None, termos: list[str] = TERMOS_TEMATICOS) -> bool:
    """Retorna True se texto contiver algum dos termos temáticos (case-insensitive)."""
    if texto is None or (isinstance(texto, float)):
        return False
    texto_lower = str(texto).lower()
    return any(t.lower() in texto_lower for t in termos)


def filtrar_por_tema(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas acórdãos de Saúde ou Educação (busca em SUMARIO e ASSUNTO)."""
    mascara = df["SUMARIO"].apply(_contem_termos) | df["ASSUNTO"].apply(_contem_termos)
    filtrado = df[mascara].copy()
    logger.info(
        "Filtro temático: %d → %d acórdãos (%.1f%%)",
        len(df),
        len(filtrado),
        100 * len(filtrado) / max(len(df), 1),
    )
    return filtrado


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------


def _carregar_ano(ano: int, data_dir: Path) -> pd.DataFrame | None:
    """Carrega CSV de um ano com apenas as colunas necessárias."""
    caminho = data_dir / f"acordao-completo-{ano}.csv"
    if not caminho.exists():
        logger.warning("Arquivo não encontrado: %s", caminho)
        return None

    df = pd.read_csv(
        caminho,
        sep="|",
        encoding="utf-8-sig",
        usecols=COLUNAS_USADAS,
        dtype=str,
        low_memory=False,
    )
    # Normaliza nomes de colunas (remove espaços e BOM residual)
    df.columns = df.columns.str.strip()
    df["ANO"] = ano
    logger.info("Ano %d: %d acórdãos carregados", ano, len(df))
    return df


def combinar_anos(
    anos: list[int],
    data_dir: Path = DATA_RAW,
) -> pd.DataFrame:
    """Carrega, concatena, extrai labels e filtra por tema. Retorna DataFrame limpo."""
    dfs = []
    for ano in anos:
        df_ano = _carregar_ano(ano, data_dir)
        if df_ano is not None:
            dfs.append(df_ano)

    if not dfs:
        raise RuntimeError(f"Nenhum CSV encontrado em {data_dir} para os anos {anos}")

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total carregado: %d acórdãos (%d anos)", len(df), len(dfs))

    # Extrai label (usa SITUACAO → SUMARIO → ACORDAO, nessa ordem)
    df["LABEL"] = df.apply(_extrair_label_from_row, axis=1)
    n_sem_label = df["LABEL"].isna().sum()
    df = df.dropna(subset=["LABEL"]).copy()
    logger.info("Após extração de label: %d acórdãos (%d descartados sem label)", len(df), n_sem_label)

    # Filtro temático
    df = filtrar_por_tema(df)

    logger.info("Distribuição de classes:\n%s", df["LABEL"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------


def salvar_parquet(df: pd.DataFrame, destino: Path = DATA_INTERIM / "acordaos_filtrados.parquet") -> None:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False, engine="pyarrow")
    size_mb = destino.stat().st_size / 1_000_000
    logger.info("Salvo: %s (%.2f MB, %d linhas)", destino, size_mb, len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    anos = list(range(2020, 2025))
    df = combinar_anos(anos)
    salvar_parquet(df)
    print(f"\nDataset filtrado: {len(df)} acórdãos")
    print(df["LABEL"].value_counts())
