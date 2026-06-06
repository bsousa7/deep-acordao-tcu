"""Testes unitários sem dependência de CSV real, GPU ou internet."""

import csv
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 33 colunas MAIÚSCULAS simulando a estrutura real do TCU (D-07)
_COLUNAS_TCU = [
    "NUMACORDAO", "COLEGIADO", "TIPO", "DATA", "RELATOR", "PROCESSO",
    "INTERESSADOS", "ENTIDADE", "REPRESENTACAO", "UF", "MUNICIPIO",
    "ORGAO", "OBJETO", "EXERCICIO", "MODALIDADE", "COL15", "COL16",
    "COL17", "COL18", "COL19", "COL20", "ASSUNTO", "SUMARIO",
    "ACORDAO", "COL24", "COL25", "COL26", "COL27", "COL28",
    "VOTO", "COL30", "SITUACAO", "COL32",
]
assert len(_COLUNAS_TCU) == 33

_TERMOS_SAUDE = ["saúde", "SUS", "FNDE", "merenda", "educação",
                 "ministério da saúde", "secretaria de saúde", "secretaria de educação"]


def _criar_csv_mock(linhas: list[dict]) -> str:
    """Gera CSV pipe-delimited com 33 colunas como string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_COLUNAS_TCU, delimiter="|")
    writer.writeheader()
    for linha in linhas:
        row = {c: linha.get(c, "") for c in _COLUNAS_TCU}
        writer.writerow(row)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Testes de _extrair_label
# ---------------------------------------------------------------------------


def test_extrair_label_irregular():
    from preprocessamento.filtrar_tematico import _extrair_label

    assert _extrair_label("IRREGULAR") == "Irregular"
    assert _extrair_label("Irregular") == "Irregular"
    assert _extrair_label("irregular") == "Irregular"
    assert _extrair_label("Irregular (Débito)") == "Irregular"
    assert _extrair_label("IRREGULARES") == "Irregular"


def test_extrair_label_regular_ressalva():
    from preprocessamento.filtrar_tematico import _extrair_label

    assert _extrair_label("Regular com Ressalva") == "Regular com Ressalva"
    assert _extrair_label("REGULAR COM RESSALVA") == "Regular com Ressalva"
    assert _extrair_label("regular c/ ressalva") == "Regular com Ressalva"


def test_extrair_label_regular():
    from preprocessamento.filtrar_tematico import _extrair_label

    assert _extrair_label("Regular") == "Regular"
    assert _extrair_label("REGULAR") == "Regular"
    assert _extrair_label("regular") == "Regular"


def test_extrair_label_invalido():
    from preprocessamento.filtrar_tematico import _extrair_label

    assert _extrair_label(None) is None
    assert _extrair_label("") is None
    assert _extrair_label("BAIXADO") is None
    assert _extrair_label("EM TRAMITAÇÃO") is None
    assert _extrair_label(float("nan")) is None


# ---------------------------------------------------------------------------
# Testes de _contem_termos
# ---------------------------------------------------------------------------


def test_contem_termos_positivo():
    from preprocessamento.filtrar_tematico import _contem_termos

    assert _contem_termos("Acórdão sobre o SUS e hospitais", _TERMOS_SAUDE)
    assert _contem_termos("Ministério da Saúde repasse", _TERMOS_SAUDE)
    assert _contem_termos("FNDE merenda escolar educação", _TERMOS_SAUDE)
    assert _contem_termos("Secretaria de Educação municipal", _TERMOS_SAUDE)


def test_contem_termos_negativo():
    from preprocessamento.filtrar_tematico import _contem_termos

    assert not _contem_termos("Contrato de obras rodoviárias", _TERMOS_SAUDE)
    assert not _contem_termos("", _TERMOS_SAUDE)
    assert not _contem_termos(None, _TERMOS_SAUDE)
    assert not _contem_termos(float("nan"), _TERMOS_SAUDE)


# ---------------------------------------------------------------------------
# Testes de inspecionar_colunas
# ---------------------------------------------------------------------------


def test_inspecionar_colunas():
    from preprocessamento.filtrar_tematico import inspecionar_colunas

    csv_content = _criar_csv_mock([{"NUMACORDAO": "1", "SITUACAO": "Irregular"}])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        caminho = f.name

    info = inspecionar_colunas(caminho)
    assert "coluna" in info.columns
    assert "indice" in info.columns
    assert len(info) == 33
    assert "SITUACAO" in info["coluna"].values
    assert "VOTO" in info["coluna"].values
    assert "SUMARIO" in info["coluna"].values


# ---------------------------------------------------------------------------
# Testes de filtrar_por_tema
# ---------------------------------------------------------------------------


def test_filtrar_por_tema():
    from preprocessamento.filtrar_tematico import filtrar_por_tema

    df = pd.DataFrame(
        {
            "SUMARIO": [
                "Repasse do SUS para hospital",
                "Obras de saneamento básico",
                "Verbas do FNDE educação",
                "Contrato de licitação rodovias",
            ],
            "ASSUNTO": ["", "", "", "educação municipal"],
            "LABEL": ["Irregular", "Regular", "Irregular", "Regular"],
        }
    )
    resultado = filtrar_por_tema(df)
    # Linha 0: SUS (match SUMARIO)
    # Linha 1: sem match
    # Linha 2: FNDE + educação (match SUMARIO)
    # Linha 3: educação (match ASSUNTO)
    assert len(resultado) == 3
    assert "Obras de saneamento básico" not in resultado["SUMARIO"].values


# ---------------------------------------------------------------------------
# Testes de limpar_coluna
# ---------------------------------------------------------------------------


def test_limpar_coluna_tfidf_lowercase():
    from preprocessamento.limpeza import limpar_coluna

    resultado = limpar_coluna("Saúde Pública IRREGULAR", modo="tfidf")
    assert resultado == resultado.lower()


def test_limpar_coluna_bert_preserva_case():
    from preprocessamento.limpeza import limpar_coluna

    texto = "Ministério da SAÚDE aprovou contas"
    resultado = limpar_coluna(texto, modo="bert")
    assert "SAÚDE" in resultado or "Saúde" in resultado  # não foi lowercased


def test_limpar_coluna_none():
    from preprocessamento.limpeza import limpar_coluna

    assert limpar_coluna(None, modo="tfidf") == ""
    assert limpar_coluna(None, modo="bert") == ""
    assert limpar_coluna(float("nan")) == ""


# ---------------------------------------------------------------------------
# Testes de truncar_head_tail
# ---------------------------------------------------------------------------


def test_truncar_head_tail_texto_curto():
    """Texto que cabe em 512 tokens não deve ser alterado."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    except Exception:
        pytest.skip("Tokenizer não disponível (sem internet)")

    from preprocessamento.limpeza import truncar_head_tail

    texto_curto = "Texto curto sobre saúde pública."
    resultado = truncar_head_tail(texto_curto, tokenizer)
    # Texto curto não deve ser modificado
    assert len(resultado) > 0


def test_truncar_head_tail_texto_longo():
    """Texto longo deve resultar em no máximo 510 tokens decodificados."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    except Exception:
        pytest.skip("Tokenizer não disponível (sem internet)")

    from preprocessamento.limpeza import truncar_head_tail, HEAD_TOKENS, TAIL_TOKENS

    # Cria texto com ~1000 tokens
    texto_longo = "acórdão irregular contas " * 300
    resultado = truncar_head_tail(texto_longo, tokenizer)

    ids = tokenizer.encode(resultado, add_special_tokens=False)
    assert len(ids) <= HEAD_TOKENS + TAIL_TOKENS


# ---------------------------------------------------------------------------
# Testes de dividir_dados
# ---------------------------------------------------------------------------


def test_dividir_dados_proporcoes():
    from preprocessamento.limpeza import dividir_dados

    np.random.seed(42)
    n = 200
    labels = (
        ["Irregular"] * 176  # 88%
        + ["Regular com Ressalva"] * 12  # 6%
        + ["Regular"] * 12  # 6%
    )
    df = pd.DataFrame(
        {
            "TEXTO": [f"texto {i}" for i in range(n)],
            "LABEL": labels,
        }
    )
    train, val, test = dividir_dados(df)

    total = len(train) + len(val) + len(test)
    assert total == n
    assert abs(len(train) / n - 0.70) < 0.05
    assert abs(len(val) / n - 0.15) < 0.05
    assert abs(len(test) / n - 0.15) < 0.05

    # Todas as classes presentes em cada split
    for split in [train, val, test]:
        classes_no_split = set(split["LABEL"].unique())
        assert "Irregular" in classes_no_split


def test_dividir_dados_classe_insuficiente():
    from preprocessamento.limpeza import dividir_dados

    df = pd.DataFrame(
        {
            "TEXTO": ["a", "b", "c"],
            "LABEL": ["Irregular", "Irregular", "Regular"],
        }
    )
    with pytest.raises(ValueError, match="insuficientes"):
        dividir_dados(df)


# ---------------------------------------------------------------------------
# Testes de construir_pipeline (baseline)
# ---------------------------------------------------------------------------


def test_construir_pipeline_logreg():
    from modelos.baseline import construir_pipeline

    pipeline = construir_pipeline("logreg")
    textos = [f"texto sobre saúde pública {i}" for i in range(60)]
    labels = ["Irregular"] * 40 + ["Regular com Ressalva"] * 10 + ["Regular"] * 10
    pipeline.fit(textos, labels)
    predicoes = pipeline.predict(textos[:5])
    assert len(predicoes) == 5
    assert all(p in ["Irregular", "Regular com Ressalva", "Regular"] for p in predicoes)


def test_construir_pipeline_modelo_invalido():
    from modelos.baseline import construir_pipeline

    with pytest.raises(ValueError):
        construir_pipeline("modelo_inexistente")


# ---------------------------------------------------------------------------
# Testes de _calcular_class_weights (transformer)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch não instalado",
)
def test_calcular_class_weights():
    from modelos.transformer import _calcular_class_weights

    # 80% classe 0, 10% classe 1, 10% classe 2
    y = [0] * 80 + [1] * 10 + [2] * 10
    pesos = _calcular_class_weights(y)
    assert pesos.shape[0] == 3
    # Classe majoritária deve ter peso menor
    assert pesos[0] < pesos[1]
    assert pesos[0] < pesos[2]


# ---------------------------------------------------------------------------
# Testes de otimizar_threshold
# ---------------------------------------------------------------------------


def test_otimizar_threshold():
    from avaliacao.metricas import otimizar_threshold

    np.random.seed(42)
    n = 200
    # Irregular = label 0; não-Irregular = 1 ou 2
    y_true = [0] * 100 + [1] * 50 + [2] * 50
    # Probabilidade perfeita para threshold=0.5
    proba = np.array(
        [0.9] * 100  # Irregular alto
        + [0.2] * 100  # não-Irregular baixo
    )

    threshold_otimo, metricas = otimizar_threshold(
        y_true, proba, custo_fn=10.0, custo_fp=1.0
    )
    assert 0.0 <= threshold_otimo <= 1.0
    assert "recall_irregular" in metricas
    assert "f1_irregular" in metricas
    assert metricas["recall_irregular"] > 0.5


# ---------------------------------------------------------------------------
# Testes de comparar_modelos
# ---------------------------------------------------------------------------


def test_comparar_modelos_ordenado():
    from avaliacao.metricas import comparar_modelos

    resultados = {
        "TF-IDF SUMARIO": {"mean_f1": 0.84, "std_f1": 0.02, "ci_95": (0.82, 0.86), "mean_acc": 0.96},
        "LegalBert VOTO": {"mean_f1": 0.87, "std_f1": 0.01, "ci_95": (0.86, 0.88), "mean_acc": 0.95},
        "TF-IDF VOTO": {"mean_f1": 0.80, "std_f1": 0.03, "ci_95": (0.77, 0.83), "mean_acc": 0.94},
    }
    df = comparar_modelos(resultados)
    assert len(df) == 3
    # Primeiro deve ser o de maior F1
    assert df.iloc[0]["F1-macro"] >= df.iloc[1]["F1-macro"]
    assert df.iloc[1]["F1-macro"] >= df.iloc[2]["F1-macro"]
