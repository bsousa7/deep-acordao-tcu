# deep-acordao-tcu — Guia para Claude Code

## O Projeto

Jurimetria preditiva: classificar acórdãos do TCU (Saúde e Educação) em 3 classes de desfecho —
`Irregular`, `Regular com Ressalva`, `Regular` — usando TF-IDF baseline e LegalBert-pt.

## Guardrails (Regras Inegociáveis)

1. **Sem scraping** — apenas CSVs oficiais do Portal de Dados Abertos do TCU.
2. **`usecols=[...]` obrigatório** em todo `pd.read_csv` — CSVs pesam 175–445 MB.
3. **Filtro temático imediato** após carregar qualquer CSV.
4. **`RANDOM_STATE = 42`** em todo split, treino e inicialização de modelos.
5. **Não inventar referências** — apenas as listadas em `docs/referencias.md`.
6. **Incremental** — uma etapa por vez.

## Estrutura de Pastas

```
src/aquisicao/       — download CSVs TCU
src/preprocessamento/ — filtro, label, limpeza, split
src/modelos/         — baseline TF-IDF e Transformer LegalBert-pt
src/avaliacao/       — métricas, threshold, LIME
notebooks/           — orquestrador Colab
tests/               — testes unitários sem CSV real
```

## Comandos Principais

```bash
# Instalar dependências (sem GPU para testes locais)
pip install pandas pyarrow scikit-learn scipy nltk pytest pytest-cov

# Rodar testes (sem CSV real, sem GPU)
python -m pytest tests/ -v -k "not download"

# Rodar testes com cobertura
python -m pytest tests/ --cov=src --cov-report=term-missing

# Lint e formatação
ruff check src/ tests/
black src/ tests/
```

## Decisões Arquiteturais (D-01 a D-08) — Não Revisitar

| # | Decisão | Escolha |
|---|---|---|
| D-01 | Truncação para docs longos | Head+tail: 128 head + 382 tail = 510 + 2 especiais = 512 |
| D-02 | Modelo Transformer | `dominguesm/legal-bert-base-cased-ptbr` (LegalBert-pt) |
| D-03 | Métrica principal | F1-macro (penaliza desbalanceamento) |
| D-04 | Filtro temático | Texto em SUMARIO/ASSUNTO com termos de saúde e educação |
| D-05 | Campo de label | `SITUACAO` do CSV (contém desfecho real) |
| D-06 | Campo de texto | `SUMARIO` para baseline; `VOTO` para Transformer |
| D-07 | Estrutura CSV TCU | Pipe (`|`), UTF-8-sig, 33 colunas MAIÚSCULAS |
| D-08 | Arquiteturas alternativas | Manter LegalBert-pt (sem Longformer/BigBird/Mamba PT-BR jurídico) |

## Estrutura Real do CSV TCU

| Coluna | Índice | Uso |
|---|---|---|
| `NUMACORDAO` | 0 | Identificador |
| `SITUACAO` | variável | **Label** — desfecho do processo |
| `ASSUNTO` | 21 | Filtro temático |
| `SUMARIO` | 22 | Texto curto — baseline TF-IDF |
| `ACORDAO` | 23 | Dispositivo — fallback label |
| `VOTO` | 29 | Texto longo — Transformer |

## Parâmetros Testados e Funcionais

### TF-IDF
```python
TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42, class_weight="balanced")
```

### LegalBert-pt
```python
epocas=5, batch_size=16, lr=1e-5, weight_decay=0.01, warmup_ratio=0.10
early_stopping_patience=2  # monitorar F1-macro no val
```

### LoRA
```python
r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"]
# ~300K parâmetros treináveis de 110M (-99.7%)
```

## Estratégia de Testes

Os testes em `tests/test_pipeline.py` criam CSVs mock em memória com o formato exato
do TCU (pipe-delimited, 33 colunas MAIÚSCULAS) e não dependem de GPU, internet ou
CSV real. Executar com `pytest tests/ -v -k "not download"`.
