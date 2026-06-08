# deep-acordao-tcu

Jurimetria preditiva em acórdãos do Tribunal de Contas da União (TCU) nas áreas de
**Saúde** e **Educação**: dado o texto de um acórdão, predizer o desfecho do processo.

## Classes de Desfecho

| Classe | Descrição |
|---|---|
| `Irregular` | Contas julgadas irregulares — multa e/ou condenação |
| `Regular com Ressalva` | Aprovadas com ressalvas formais |
| `Regular` | Aprovadas sem restrições |

## Resultados (Ablação 2×2)

Corpus: 4.444 acórdãos TCU — Saúde e Educação, 2016–2024. Validação cruzada 5-fold estratificada.

| Arquitetura | Campo | F1-macro | IC 95% | Acurácia |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | SUMARIO | **0,7477** | [0,687; 0,809] | 0,9498 |
| TF-IDF + LogisticRegression | VOTO | 0,5536 | [0,522; 0,585] | 0,9395 |
| LegalBert-pt + LoRA | VOTO | 0,4180 | [0,365; 0,471] | 0,9325 |
| LegalBert-pt + LoRA | SUMARIO | 0,4095 | [0,379; 0,440] | 0,9135 |

Threshold ótimo (custo FN=10×FP): **0,08** → Recall-Irregular = 1,000 · Precision = 0,949

## Instalação

```bash
# Clonar repositório
git clone https://github.com/bsousa7/deep-acordao-tcu.git
cd deep-acordao-tcu

# Ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Dependências
pip install -r requirements.txt

# Para Colab T4 (CUDA 12.1)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline

```
Etapa 1: Download CSVs TCU 2020–2024  →  data/raw/
Etapa 2: Filtro temático + label       →  data/interim/acordaos_filtrados.parquet
Etapa 3: EDA                           →  resultados/figuras/eda_*.png
Etapa 4: Limpeza + split 70/15/15      →  data/processed/
Etapa 5: Baseline TF-IDF (ablação 2×2) →  resultados/figuras/baseline_*.png
Etapa 6: Fine-tuning LegalBert-pt       →  resultados/modelos/transformer/
Etapa 7: Comparação + threshold ótimo  →  resultados/metricas.json
Etapa 8: Explicabilidade LIME           →  resultados/figuras/lime_*.html
```

## Uso Rápido

```python
# Executar o notebook orquestrador no Google Colab:
# notebooks/00_projeto_completo.ipynb
```

## Estrutura de Arquivos

```
deep-acordao-tcu/
├── src/
│   ├── aquisicao/baixar_csvs.py
│   ├── preprocessamento/{filtrar_tematico,limpeza}.py
│   ├── modelos/{baseline,transformer}.py
│   └── avaliacao/metricas.py
├── notebooks/00_projeto_completo.ipynb
├── tests/test_pipeline.py
├── docs/{decisoes,referencias}.md
└── resultados/
```

## Decisões Técnicas

Consulte `docs/decisoes.md` para o log completo de decisões arquiteturais (D-01 a D-08).
Ponto central: **LegalBert-pt** pré-treinado em corpus jurídico brasileiro com estratégia
**head+tail** (128+382 tokens) — captura contexto processual + dispositivo final.

## Fonte de Dados

Portal de Dados Abertos do TCU — acórdãos completos, arquivos oficiais CSV:
`https://sites.tcu.gov.br/dados-abertos/jurisprudencia/`

Uso conforme política de dados abertos do TCU. Sem scraping.

## Referências

Ver `docs/referencias.md`.
