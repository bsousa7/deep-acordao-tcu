# deep-acordao-tcu

> Mestrado em Administração Pública — Ciência de Dados e Inteligência Artificial
> Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)
> **Autores:** Bruno Aires · Candice Trigueiro · Rafael Ayoroa

---

## Objetivo

O Tribunal de Contas da União (TCU) profere milhares de acórdãos por ano julgando a regularidade do uso de recursos públicos federais nas áreas de Saúde e Educação, abrangendo desde ministérios até secretarias estaduais e municipais que recebem transferências da União. Identificar automaticamente o desfecho desses julgamentos — *Irregular*, *Regular com Ressalva* ou *Regular* — é uma tarefa de alto valor para o controle externo, pois permite priorizar investigações, antecipar riscos e escalar a capacidade analítica dos órgãos de fiscalização sem ampliar proporcionalmente o quadro de auditores.

Este projeto constrói e avalia um pipeline completo de jurimetria preditiva: da coleta automática dos CSVs oficiais do TCU (2016–2024) à classificação supervisionada do texto dos acórdãos com modelos de linguagem, incluindo otimização de threshold orientada ao custo assimétrico do controle externo — onde deixar de detectar uma irregularidade tem impacto institucional muito maior do que um falso alarme.

---

## Modelos Utilizados

Foram avaliadas duas arquiteturas em uma ablação controlada 2×2, cruzando modelo e campo de texto. O **TF-IDF com Regressão Logística** (50.000 features, bigramas, pesos de classe balanceados) representa a abordagem clássica de recuperação de informação: vetoriza o texto em frequências ponderadas de n-gramas e treina um classificador linear. O **LegalBert-pt** (`dominguesm/legal-bert-base-cased-ptbr`) é um modelo BERT pré-treinado em corpus jurídico brasileiro (STF, petições, acórdãos), adaptado ao problema via LoRA (*Low-Rank Adaptation*, r=16) — técnica que congela os 125 M de parâmetros originais e treina apenas ~1,2 M de parâmetros adicionais de baixo rank, tornando o fine-tuning viável em GPU de consumo (T4 16 GB).

Os dois modelos foram testados sobre dois campos de texto dos acórdãos: o **SUMARIO** (ementa estruturada, curta e formulaica) e o **VOTO** (raciocínio jurídico completo do ministro relator, longo e livre). Textos com mais de 512 tokens receberam truncação *head+tail* (128 tokens iniciais + 382 finais), estratégia que preserva o contexto processual e o dispositivo da decisão simultaneamente.

---

## Resultados (Ablação 2×2)

Corpus: 4.444 acórdãos TCU — Saúde e Educação, 2016–2024. Validação cruzada estratificada de 5 folds (RANDOM_STATE = 42).

| Arquitetura | Campo | F1-macro | IC 95% | Acurácia |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | SUMARIO | **0,7477** | [0,687; 0,809] | 0,9498 |
| TF-IDF + LogisticRegression | VOTO | 0,5536 | [0,522; 0,585] | 0,9395 |
| LegalBert-pt + LoRA | VOTO | 0,4180 | [0,365; 0,471] | 0,9325 |
| LegalBert-pt + LoRA | SUMARIO | 0,4095 | [0,379; 0,440] | 0,9135 |

Threshold ótimo (custo FN=10×FP): **0,08** → Recall-Irregular = 1,000 · Precision = 0,949

---

## Impacto para o Serviço Público

A capacidade de classificar automaticamente acórdãos do TCU cria uma camada de inteligência analítica que potencializa o trabalho de auditores sem substituí-los. Com o modelo TF-IDF + SUMARIO (F1-macro = 0,7477) operando com threshold ajustado para recall perfeito na classe *Irregular*, é possível construir um sistema de radar que sinaliza automaticamente processos de alto risco nas áreas de Saúde e Educação — áreas que concentram volumes expressivos de recursos federais transferidos a milhares de municípios —, permitindo que equipes de controle externo foquem esforços onde o risco de irregularidade é maior.

Além da triagem, o pipeline gera explicabilidade via LIME, identificando quais termos do texto sustentam cada predição. Esse recurso é fundamental para a adoção em contexto governamental: auditores podem verificar o raciocínio do modelo, detectar vieses e apresentar as decisões algorítmicas com transparência a gestores e ao público. O código é integralmente aberto, reproduzível e utiliza apenas dados oficiais do Portal de Dados Abertos do TCU, garantindo conformidade com a política de dados abertos do governo federal.

---

## Conclusão

O principal achado deste trabalho é que, em cenários de baixo recurso com desbalanceamento extremo de classes (92,5% *Irregular*), modelos clássicos de n-gramas superam Transformers de domínio específico. O TF-IDF captura com precisão os marcadores formulaicos do SUMARIO ("CONTAS IRREGULARES", "CONTAS REGULARES COM RESSALVA"), enquanto o LegalBert-pt enfrenta limitações de gradiente decorrentes do reduzido volume de exemplos de classes minoritárias (~185 por fold de treino). Este resultado tem implicação prática direta: para datasets jurídicos com menos de 5.000 exemplos e desbalanceamento superior a 90%, abordagens bag-of-words oferecem melhor custo-benefício que Transformers com LoRA.

Como trabalho futuro, destacam-se a expansão do corpus para incluir todos os tipos de acórdãos do TCU (Auditorias, Tomadas de Contas Especiais, Representações), o uso de *focal loss* para desbalanceamento severo, e a integração com modelos generativos para extração automática de labels em acórdãos sem marcadores estruturados no SUMARIO. O artigo completo está disponível em [`docs/artigo_abnt.pdf`](docs/artigo_abnt.pdf).

---

## Instalação

```bash
git clone https://github.com/bsousa7/deep-acordao-tcu.git
cd deep-acordao-tcu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU (Colab T4 / CUDA 12.1):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline

```
Etapa 1: Download CSVs TCU 2016–2024  →  data/raw/
Etapa 2: Filtro temático + label       →  data/interim/acordaos_filtrados.parquet
Etapa 3: EDA                           →  resultados/figuras/eda_*.png
Etapa 4: Limpeza + split 70/15/15      →  data/processed/
Etapa 5: Baseline TF-IDF (ablação 2×2) →  resultados/figuras/baseline_*.png
Etapa 6: Fine-tuning LegalBert-pt      →  resultados/modelos/transformer/
Etapa 7: Comparação + threshold ótimo  →  resultados/metricas.json
Etapa 8: Explicabilidade LIME          →  resultados/figuras/lime_*.html
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
├── docs/{decisoes,referencias,artigo}.md
├── docs/artigo_abnt.pdf
└── resultados/metricas.json
```

## Fonte de Dados

Portal de Dados Abertos do TCU — acórdãos completos, arquivos CSV oficiais:
`https://sites.tcu.gov.br/dados-abertos/jurisprudencia/`

Uso conforme política de dados abertos do TCU. Sem scraping. Os arquivos CSV não são versionados (`.gitignore`) por conta do tamanho (~2 GB para 9 anos) e são baixados automaticamente pelo pipeline.

## Referências

Ver [`docs/referencias.md`](docs/referencias.md). Artigo completo: [`docs/artigo_abnt.pdf`](docs/artigo_abnt.pdf).
