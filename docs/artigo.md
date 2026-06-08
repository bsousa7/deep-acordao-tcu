# Jurimetria Preditiva em Acórdãos do TCU: Comparação entre TF-IDF e LegalBert-pt em Cenário de Baixo Recurso

**Bruno Aires¹ · Candice Trigueiro¹ · Rafael Ayoroa¹**

¹ Programa de Pós-Graduação — [Instituição]

`bruno.aires9@gmail.com`

---

## Resumo

Este trabalho investiga a classificação automática de acórdãos do Tribunal de Contas da União (TCU) nas áreas de Saúde e Educação em três desfechos processuais: *Irregular*, *Regular com Ressalva* e *Regular*. Utilizando 4.444 acórdãos do Portal de Dados Abertos do TCU (2016–2024), conduzimos uma ablação 2×2 entre dois modelos — TF-IDF com Regressão Logística e LegalBert-pt com LoRA — e dois campos de texto — SUMARIO e VOTO. O F1-macro foi adotado como métrica principal em razão do desbalanceamento extremo do corpus (92,5% *Irregular*). O melhor resultado foi obtido pelo modelo TF-IDF+SUMARIO (F1-macro = 0,7477; IC95%: [0,687; 0,809]), superando o LegalBert-pt em todos os cenários avaliados. Identificamos que o desempenho reduzido do Transformer decorre da combinação de dataset de pequeno porte com desbalanceamento severo de classes, limitando o sinal de gradiente para as classes minoritárias. Na otimização de threshold com função de custo assimétrica (FN = 10×FP), obtivemos recall perfeito na classe *Irregular* (1,000) com threshold = 0,08, configuração relevante para sistemas de alerta em controle externo.

**Palavras-chave:** jurimetria, controle externo, TCU, classificação de texto, BERT, LoRA, desbalanceamento de classes.

---

## Abstract

This work investigates the automatic classification of decisions from the Brazilian Federal Court of Accounts (TCU) in the Health and Education domains into three outcome classes: *Irregular*, *Regular with Qualification*, and *Regular*. Using 4,444 rulings from the TCU Open Data Portal (2016–2024), we conducted a 2×2 ablation between two models — TF-IDF with Logistic Regression and LegalBert-pt with LoRA — and two text fields — SUMARIO and VOTO. F1-macro was adopted as the primary metric given the extreme class imbalance (92.5% Irregular). The best result was achieved by TF-IDF+SUMARIO (F1-macro = 0.7477; 95%CI: [0.687; 0.809]), outperforming LegalBert-pt in all evaluated scenarios. We identify that the Transformer's reduced performance stems from the combination of a small dataset with severe class imbalance, limiting gradient signal for minority classes. In threshold optimization with an asymmetric cost function (FN = 10×FP), we achieved perfect recall on the Irregular class (1.000) with threshold = 0.08, a configuration relevant for alert systems in external auditing.

**Keywords:** jurimetrics, external control, TCU, text classification, BERT, LoRA, class imbalance.

---

## 1. Introdução

O Tribunal de Contas da União (TCU) é o órgão de controle externo responsável pela fiscalização do uso dos recursos públicos federais brasileiros. Anualmente, o TCU profere milhares de acórdãos que julgam a regularidade das contas de gestores públicos, classificando-as como *Irregular*, *Regular com Ressalva* ou *Regular*. A identificação automática desses desfechos a partir do texto dos acórdãos constitui uma aplicação natural de técnicas de Processamento de Linguagem Natural (PLN), com potencial para apoiar sistemas de auditoria inteligente, priorização de investigações e análise jurimétrica em larga escala.

A tarefa apresenta desafios específicos: (i) desbalanceamento severo de classes, com predominância massiva de acórdãos irregulares nas áreas de Saúde e Educação; (ii) textos jurídicos longos, com linguagem técnica e estrutura específica; (iii) volume limitado de dados rotulados disponíveis; e (iv) necessidade de recall elevado na classe de maior risco (*Irregular*), pois falsos negativos têm consequências institucionais severas.

Neste cenário, surge a pergunta central deste trabalho: **modelos Transformer de domínio específico superam abordagens clássicas de bag-of-words em classificação de acórdãos TCU?** Formulamos uma ablação controlada 2×2 — cruzando arquitetura (TF-IDF vs. LegalBert-pt) com campo de texto (SUMARIO vs. VOTO) — para isolar o efeito de cada fator sobre o F1-macro.

As principais contribuições do trabalho são:
1. Dataset filtrado e rotulado de 4.444 acórdãos TCU (Saúde e Educação, 2016–2024) obtido exclusivamente de fonte oficial aberta;
2. Ablação 2×2 sistemática com validação cruzada estratificada de 5 folds e intervalos de confiança de 95%;
3. Diagnóstico técnico de falha intermitente do LegalBert-pt causada por ausência dos pesos do *pooler* no checkpoint original;
4. Otimização de threshold com função de custo assimétrica orientada ao controle externo.

---

## 2. Trabalhos Relacionados

### 2.1 Predição de Decisões Judiciais

Aletras et al. (2016) foram pioneiros ao aplicar PLN para prever decisões do Tribunal Europeu de Direitos Humanos com F1 de 0,79 usando SVMs sobre n-gramas. Medvedeva et al. (2020) expandiram essa linha com análise sistemática de variações metodológicas. No contexto brasileiro, Lage-Freitas et al. (2022) propuseram modelos de predição de decisões judiciais do STJ, demonstrando F1-macro acima de 0,80 em cenários balanceados.

### 2.2 Modelos de Linguagem para Português Jurídico

Souza, Nogueira e Lotufo (2020) introduziram o BERTimbau, modelo BERT pré-treinado em corpus generalista em português, que serviu como base para especializações posteriores. Domingues (2022) disponibilizou o LegalBert-pt (`dominguesm/legal-bert-base-cased-ptbr`), pré-treinado em corpus jurídico brasileiro incluindo decisões do STF, petições e acórdãos — o qual utilizamos neste trabalho.

### 2.3 Adaptação Eficiente de Parâmetros (PEFT)

Hu et al. (2022) propuseram o LoRA (*Low-Rank Adaptation*), que insere matrizes de baixo rank nas camadas de atenção, reduzindo drasticamente o número de parâmetros treináveis sem degradação significativa de desempenho. Esta técnica é especialmente relevante em cenários com restrição computacional e datasets pequenos, pois reduz o risco de overfitting.

### 2.4 Desbalanceamento em Classificação Jurídica

Lin et al. (2017) propuseram a *Focal Loss* como alternativa ao cross-entropy ponderado para classes muito desbalanceadas. Sun et al. (2019) demonstraram que a estratégia head+tail de truncação supera a truncação simples em documentos jurídicos longos, preservando tanto o contexto processual (cabeçalho) quanto o dispositivo da decisão (final do texto).

---

## 3. Dados e Metodologia

### 3.1 Fonte de Dados

Utilizamos exclusivamente os arquivos CSV de acórdãos completos disponibilizados pelo Portal de Dados Abertos do TCU (Brasil, 2024), cobrindo os anos de 2016 a 2024. Os arquivos são fornecidos no formato pipe-delimited (`|`), codificação UTF-8-sig, com 33 colunas em maiúsculas. O campo `VOTO` (coluna 29) contém o texto integral do voto do ministro relator; `SUMARIO` (coluna 22) contém a ementa estruturada; `ACORDAO` (coluna 23) contém o dispositivo da decisão.

### 3.2 Extração de Label

O campo `SITUACAO` registra o status de publicação ("OFICIALIZADO") para acórdãos a partir de 2020, e não o desfecho processual. Implementamos extração hierárquica de label: (1) mapeamento direto de `SITUACAO` para acórdãos anteriores a 2020; (2) busca por expressões regulares no `SUMARIO` (*e.g.*, `\bcontas?\s+irregulares?\b`); (3) busca no dispositivo `ACORDAO` como fallback. Este pipeline recuperou labels para 4.444 dos 60.000+ acórdãos nos anos estudados, após o filtro temático.

### 3.3 Filtro Temático

Mantivemos apenas acórdãos relacionados às áreas de Saúde e Educação, identificados por presença dos termos `["saúde", "sus", "fnde", "merenda", "educação", "ministério da saúde", "secretaria de saúde", "secretaria de educação"]` nos campos SUMARIO ou ASSUNTO (operador OR, case-insensitive).

### 3.4 Corpus Final

A Tabela 1 apresenta a distribuição do corpus final.

**Tabela 1 — Distribuição do corpus por classe e por ano.**

| Ano  | Irregular | Reg. c/ Ressalva | Regular | Total |
|------|----------:|----------------:|--------:|------:|
| 2016 | 286       | 10              | 12      | 308   |
| 2017 | 543       | 17              | 10      | 570   |
| 2018 | 460       | 12              | 8       | 480   |
| 2019 | 401       | 18              | 8       | 427   |
| 2020 | 533       | 19              | 12      | 564   |
| 2021 | 573       | 21              | 27      | 621   |
| 2022 | 509       | 21              | 18      | 548   |
| 2023 | 367       | 25              | 16      | 408   |
| 2024 | 440       | 54              | 24      | 518   |
| **Total** | **4.112** | **197** | **135** | **4.444** |

A classe *Irregular* representa 92,5% do corpus, *Regular com Ressalva* 4,4% e *Regular* 3,0%. Este desbalanceamento extremo motivou o uso de F1-macro como métrica principal (D-03) e de pesos de classe balanceados no treinamento.

### 3.5 Pré-processamento e Divisão

Para o modo TF-IDF: conversão para minúsculas, remoção de stopwords e pontuação. Para o modo BERT: preservação de caixa (modelo case-sensitive), normalização Unicode e remoção de quebras de linha excessivas. Textos com mais de 510 tokens recebem truncação head+tail: primeiros 128 + últimos 382 tokens (Sun et al., 2019), capturando contexto processual e dispositivo final.

O corpus foi dividido de forma estratificada em treino (70%, 3.110 acórdãos), validação (15%, 667) e teste (15%, 667). Para a validação cruzada K-Fold, utilizamos o conjunto de treino com 5 folds estratificados, `RANDOM_STATE = 42`.

### 3.6 Modelos

**TF-IDF + Regressão Logística:** Vetorizador TF-IDF com `max_features = 50.000`, `ngram_range = (1,2)`, `sublinear_tf = True`, `min_df = 2`; Regressão Logística com `C = 1,0`, `solver = lbfgs`, `class_weight = "balanced"`, `max_iter = 1.000` (Pedregosa et al., 2011).

**LegalBert-pt com LoRA:** Fine-tuning do modelo `dominguesm/legal-bert-base-cased-ptbr` (Domingues, 2022) com adaptadores LoRA (Hu et al., 2022) de rank `r = 16`, `lora_alpha = 32`, `lora_dropout = 0,1`, inseridos nas matrizes de consulta e valor (*query*, *value*) de todas as camadas de atenção. Parâmetros treináveis: 1.182.723 de 126.571.782 totais (0,93%). Treinamento com cross-entropy ponderado, pesos de classe balanceados normalizados pela média, `lr = 3×10⁻⁵`, `warmup_ratio = 0,10`, `fp16 = True`, *early stopping* com paciência de 2 epochs monitorando F1-macro no conjunto de validação.

**Correção técnica — pooler ausente:** O checkpoint do LegalBert-pt não inclui os pesos de `bert.pooler.dense`. A inicialização padrão do PyTorch (Kaiming uniforme) produzia normas elevadas (~15,3 para pesos que deveriam ter norma ~0,3), saturando o tanh do pooler e travando o gradiente do token `[CLS]`. Aplicamos reinicialização explícita com `N(0, σ²)` onde `σ = 0,02` (valor padrão de configuração BERT), e descongelamos os parâmetros do pooler após a aplicação do LoRA.

---

## 4. Experimentos

### 4.1 Protocolo de Avaliação

Todos os experimentos utilizam validação cruzada estratificada de 5 folds sobre o conjunto de treino. Reportamos F1-macro médio, desvio padrão e intervalo de confiança de 95% calculado pela distribuição t de Student com `n-1 = 4` graus de liberdade. A Tabela 2 apresenta a ablação completa.

**Tabela 2 — Ablação 2×2: F1-macro (média ± desvio padrão, IC 95%) e acurácia.**

| Modelo | Campo | F1-macro | ± std | IC 95% | Acurácia |
|--------|-------|:--------:|:-----:|:------:|:--------:|
| TF-IDF + LogReg | SUMARIO | **0,7477** | 0,0492 | [0,687; 0,809] | 0,9498 |
| TF-IDF + LogReg | VOTO | 0,5536 | 0,0256 | [0,522; 0,585] | 0,9395 |
| LegalBert-pt | VOTO | 0,4180 | 0,0426 | [0,365; 0,471] | 0,9325 |
| LegalBert-pt | SUMARIO | 0,4095 | 0,0249 | [0,379; 0,440] | 0,9135 |

### 4.2 Otimização de Threshold

Para aplicação em sistemas de alerta de controle externo, modelamos a função de custo assimétrica com falso negativo penalizado 10× (custo de não detectar irregularidade) em relação ao falso positivo. Realizamos varredura no intervalo [0,05; 0,95] com passo de 0,005 sobre as probabilidades do classificador TF-IDF+SUMARIO (melhor modelo). O threshold ótimo encontrado foi **0,08**, resultando em:

- **Recall-Irregular = 1,000** (zero falsos negativos)
- **Precision-Irregular = 0,949**
- **F1-Irregular = 0,974**
- Custo total da função assimétrica: 33

---

## 5. Resultados e Discussão

### 5.1 TF-IDF supera LegalBert-pt em todos os cenários

O modelo TF-IDF+SUMARIO atingiu F1-macro de 0,7477, superando o LegalBert-pt+VOTO em 33 pontos percentuais (0,4180) e o LegalBert-pt+SUMARIO em 35 pontos (0,4095). Este resultado contraria a expectativa padrão de superioridade dos Transformers, mas é consistente com achados em cenários de baixo recurso (Medvedeva et al., 2020).

### 5.2 Por que TF-IDF+SUMARIO é superior

O campo SUMARIO contém ementas estruturadas com expressões formulaicas como "CONTAS IRREGULARES", "CONTAS REGULARES COM RESSALVA" e "CONTAS REGULARES". Bigramas TF-IDF capturam esses padrões com alta precisão. O LegalBert-pt, ao processar as mesmas expressões, introduz custo computacional sem ganho discriminativo — a tarefa é trivial para n-gramas.

### 5.3 VOTO é mais difícil que SUMARIO para ambos os modelos

No campo VOTO (raciocínio jurídico longo, 2.000–8.000 tokens), TF-IDF cai de 0,7477 para 0,5536 (-20pp): o ruído lexical dilui os marcadores discriminativos. O LegalBert-pt sobe levemente de 0,4095 (SUMARIO) para 0,4180 (VOTO), sugerindo que o modelo captura algum sinal semântico em textos longos, mas insuficiente para superar o TF-IDF.

### 5.4 Limitações do LegalBert-pt neste cenário

Identificamos três fatores que limitam o desempenho do Transformer:

1. **Dataset pequeno com desbalanceamento extremo:** O conjunto de treino por fold contém apenas ~185 exemplos de classes minoritárias (Regular + Regular com Ressalva). Com LoRA de rank 16 (~1,2M parâmetros), o sinal de gradiente dessas classes é insuficiente para deslocar a fronteira de decisão de forma robusta.

2. **Ausência do pooler no checkpoint:** O LegalBert-pt foi publicado sem os pesos de `bert.pooler.dense`. A inicialização aleatória padrão do PyTorch resulta em saturação do tanh, travando o gradiente do token `[CLS]`. A correção implementada (reinicialização N(0; 0,02) + descongelamento do pooler) eliminou o travamento mas não resolveu o problema de volume de dados.

3. **Acurácia enganosa:** A alta acurácia de todos os modelos (91–95%) reflete a predominância da classe *Irregular* (92,5%) e não o aprendizado efetivo. O modelo trivial "prever sempre Irregular" obteria ~92,5% de acurácia — reforçando a escolha do F1-macro como métrica principal.

### 5.5 Threshold ótimo para controle externo

A otimização com custo assimétrico (FN = 10×FP) reduziu o threshold de decisão de 0,50 (padrão) para 0,08, zerando os falsos negativos na classe *Irregular* ao custo de um pequeno aumento em falsos positivos (precision = 0,949). Para um sistema de radar jurimétrico, esta configuração é preferível: investigar um processo Regular por equívoco tem custo administrativo baixo, mas deixar de detectar uma irregularidade tem impacto institucional e financeiro significativo.

---

## 6. Conclusão

Este trabalho apresentou uma ablação controlada 2×2 para classificação automática de acórdãos TCU nas áreas de Saúde e Educação. O melhor resultado foi obtido pelo modelo TF-IDF com Regressão Logística aplicado ao campo SUMARIO (F1-macro = 0,7477), superando o modelo LegalBert-pt com LoRA em todos os cenários avaliados.

O achado central — TF-IDF supera LegalBert-pt em cenário de baixo recurso com desbalanceamento extremo — tem implicação prática direta: para datasets jurídicos com menos de 5.000 exemplos e desbalanceamento >90%, modelos baseados em n-gramas oferecem melhor relação custo-benefício que Transformers. O LegalBert-pt deve ser preferido quando houver volume suficiente de exemplos de classes minoritárias para alimentar o sinal de gradiente dos adaptadores LoRA.

Como trabalho futuro, destacamos: (i) expansão do corpus para anos anteriores a 2016 e inclusão de outras áreas temáticas do TCU; (ii) comparação com BERTimbau sem LoRA (*full fine-tuning*) em configuração de hold-out simples; (iii) investigação de *focal loss* como alternativa ao cross-entropy ponderado; e (iv) integração com modelos generativos para extração automática de labels em acórdãos sem marcadores estruturados.

---

## Referências

ALETRAS, Nikolaos et al. **Predicting judicial decisions of the European Court of Human Rights: A Natural Language Processing perspective**. *PeerJ Computer Science*, v. 2, e93, 2016.

BRASIL. Tribunal de Contas da União. **Portal de Dados Abertos do TCU: Acórdãos Completos**. Brasília: TCU, 2024. Disponível em: <https://sites.tcu.gov.br/dados-abertos/jurisprudencia/>. Acesso em: 05 jun. 2026.

DEVLIN, Jacob et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. In: CONFERENCE OF THE NORTH AMERICAN CHAPTER OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. *Proceedings...* Stroudsburg: ACL, 2019. p. 4171–4186.

DOMINGUES, Luciano. **legal-bert-base-cased-ptbr: BERT model pre-trained on Brazilian legal corpus**. HuggingFace Hub, 2022. Disponível em: <https://huggingface.co/dominguesm/legal-bert-base-cased-ptbr>. Acesso em: 05 jun. 2026.

HU, Edward J. et al. **LoRA: Low-Rank Adaptation of Large Language Models**. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 2022, online. *Proceedings...* [S.l.]: OpenReview, 2022.

LAGE-FREITAS, André et al. **Predicting Brazilian court decisions**. *PeerJ Computer Science*, v. 8, e904, 2022.

LIN, Tsung-Yi et al. **Focal Loss for Dense Object Detection**. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. *Proceedings...* [S.l.]: IEEE, 2017. p. 2980–2988.

MEDVEDEVA, Masha; VOLS, Michel; WIELING, Martijn. **Using machine learning to predict decisions of the European Court of Human Rights**. *Artificial Intelligence and Law*, v. 28, n. 2, p. 237–266, 2020.

PEDREGOSA, Fabian et al. **Scikit-learn: machine learning in Python**. *Journal of Machine Learning Research*, Cambridge, v. 12, p. 2825–2830, 2011.

RIBEIRO, Marco Tulio; SINGH, Sameer; GUESTRIN, Carlos. **"Why should I trust you?": explaining the predictions of any classifier**. In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., 2016, San Francisco. *Proceedings…* New York: ACM, 2016. p. 1135–1144.

SOUZA, Fábio; NOGUEIRA, Rodrigo; LOTUFO, Roberto. **BERTimbau: Pretrained BERT Models for Brazilian Portuguese**. In: INTELLIGENT SYSTEMS — BRACIS, 9., 2020, Rio Grande. *Proceedings...* Cham: Springer, 2020. p. 403–417.

SUN, Chi; QIU, Xipeng; XU, Yuanbin; HUANG, Xuanjing. **How to Fine-Tune BERT for Text Classification?** In: CHINESE COMPUTATIONAL LINGUISTICS, 18., 2019, Kunming. *Proceedings...* Cham: Springer, 2019. p. 194–206.

TVEITA, Sondre; HUSTAD, Eli. **Benefits and challenges of AI in the public sector**. In: HAWAII INTERNATIONAL CONFERENCE ON SYSTEM SCIENCES, 58., 2025, Maui. *Proceedings...* Honolulu: University of Hawaii, 2025. p. 1–10.

WOLF, Thomas et al. **Transformers: state-of-the-art natural language processing**. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM DEMONSTRATIONS, 2020, online. *Proceedings…* Stroudsburg: ACL, 2020. p. 38–45.
