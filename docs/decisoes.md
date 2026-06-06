# Log de Decisões Arquiteturais

> Decisões tomadas e consolidadas. Não revisitar sem nova evidência empírica.

---

## D-01 — Estratégia de Truncação para Documentos Longos

**Status:** Aceita
**Contexto:** Modelos BERT têm limite de 512 tokens; textos de VOTO em acórdãos TCU
têm frequentemente milhares de tokens.
**Decisão:** Head+tail — primeiros 128 + últimos 382 tokens (510 + 2 especiais = 512).
**Motivo:** Acórdãos TCU têm estrutura previsível: cabeçalho/contexto no início,
Dispositivo no final. Head+tail captura as duas regiões mais discriminativas.
Respaldada por Sun et al. (2019), que demonstrou superioridade desta estratégia em
documentos jurídicos longos sobre truncação simples.
**Consequências:** Tokens intermediários (argumentação processual) são descartados.
Aceitável dado F1-macro de 0.8686 obtido com esta estratégia.

---

## D-02 — Modelo Transformer Base

**Status:** Aceita
**Contexto:** Escolha do modelo pré-treinado para fine-tuning em classificação de acórdãos.
**Decisão:** `dominguesm/legal-bert-base-cased-ptbr` (LegalBert-pt) como modelo principal;
`neuralmind/bert-base-portuguese-cased` (BERTimbau) como fallback.
**Motivo:** LegalBert-pt foi pré-treinado em corpus jurídico brasileiro (STF, petições,
decisões), reduzindo o número de épocas para convergência e melhorando F1 em vocabulário
técnico-jurídico. Chalkidis et al. (2020) e Lage-Freitas et al. (2022) confirmam a
superioridade de modelos de domínio sobre modelos genéricos em tarefas jurídicas.
**Consequências:** Dependência de modelo hospedado no HuggingFace Hub (~440 MB download).

---

## D-03 — Métrica Principal de Avaliação

**Status:** Aceita
**Contexto:** Corpus desbalanceado (~88% Irregular, ~6% cada classe minoritária).
**Decisão:** F1-macro como métrica principal de comparação entre modelos.
**Motivo:** F1-macro penaliza igualmente desempenho ruim em classes minoritárias.
Acurácia seria enganosa (modelo que prevê sempre "Irregular" obteria ~88%).
"Irregular" é a classe de maior interesse prático para o radar jurimétrico.
**Consequências:** Modelos serão selecionados por F1-macro mesmo que acurácia seja menor.

---

## D-04 — Campo e Critério do Filtro Temático

**Status:** Aceita
**Contexto:** Corpus completo inclui acórdãos de todos os temas; queremos apenas Saúde
e Educação.
**Decisão:** Filtro textual nos campos SUMARIO e ASSUNTO (OR), com termos:
`["saúde", "sus", "fnde", "merenda", "educação", "ministério da saúde",
"secretaria de saúde", "secretaria de educação"]`
**Motivo:** Mais robusto do que filtro por órgão — captura acórdãos de saúde/educação
independente do órgão gestor ou relator. Termos cobrem nível federal (SUS, FNDE) e
estadual/municipal (secretarias).
**Consequências:** Pode incluir acórdãos de saúde ocupacional ou educação corporativa
(falsos positivos temáticos). Volume filtrado esperado: 300–800 por ano.

---

## D-05 — Campo de Label

**Status:** Aceita (atualizada após inspeção do CSV real em 06/06/2026)
**Contexto:** Identificar o campo correto do CSV que contém o desfecho do processo.
**Decisão:** Campo `SITUACAO` do CSV como label principal.
**Motivo:** Inspeção do CSV real revelou que `SITUACAO` contém o desfecho da auditoria
("Irregular", "Regular com Ressalva", "Regular"). O campo `TIPO` contém apenas "Acórdão"
(tipo do documento, sem discriminação de desfecho). `ACORDAO` (dispositivo) é fallback.
**Consequências:** Extração de label requer fuzzy matching pois SITUACAO pode conter
variações como "IRREGULAR", "Irregular (Débito)", etc.

---

## D-06 — Campo de Texto para o Transformer

**Status:** Aceita (atualizada após inspeção do CSV real em 06/06/2026)
**Contexto:** Definir qual campo de texto usar como entrada do LegalBert-pt.
**Decisão:** `SUMARIO` para baseline TF-IDF; `VOTO` para Transformer.
**Motivo:** Inspeção do CSV revelou que `VOTO` (coluna 29) está disponível diretamente
no arquivo, contendo o texto integral do Voto do Ministro — sem necessidade de baixar
PDFs. Pipeline de ablação 2×2 exige execução em ambos os campos para isolar
efeito-campo vs. efeito-arquitetura.
**Consequências:** Ablação 2×2 obrigatória: TF-IDF×(SUMARIO, VOTO) e
LegalBert-pt×(SUMARIO, VOTO).

---

## D-07 — Estrutura Real do CSV TCU

**Status:** Confirmada por inspeção em 06/06/2026
**Contexto:** Necessidade de saber o formato exato dos arquivos CSV do TCU.

| Característica | Valor |
|---|---|
| Separador | pipe (`\|`) |
| Encoding | UTF-8 (possível BOM — usar `utf-8-sig`) |
| Total de colunas | 33 |
| Nomes de colunas | MAIÚSCULAS, sem acentos |
| Coluna identificador | `NUMACORDAO` |
| Coluna de texto curto | `SUMARIO` (col 22) |
| Coluna de texto longo | `VOTO` (col 29) |
| Coluna de palavras-chave | `ASSUNTO` (col 21) |
| Coluna do dispositivo | `ACORDAO` (col 23) |
| Coluna de desfecho | `SITUACAO` |

**Consequências:** Todos os `pd.read_csv` devem usar `sep='|'`, `encoding='utf-8-sig'`,
`usecols=[...]`, `dtype=str`.

---

## D-08 — Arquiteturas Alternativas (Longformer, BigBird, Mamba)

**Status:** Aceita — manter LegalBert-pt
**Contexto:** Avaliar se arquiteturas de janela longa são substitutos viáveis.
**Decisão:** Manter LegalBert-pt com head+tail.
**Motivo:** Três critérios ordenados por prioridade:
1. **Domínio/língua (determinante):** Longformer, BigBird e Mamba não têm versões
   pré-treinadas em português jurídico brasileiro.
2. **Suficiência do head+tail:** A estrutura previsível do acórdão TCU torna os tokens
   intermediários (argumentação) marginalmente informativos para predição de desfecho.
3. **Viabilidade computacional:** Longformer/BigBird exigem ~4× mais VRAM que BERT-base
   para sequências de 4.096 tokens; Mamba requer kernels CUDA incompatíveis com Colab.
**Trabalho futuro:** Comparação válida se surgir Longformer/SSM pré-treinado em PT-BR jurídico.
