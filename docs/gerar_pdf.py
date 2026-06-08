#!/usr/bin/env python3
"""Gera PDF do artigo científico com formatação ABNT usando ReportLab."""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Registrar Liberation Serif (equivalente métrico a Times New Roman)
# ---------------------------------------------------------------------------
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
pdfmetrics.registerFont(TTFont("LiberSerif",   str(FONT_DIR / "LiberationSerif-Regular.ttf")))
pdfmetrics.registerFont(TTFont("LiberSerifB",  str(FONT_DIR / "LiberationSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont("LiberSerifI",  str(FONT_DIR / "LiberationSerif-Italic.ttf")))
pdfmetrics.registerFont(TTFont("LiberSerifBI", str(FONT_DIR / "LiberationSerif-BoldItalic.ttf")))
pdfmetrics.registerFontFamily(
    "LiberSerif",
    normal="LiberSerif", bold="LiberSerifB",
    italic="LiberSerifI", boldItalic="LiberSerifBI"
)

# ---------------------------------------------------------------------------
# Estilos ABNT
# ---------------------------------------------------------------------------
BODY      = 12
SMALL     = 10
TITLE_SZ  = 14
SPACING   = BODY * 1.5 * 0.3528   # 1,5× em mm → pontos ≈ 18pt

def build_styles():
    s = getSampleStyleSheet()

    base = dict(fontName="LiberSerif", fontSize=BODY, leading=BODY * 1.5)

    styles = {
        "titulo": ParagraphStyle("titulo", fontName="LiberSerifB", fontSize=TITLE_SZ,
                                 leading=TITLE_SZ * 1.4, alignment=TA_CENTER,
                                 spaceAfter=6),
        "autores": ParagraphStyle("autores", fontName="LiberSerif", fontSize=BODY,
                                  leading=BODY * 1.5, alignment=TA_CENTER),
        "afiliacao": ParagraphStyle("afiliacao", fontName="LiberSerifI", fontSize=SMALL,
                                    leading=SMALL * 1.5, alignment=TA_CENTER),
        "body": ParagraphStyle("body", **base, alignment=TA_JUSTIFY,
                               firstLineIndent=12.5 * mm, spaceAfter=3),
        "body_no_indent": ParagraphStyle("body_no_indent", **base, alignment=TA_JUSTIFY,
                                         firstLineIndent=0, spaceAfter=3),
        "section": ParagraphStyle("section", fontName="LiberSerifB", fontSize=BODY,
                                  leading=BODY * 1.5, alignment=TA_LEFT,
                                  spaceBefore=12, spaceAfter=6, textTransform="uppercase"),
        "subsection": ParagraphStyle("subsection", fontName="LiberSerifB", fontSize=BODY,
                                     leading=BODY * 1.5, alignment=TA_LEFT,
                                     spaceBefore=8, spaceAfter=4),
        "abstract_label": ParagraphStyle("abstract_label", fontName="LiberSerifB",
                                         fontSize=SMALL, leading=SMALL * 1.5,
                                         spaceAfter=2),
        "abstract_text": ParagraphStyle("abstract_text", fontName="LiberSerif",
                                        fontSize=SMALL, leading=SMALL * 1.5,
                                        alignment=TA_JUSTIFY, leftIndent=0, spaceAfter=4),
        "keywords": ParagraphStyle("keywords", fontName="LiberSerif", fontSize=SMALL,
                                   leading=SMALL * 1.5, alignment=TA_LEFT, spaceAfter=8),
        "table_cap": ParagraphStyle("table_cap", fontName="LiberSerifB", fontSize=SMALL,
                                    leading=SMALL * 1.5, alignment=TA_LEFT,
                                    spaceBefore=8, spaceAfter=2),
        "table_fonte": ParagraphStyle("table_fonte", fontName="LiberSerifI", fontSize=SMALL,
                                      leading=SMALL * 1.5, alignment=TA_LEFT, spaceAfter=6),
        "ref": ParagraphStyle("ref", fontName="LiberSerif", fontSize=SMALL,
                              leading=SMALL * 1.3, alignment=TA_JUSTIFY,
                              firstLineIndent=0, leftIndent=12.5 * mm,
                              spaceAfter=4),
        "ref_heading": ParagraphStyle("ref_heading", fontName="LiberSerifB", fontSize=BODY,
                                      leading=BODY * 1.5, alignment=TA_LEFT,
                                      spaceBefore=0, spaceAfter=8,
                                      textTransform="uppercase"),
    }
    return styles


# ---------------------------------------------------------------------------
# Numeração de seção
# ---------------------------------------------------------------------------
class Counter:
    def __init__(self):
        self.sec = 0
        self.sub = 0

    def section(self, title: str, styles) -> list:
        self.sec += 1
        self.sub = 0
        return [Spacer(1, 4 * mm),
                Paragraph(f"{self.sec}  {title.upper()}", styles["section"])]

    def subsection(self, title: str, styles) -> list:
        self.sub += 1
        return [Spacer(1, 2 * mm),
                Paragraph(f"{self.sec}.{self.sub}  {title}", styles["subsection"])]


# ---------------------------------------------------------------------------
# Tabelas ABNT
# ---------------------------------------------------------------------------
def make_table(headers, rows, col_widths_mm):
    col_widths = [w * mm for w in col_widths_mm]
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONTNAME",    (0, 0), (-1, 0),  "LiberSerifB"),
        ("FONTSIZE",    (0, 0), (-1, -1), SMALL),
        ("LEADING",     (0, 0), (-1, -1), SMALL * 1.3),
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.Color(0.85, 0.85, 0.85)),
        ("ALIGN",       (0, 0), (-1, 0),  "CENTER"),
        ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",       (0, 1), (0, -1),  "LEFT"),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.black),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Zebragem nas linhas de dados
    for i, _ in enumerate(rows):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i + 1), (-1, i + 1),
                           colors.Color(0.96, 0.96, 0.96)))
    # Linha de total em negrito
    if rows and rows[-1][0].lower() in ("total",):
        style.append(("FONTNAME", (0, len(rows)), (-1, len(rows)), "LiberSerifB"))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Numeração de páginas
# ---------------------------------------------------------------------------
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("LiberSerif", SMALL)
    canvas.drawRightString(210 * mm - 20 * mm, 15 * mm, str(doc.page))
    canvas.restoreState()


# ===========================================================================
# Construção do documento
# ===========================================================================
def build_pdf() -> Path:
    out = Path(__file__).parent / "artigo_abnt.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=30 * mm, rightMargin=20 * mm,
        topMargin=30 * mm, bottomMargin=20 * mm,
        title="Jurimetria Preditiva em Acórdãos do TCU",
        author="Bruno Aires; Candice Trigueiro; Rafael Ayoroa",
        subject="PLN; TCU; TF-IDF; LegalBert-pt; LoRA",
    )

    st = build_styles()
    cnt = Counter()
    story = []

    # -----------------------------------------------------------------------
    # Cabeçalho institucional
    # -----------------------------------------------------------------------
    story.append(Paragraph(
        "Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)", st["afiliacao"]
    ))
    story.append(Paragraph(
        "Mestrado em Administração Pública — Ciência de Dados e Inteligência Artificial",
        st["afiliacao"]
    ))
    story.append(Spacer(1, 8 * mm))

    # -----------------------------------------------------------------------
    # Título
    # -----------------------------------------------------------------------
    story.append(Paragraph(
        "JURIMETRIA PREDITIVA EM ACÓRDÃOS DO TCU:<br/>"
        "COMPARAÇÃO ENTRE TF-IDF E LEGALBERT-PT EM CENÁRIO DE BAIXO RECURSO",
        st["titulo"]
    ))
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # Autores
    # -----------------------------------------------------------------------
    story.append(Paragraph("Bruno Aires &nbsp;·&nbsp; Candice Trigueiro &nbsp;·&nbsp; Rafael Ayoroa", st["autores"]))
    story.append(Paragraph(
        "Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)", st["afiliacao"]
    ))
    story.append(Paragraph("bruno.aires9@gmail.com", st["afiliacao"]))
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
    story.append(Spacer(1, 4 * mm))

    # -----------------------------------------------------------------------
    # Resumo
    # -----------------------------------------------------------------------
    story.append(Paragraph("<b>Resumo:</b>", st["abstract_label"]))
    story.append(Paragraph(
        "Este trabalho investiga a classificação automática de acórdãos do Tribunal de "
        "Contas da União (TCU) nas áreas de Saúde e Educação em três desfechos processuais: "
        "<i>Irregular</i>, <i>Regular com Ressalva</i> e <i>Regular</i>. Utilizando 4.444 "
        "acórdãos do Portal de Dados Abertos do TCU (2016–2024), foi conduzida uma ablação "
        "2×2 entre dois modelos — TF-IDF com Regressão Logística e LegalBert-pt com LoRA — "
        "e dois campos de texto — SUMARIO e VOTO. O F1-macro foi adotado como métrica "
        "principal em razão do desbalanceamento extremo do corpus (92,5% <i>Irregular</i>). "
        "O melhor resultado foi obtido pelo modelo TF-IDF + SUMARIO (F1-macro = 0,7477; "
        "IC95%: [0,687; 0,809]), superando o LegalBert-pt em todos os cenários avaliados. "
        "Identificou-se que o desempenho reduzido do Transformer decorre da combinação de "
        "dataset de pequeno porte com desbalanceamento severo de classes, limitando o sinal "
        "de gradiente para as classes minoritárias. Na otimização de <i>threshold</i> com "
        "função de custo assimétrica (FN = 10×FP), obteve-se <i>recall</i> perfeito na "
        "classe <i>Irregular</i> (1,000) com <i>threshold</i> = 0,08, configuração relevante "
        "para sistemas de alerta em controle externo.",
        st["abstract_text"]
    ))
    story.append(Paragraph(
        "<b>Palavras-chave:</b> jurimetria; controle externo; TCU; classificação de texto; "
        "BERT; LoRA; desbalanceamento de classes.",
        st["keywords"]
    ))

    story.append(Spacer(1, 3 * mm))

    # Abstract
    story.append(Paragraph("<b>Abstract:</b>", st["abstract_label"]))
    story.append(Paragraph(
        "This work investigates the automatic classification of decisions from the Brazilian "
        "Federal Court of Accounts (TCU) in the Health and Education domains into three "
        "outcome classes: <i>Irregular</i>, <i>Regular with Qualification</i>, and "
        "<i>Regular</i>. Using 4,444 rulings from the TCU Open Data Portal (2016–2024), "
        "a 2×2 ablation was conducted between two models — TF-IDF with Logistic Regression "
        "and LegalBert-pt with LoRA — and two text fields — SUMARIO and VOTO. F1-macro was "
        "adopted as the primary metric given the extreme class imbalance (92.5% "
        "<i>Irregular</i>). The best result was achieved by TF-IDF + SUMARIO (F1-macro = "
        "0.7477; 95%CI: [0.687; 0.809]), outperforming LegalBert-pt in all evaluated "
        "scenarios. The Transformer's reduced performance stems from the combination of a "
        "small dataset with severe class imbalance, limiting gradient signal for minority "
        "classes. In threshold optimization with an asymmetric cost function (FN = 10×FP), "
        "perfect recall on the <i>Irregular</i> class (1.000) was achieved with threshold = "
        "0.08, relevant for alert systems in external auditing.",
        st["abstract_text"]
    ))
    story.append(Paragraph(
        "<b>Keywords:</b> jurimetrics; external control; TCU; text classification; BERT; "
        "LoRA; class imbalance.",
        st["keywords"]
    ))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))

    # -----------------------------------------------------------------------
    # 1. INTRODUÇÃO
    # -----------------------------------------------------------------------
    story += cnt.section("Introdução", st)

    story.append(Paragraph(
        "O Tribunal de Contas da União (TCU) é o órgão de controle externo responsável "
        "pela fiscalização do uso dos recursos públicos federais brasileiros. Anualmente, "
        "o TCU profere milhares de acórdãos que julgam a regularidade das contas de "
        "gestores públicos, classificando-as como <i>Irregular</i>, <i>Regular com "
        "Ressalva</i> ou <i>Regular</i>. A identificação automática desses desfechos a "
        "partir do texto dos acórdãos constitui uma aplicação natural de técnicas de "
        "Processamento de Linguagem Natural (PLN), com potencial para apoiar sistemas de "
        "auditoria inteligente, priorização de investigações e análise jurimétrica em larga "
        "escala (TVEITA; HUSTAD, 2025).",
        st["body"]
    ))
    story.append(Paragraph(
        "A tarefa apresenta desafios específicos: (i) desbalanceamento severo de classes, "
        "com predominância massiva de acórdãos irregulares nas áreas de Saúde e Educação; "
        "(ii) textos jurídicos longos, com linguagem técnica e estrutura específica; "
        "(iii) volume limitado de dados rotulados disponíveis; e (iv) necessidade de "
        "<i>recall</i> elevado na classe de maior risco (<i>Irregular</i>), pois falsos "
        "negativos têm consequências institucionais severas.",
        st["body"]
    ))
    story.append(Paragraph(
        "Neste cenário, este trabalho propõe uma ablação controlada 2×2 — cruzando "
        "arquitetura (TF-IDF vs. LegalBert-pt) com campo de texto (SUMARIO vs. VOTO) — "
        "para isolar o efeito de cada fator sobre o F1-macro. As principais contribuições "
        "são: (i) <i>dataset</i> filtrado e rotulado de 4.444 acórdãos TCU (Saúde e "
        "Educação, 2016–2024) obtido exclusivamente de fonte oficial aberta; "
        "(ii) ablação 2×2 sistemática com validação cruzada estratificada de 5 folds e "
        "intervalos de confiança de 95%; (iii) diagnóstico técnico de falha intermitente "
        "do LegalBert-pt causada por ausência dos pesos do <i>pooler</i> no <i>checkpoint</i> "
        "original; e (iv) otimização de <i>threshold</i> com função de custo assimétrica "
        "orientada ao controle externo.",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # 2. TRABALHOS RELACIONADOS
    # -----------------------------------------------------------------------
    story += cnt.section("Trabalhos Relacionados", st)

    story += cnt.subsection("Predição de Decisões Judiciais", st)
    story.append(Paragraph(
        "Aletras et al. (2016) foram pioneiros ao aplicar PLN para prever decisões do "
        "Tribunal Europeu de Direitos Humanos com F1 de 0,79 usando SVMs sobre n-gramas. "
        "Medvedeva, Vols e Wieling (2020) expandiram essa linha com análise sistemática de "
        "variações metodológicas, demonstrando sensibilidade dos resultados à seleção de "
        "<i>features</i> e ao período temporal dos dados. No contexto brasileiro, "
        "Lage-Freitas et al. (2022) propuseram modelos de predição de decisões judiciais "
        "do Superior Tribunal de Justiça (STJ), obtendo F1-macro acima de 0,80 em cenários "
        "de classes balanceadas.",
        st["body"]
    ))

    story += cnt.subsection("Modelos de Linguagem para Português Jurídico", st)
    story.append(Paragraph(
        "Souza, Nogueira e Lotufo (2020) introduziram o BERTimbau, modelo BERT pré-treinado "
        "em corpus generalista em português, que serviu como base para especializações "
        "posteriores. Domingues (2022) disponibilizou o LegalBert-pt "
        "(<i>dominguesm/legal-bert-base-cased-ptbr</i>), pré-treinado em corpus jurídico "
        "brasileiro incluindo decisões do STF, petições e acórdãos. Sun et al. (2019) "
        "demonstraram que estratégias de truncação <i>head+tail</i> superam truncação simples "
        "em documentos jurídicos longos, preservando tanto o contexto processual quanto o "
        "dispositivo da decisão.",
        st["body"]
    ))

    story += cnt.subsection("Adaptação Eficiente de Parâmetros (PEFT)", st)
    story.append(Paragraph(
        "Hu et al. (2022) propuseram o LoRA (<i>Low-Rank Adaptation</i>), que insere "
        "matrizes de baixo <i>rank</i> nas camadas de atenção, reduzindo drasticamente o "
        "número de parâmetros treináveis sem degradação significativa de desempenho. Esta "
        "técnica é especialmente relevante em cenários com restrição computacional e "
        "<i>datasets</i> pequenos, pois reduz o risco de <i>overfitting</i>. Lin et al. "
        "(2017) propuseram a <i>Focal Loss</i> como alternativa ao <i>cross-entropy</i> "
        "ponderado para classes muito desbalanceadas.",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # 3. DADOS E METODOLOGIA
    # -----------------------------------------------------------------------
    story += cnt.section("Dados e Metodologia", st)

    story += cnt.subsection("Fonte de Dados", st)
    story.append(Paragraph(
        "Foram utilizados exclusivamente os arquivos CSV de acórdãos completos "
        "disponibilizados pelo Portal de Dados Abertos do TCU (BRASIL, 2024), cobrindo "
        "os anos de 2016 a 2024. Os arquivos são fornecidos no formato <i>pipe-delimited</i> "
        "(|), codificação UTF-8-sig, com 33 colunas em maiúsculas. O campo VOTO (coluna 29) "
        "contém o texto integral do voto do ministro relator; SUMARIO (coluna 22) contém "
        "a ementa estruturada; ACORDAO (coluna 23) contém o dispositivo da decisão.",
        st["body"]
    ))

    story += cnt.subsection("Extração de Label", st)
    story.append(Paragraph(
        "O campo SITUACAO registra o status de publicação (\"OFICIALIZADO\") para acórdãos "
        "a partir de 2020, e não o desfecho processual. Foi implementada extração hierárquica "
        "de <i>label</i>: (1) mapeamento direto de SITUACAO para acórdãos anteriores a 2020; "
        "(2) busca por expressões regulares no campo SUMARIO (e.g., "
        "<i>contas irregulares</i>); (3) busca no dispositivo ACORDAO como <i>fallback</i>. "
        "Este <i>pipeline</i> recuperou <i>labels</i> para 4.444 dos 60.000+ acórdãos nos "
        "anos estudados, após o filtro temático.",
        st["body"]
    ))

    story += cnt.subsection("Filtro Temático", st)
    story.append(Paragraph(
        "Foram mantidos apenas acórdãos relacionados às áreas de Saúde e Educação, "
        "identificados por presença dos termos [saúde, sus, fnde, merenda, educação, "
        "ministério da saúde, secretaria de saúde, secretaria de educação] nos campos "
        "SUMARIO ou ASSUNTO (operador OR, insensível a maiúsculas/minúsculas).",
        st["body"]
    ))

    story += cnt.subsection("Corpus Final", st)
    story.append(Paragraph(
        "A Tabela 1 apresenta a distribuição do corpus final por classe e ano. A classe "
        "<i>Irregular</i> representa 92,5% do corpus, <i>Regular com Ressalva</i> 4,4% e "
        "<i>Regular</i> 3,0%. Este desbalanceamento extremo motivou o uso de F1-macro como "
        "métrica principal e de pesos de classe balanceados no treinamento.",
        st["body"]
    ))

    story.append(Paragraph("Tabela 1 — Distribuição do corpus por classe e por ano.", st["table_cap"]))
    t1 = make_table(
        headers=["Ano", "Irregular", "Reg. c/ Ressalva", "Regular", "Total"],
        col_widths_mm=[18, 30, 42, 28, 22],
        rows=[
            ["2016", "286",   "10",  "12",  "308"],
            ["2017", "543",   "17",  "10",  "570"],
            ["2018", "460",   "12",  "8",   "480"],
            ["2019", "401",   "18",  "8",   "427"],
            ["2020", "533",   "19",  "12",  "564"],
            ["2021", "573",   "21",  "27",  "621"],
            ["2022", "509",   "21",  "18",  "548"],
            ["2023", "367",   "25",  "16",  "408"],
            ["2024", "440",   "54",  "24",  "518"],
            ["Total", "4.112", "197", "135", "4.444"],
        ]
    )
    story.append(t1)
    story.append(Paragraph("Fonte: elaborado pelos autores a partir dos dados abertos do TCU.", st["table_fonte"]))

    story += cnt.subsection("Pré-processamento e Divisão", st)
    story.append(Paragraph(
        "Para o modo TF-IDF: conversão para minúsculas, remoção de <i>stopwords</i> e "
        "pontuação. Para o modo BERT: preservação de caixa (modelo <i>case-sensitive</i>), "
        "normalização Unicode e remoção de quebras de linha excessivas. Textos com mais de "
        "510 <i>tokens</i> recebem truncação <i>head+tail</i>: primeiros 128 + últimos 382 "
        "<i>tokens</i> (SUN et al., 2019), capturando contexto processual e dispositivo "
        "final. O corpus foi dividido de forma estratificada em treino (70%, 3.110 "
        "acórdãos), validação (15%, 667) e teste (15%, 667), com RANDOM_STATE = 42.",
        st["body"]
    ))

    story += cnt.subsection("Modelos", st)
    story.append(Paragraph(
        "<b>TF-IDF + Regressão Logística:</b> vetorizador TF-IDF com "
        "<i>max_features</i> = 50.000, <i>ngram_range</i> = (1,2), "
        "<i>sublinear_tf</i> = True, <i>min_df</i> = 2; Regressão Logística com "
        "C = 1,0, <i>solver lbfgs</i>, <i>class_weight</i> balanceado, "
        "<i>max_iter</i> = 1.000 (PEDREGOSA et al., 2011).",
        st["body"]
    ))
    story.append(Paragraph(
        "<b>LegalBert-pt com LoRA:</b> <i>fine-tuning</i> do modelo "
        "<i>dominguesm/legal-bert-base-cased-ptbr</i> (DOMINGUES, 2022) com adaptadores "
        "LoRA (HU et al., 2022) de <i>rank</i> r = 16, <i>lora_alpha</i> = 32, "
        "<i>lora_dropout</i> = 0,1, inseridos nas matrizes de consulta e valor "
        "(<i>query</i>, <i>value</i>) de todas as camadas de atenção. Parâmetros "
        "treináveis: 1.182.723 de 126.571.782 totais (0,93%). Treinamento com "
        "<i>cross-entropy</i> ponderado normalizado, lr = 3×10⁻⁵, "
        "<i>warmup_ratio</i> = 0,10, fp16 = True, <i>early stopping</i> com paciência de "
        "2 <i>epochs</i> monitorando F1-macro.",
        st["body"]
    ))
    story.append(Paragraph(
        "<b>Correção técnica — <i>pooler</i> ausente:</b> o <i>checkpoint</i> do "
        "LegalBert-pt não inclui os pesos de <i>bert.pooler.dense</i>. A inicialização "
        "padrão do PyTorch (Kaiming uniforme) resultou em saturação do tanh do "
        "<i>pooler</i>, travando o gradiente do <i>token</i> [CLS]. Foi aplicada "
        "reinicialização explícita com distribuição Normal(0, σ²), σ = 0,02 (valor "
        "padrão da configuração BERT), e os parâmetros do <i>pooler</i> foram "
        "descongelados após a aplicação do LoRA.",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # 4. EXPERIMENTOS
    # -----------------------------------------------------------------------
    story += cnt.section("Experimentos", st)

    story += cnt.subsection("Protocolo de Avaliação", st)
    story.append(Paragraph(
        "Todos os experimentos utilizam validação cruzada estratificada de 5 <i>folds</i> "
        "sobre o conjunto de treino. São reportados F1-macro médio, desvio padrão e "
        "intervalo de confiança de 95% calculado pela distribuição t de Student com "
        "n − 1 = 4 graus de liberdade. A Tabela 2 apresenta a ablação completa 2×2.",
        st["body"]
    ))

    story.append(Paragraph(
        "Tabela 2 — Ablação 2×2: F1-macro (média ± dp, IC 95%) e acurácia. "
        "Validação cruzada estratificada de 5 folds. RANDOM_STATE = 42.",
        st["table_cap"]
    ))
    t2 = make_table(
        headers=["Modelo", "Campo", "F1-macro", "± dp", "IC 95%", "Acurácia"],
        col_widths_mm=[36, 20, 20, 14, 36, 20],
        rows=[
            ["TF-IDF + LogReg",  "SUMARIO", "0,7477", "0,0492", "[0,687; 0,809]", "0,9498"],
            ["TF-IDF + LogReg",  "VOTO",    "0,5536", "0,0256", "[0,522; 0,585]", "0,9395"],
            ["LegalBert-pt",     "VOTO",    "0,4180", "0,0426", "[0,365; 0,471]", "0,9325"],
            ["LegalBert-pt",     "SUMARIO", "0,4095", "0,0249", "[0,379; 0,440]", "0,9135"],
        ]
    )
    story.append(t2)
    story.append(Paragraph("Fonte: elaborado pelos autores.", st["table_fonte"]))

    story += cnt.subsection("Otimização de Threshold", st)
    story.append(Paragraph(
        "Para aplicação em sistemas de alerta de controle externo, foi modelada a função "
        "de custo assimétrica com falso negativo penalizado 10 vezes em relação ao falso "
        "positivo (custo de não detectar irregularidade &gt;&gt; custo de falso alarme). "
        "Realizou-se varredura no intervalo [0,05; 0,95] com passo de 0,005 sobre as "
        "probabilidades do classificador TF-IDF + SUMARIO. O <i>threshold</i> ótimo "
        "encontrado foi <b>0,08</b>, resultando em: <i>Recall</i>-Irregular = <b>1,000</b> "
        "(zero falsos negativos), <i>Precision</i>-Irregular = 0,949, "
        "F1-Irregular = 0,974.",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # 5. RESULTADOS E DISCUSSÃO
    # -----------------------------------------------------------------------
    story += cnt.section("Resultados e Discussão", st)

    story += cnt.subsection("TF-IDF supera LegalBert-pt em todos os cenários", st)
    story.append(Paragraph(
        "O modelo TF-IDF + SUMARIO atingiu F1-macro de 0,7477, superando o LegalBert-pt + "
        "VOTO em 33 pontos percentuais (0,4180) e o LegalBert-pt + SUMARIO em 35 pontos "
        "(0,4095). Este resultado contraria a expectativa padrão de superioridade dos "
        "Transformers, mas é consistente com achados na literatura em cenários de baixo "
        "recurso (MEDVEDEVA; VOLS; WIELING, 2020).",
        st["body"]
    ))

    story += cnt.subsection("Por que TF-IDF + SUMARIO é superior", st)
    story.append(Paragraph(
        "O campo SUMARIO contém ementas estruturadas com expressões formulaicas como "
        "\"CONTAS IRREGULARES\", \"CONTAS REGULARES COM RESSALVA\" e \"CONTAS REGULARES\". "
        "Bigramas TF-IDF capturam esses padrões com alta precisão por meio de "
        "correspondência lexical direta. O LegalBert-pt, ao processar as mesmas "
        "expressões, introduz custo computacional sem ganho discriminativo — a tarefa é "
        "trivial para n-gramas quando os marcadores de classe estão explicitamente "
        "presentes no texto.",
        st["body"]
    ))

    story += cnt.subsection("VOTO é mais difícil que SUMARIO para ambos os modelos", st)
    story.append(Paragraph(
        "No campo VOTO (raciocínio jurídico longo, 2.000–8.000 <i>tokens</i>), o TF-IDF "
        "cai de 0,7477 para 0,5536 (−20 p.p.): o ruído lexical dilui os marcadores "
        "discriminativos. O LegalBert-pt sobe levemente de 0,4095 (SUMARIO) para 0,4180 "
        "(VOTO), sugerindo que o modelo captura algum sinal semântico em textos longos, "
        "mas insuficiente para superar o TF-IDF.",
        st["body"]
    ))

    story += cnt.subsection("Limitações do LegalBert-pt neste cenário", st)
    story.append(Paragraph(
        "Identificaram-se três fatores que limitam o desempenho do Transformer: "
        "(i) <i>dataset</i> pequeno com desbalanceamento extremo — cada <i>fold</i> de "
        "treino contém apenas ~185 exemplos de classes minoritárias, insuficiente para o "
        "sinal de gradiente deslocar a fronteira de decisão dos adaptadores LoRA; "
        "(ii) ausência do <i>pooler</i> no <i>checkpoint</i> — a reinicialização e o "
        "descongelamento do <i>pooler</i> corrigiram o travamento do gradiente, mas não "
        "superam a limitação de dados; e (iii) acurácia enganosa — a alta acurácia de "
        "todos os modelos (91–95%) reflete a predominância da classe <i>Irregular</i> "
        "(92,5%), não o aprendizado efetivo, reforçando a escolha do F1-macro como "
        "métrica principal.",
        st["body"]
    ))

    story += cnt.subsection("Threshold ótimo para controle externo", st)
    story.append(Paragraph(
        "A otimização com custo assimétrico (FN = 10×FP) reduziu o <i>threshold</i> de "
        "decisão de 0,50 (padrão) para 0,08, zerando os falsos negativos na classe "
        "<i>Irregular</i> ao custo de um pequeno aumento em falsos positivos "
        "(<i>precision</i> = 0,949). Para um sistema de radar jurimétrico, esta "
        "configuração é preferível: investigar um processo <i>Regular</i> por equívoco tem "
        "custo administrativo baixo, enquanto deixar de detectar uma irregularidade tem "
        "impacto institucional e financeiro significativo (ALETRAS et al., 2016; "
        "RIBEIRO; SINGH; GUESTRIN, 2016).",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # 6. CONCLUSÃO
    # -----------------------------------------------------------------------
    story += cnt.section("Conclusão", st)

    story.append(Paragraph(
        "Este trabalho apresentou uma ablação controlada 2×2 para classificação automática "
        "de acórdãos do TCU nas áreas de Saúde e Educação, utilizando 4.444 acórdãos "
        "oficiais do período 2016–2024. O melhor resultado foi obtido pelo modelo TF-IDF "
        "com Regressão Logística aplicado ao campo SUMARIO (F1-macro = 0,7477; "
        "IC95%: [0,687; 0,809]), superando o LegalBert-pt com LoRA em todos os cenários.",
        st["body"]
    ))
    story.append(Paragraph(
        "O achado central — TF-IDF supera LegalBert-pt em cenário de baixo recurso com "
        "desbalanceamento extremo — tem implicação prática direta: para <i>datasets</i> "
        "jurídicos com menos de 5.000 exemplos e desbalanceamento superior a 90%, modelos "
        "baseados em n-gramas oferecem melhor relação custo-benefício que Transformers. "
        "O LegalBert-pt deve ser preferido quando houver volume suficiente de exemplos de "
        "classes minoritárias para alimentar o sinal de gradiente dos adaptadores LoRA.",
        st["body"]
    ))
    story.append(Paragraph(
        "Como trabalho futuro, destacam-se: (i) expansão do corpus para anos anteriores "
        "a 2016 e inclusão de outras áreas temáticas do TCU; (ii) comparação com "
        "BERTimbau sem LoRA (<i>full fine-tuning</i>) em configuração de <i>hold-out</i>; "
        "(iii) investigação de <i>focal loss</i> como alternativa ao "
        "<i>cross-entropy</i> ponderado (LIN et al., 2017); e (iv) integração com "
        "modelos generativos para extração automática de <i>labels</i> em acórdãos sem "
        "marcadores estruturados no SUMARIO.",
        st["body"]
    ))

    # -----------------------------------------------------------------------
    # REFERÊNCIAS
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("REFERÊNCIAS", st["ref_heading"]))

    refs = [
        ("ALETRAS, Nikolaos et al. Predicting judicial decisions of the European Court of "
         "Human Rights: A Natural Language Processing perspective. <b>PeerJ Computer "
         "Science</b>, v. 2, e93, 2016."),

        ("BRASIL. Tribunal de Contas da União. <b>Portal de Dados Abertos do TCU: "
         "Acórdãos Completos</b>. Brasília: TCU, 2024. Disponível em: "
         "&lt;https://sites.tcu.gov.br/dados-abertos/jurisprudencia/&gt;. "
         "Acesso em: 05 jun. 2026."),

        ("DEVLIN, Jacob et al. BERT: Pre-training of Deep Bidirectional Transformers for "
         "Language Understanding. In: CONFERENCE OF THE NORTH AMERICAN CHAPTER OF THE "
         "ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. "
         "<b>Proceedings...</b> Stroudsburg: ACL, 2019. p. 4171-4186."),

        ("DOMINGUES, Luciano. <b>legal-bert-base-cased-ptbr</b>: BERT model pre-trained "
         "on Brazilian legal corpus. HuggingFace Hub, 2022. Disponível em: "
         "&lt;https://huggingface.co/dominguesm/legal-bert-base-cased-ptbr&gt;. "
         "Acesso em: 05 jun. 2026."),

        ("HU, Edward J. et al. LoRA: Low-Rank Adaptation of Large Language Models. In: "
         "INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 2022, online. "
         "<b>Proceedings...</b> [S.l.]: OpenReview, 2022."),

        ("LAGE-FREITAS, André et al. Predicting Brazilian court decisions. "
         "<b>PeerJ Computer Science</b>, v. 8, e904, 2022."),

        ("LIN, Tsung-Yi et al. Focal Loss for Dense Object Detection. In: IEEE "
         "INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. "
         "<b>Proceedings...</b> [S.l.]: IEEE, 2017. p. 2980-2988."),

        ("MEDVEDEVA, Masha; VOLS, Michel; WIELING, Martijn. Using machine learning to "
         "predict decisions of the European Court of Human Rights. "
         "<b>Artificial Intelligence and Law</b>, v. 28, n. 2, p. 237-266, 2020."),

        ("PEDREGOSA, Fabian et al. Scikit-learn: machine learning in Python. "
         "<b>Journal of Machine Learning Research</b>, Cambridge, v. 12, "
         "p. 2825-2830, 2011."),

        ("RIBEIRO, Marco Tulio; SINGH, Sameer; GUESTRIN, Carlos. \"Why should I trust "
         "you?\": explaining the predictions of any classifier. In: ACM SIGKDD "
         "INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., 2016, "
         "San Francisco. <b>Proceedings...</b> New York: ACM, 2016. p. 1135-1144."),

        ("SOUZA, Fábio; NOGUEIRA, Rodrigo; LOTUFO, Roberto. BERTimbau: Pretrained BERT "
         "Models for Brazilian Portuguese. In: INTELLIGENT SYSTEMS — BRACIS, 9., 2020, "
         "Rio Grande. <b>Proceedings...</b> Cham: Springer, 2020. p. 403-417."),

        ("SUN, Chi; QIU, Xipeng; XU, Yuanbin; HUANG, Xuanjing. How to Fine-Tune BERT "
         "for Text Classification? In: CHINESE COMPUTATIONAL LINGUISTICS, 18., 2019, "
         "Kunming. <b>Proceedings...</b> Cham: Springer, 2019. p. 194-206."),

        ("TVEITA, Sondre; HUSTAD, Eli. Benefits and challenges of AI in the public "
         "sector. In: HAWAII INTERNATIONAL CONFERENCE ON SYSTEM SCIENCES, 58., 2025, "
         "Maui. <b>Proceedings...</b> Honolulu: University of Hawaii, 2025. p. 1-10."),

        ("WOLF, Thomas et al. Transformers: state-of-the-art natural language processing. "
         "In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM "
         "DEMONSTRATIONS, 2020, online. <b>Proceedings...</b> Stroudsburg: ACL, 2020. "
         "p. 38-45."),
    ]

    for ref in refs:
        story.append(Paragraph(ref, st["ref"]))

    # -----------------------------------------------------------------------
    # Gerar PDF
    # -----------------------------------------------------------------------
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    size_kb = out.stat().st_size // 1024
    print(f"PDF gerado: {out}  ({size_kb} KB, {doc.page} páginas)")
    return out


if __name__ == "__main__":
    build_pdf()
