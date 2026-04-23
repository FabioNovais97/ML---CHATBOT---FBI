"""
╔══════════════════════════════════════════════════════════════╗
║        SCRIPT 01 — GERADOR DE DATASET SINTÉTICO              ║
║        Sistema de Predição de Risco Clínico                  ║
║        Projeto Acadêmico — Aula Prática de ML                ║
╚══════════════════════════════════════════════════════════════╝

OBJETIVO:
    Criar um dataset sintético com 2.000 pacientes fictícios,
    contendo variáveis biomédicas realistas e uma variável alvo
    (risco) derivada de uma regra clínica baseada em evidências.

SAÍDA:
    pacientes.csv — entrada obrigatória para o pipeline de ML.

BIBLIOTECAS:
    numpy  — geração de dados com distribuições estatísticas
    pandas — estruturação e exportação do dataset
"""

# ──────────────────────────────────────────────────────────────
# BLOCO 1 — IMPORTAÇÕES
# ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# BLOCO 2 — CONFIGURAÇÃO INICIAL
# ──────────────────────────────────────────────────────────────

# Fixamos a semente aleatória.
# Isso garante que o dataset seja idêntico em toda execução
# (reprodutibilidade — fundamental em projetos acadêmicos).
np.random.seed(42)

# Número total de registros a gerar
N = 2000

# ──────────────────────────────────────────────────────────────
# BLOCO 3 — NOMES FICTÍCIOS SIMPLES (SEM SOBRENOME)
# ──────────────────────────────────────────────────────────────

nomes_masculinos = [
    "Carlos", "Pedro", "João", "Lucas", "Marcos", "André", "Felipe",
    "Rafael", "Bruno", "Diego", "Thiago", "Rodrigo", "Leandro", "Gustavo",
    "Henrique", "Fernando", "Eduardo", "Ricardo", "Sérgio", "Vinícius",
    "Mateus", "Gabriel", "Daniel", "Paulo", "Renato", "Fábio", "Cláudio",
    "Márcio", "Otávio", "Adriano"
]

nomes_femininos = [
    "Ana", "Maria", "Julia", "Fernanda", "Carla", "Patrícia", "Camila",
    "Beatriz", "Larissa", "Aline", "Vanessa", "Juliana", "Sandra", "Mônica",
    "Priscila", "Amanda", "Letícia", "Mariana", "Natália", "Renata",
    "Simone", "Cristina", "Débora", "Elaine", "Fabiana", "Gisele",
    "Helena", "Isabela", "Jéssica", "Karina"
]

# Sorteia aleatoriamente N nomes do pool combinado
nomes = np.random.choice(nomes_masculinos + nomes_femininos, size=N)

# ──────────────────────────────────────────────────────────────
# BLOCO 4 — GERAÇÃO DAS VARIÁVEIS BIOMÉDICAS
#
# Usamos np.random.normal(loc=média, scale=desvio) para simular
# distribuições próximas às encontradas em populações reais.
# np.clip() garante que os valores fiquem em faixas fisicamente
# possíveis (sem glicose negativa, por exemplo).
# ──────────────────────────────────────────────────────────────

# IDADE — inteiros de 18 a 99 anos (distribuição uniforme)
idade = np.random.randint(18, 100, size=N)

# GLICOSE em mg/dL
#   Normal: 70–99 | Pré-diabético: 100–125 | Diabético: ≥126
#   Média 105 reflete uma população com prevalência de pré-diabetes
glicose = np.clip(
    np.random.normal(loc=105, scale=30, size=N), 60, 300
).astype(int)

# PRESSÃO ARTERIAL SISTÓLICA em mmHg
#   Normal: <120 | Elevada: 120–139 | Hipertensão: ≥140
pressao_arterial = np.clip(
    np.random.normal(loc=125, scale=20, size=N), 80, 200
).astype(int)

# IMC — Índice de Massa Corporal (kg/m²)
#   Eutrófico: 18.5–24.9 | Sobrepeso: 25–29.9 | Obeso: ≥30
imc = np.clip(
    np.random.normal(loc=27.0, scale=5.5, size=N), 16.0, 50.0
).round(1)

# COLESTEROL TOTAL em mg/dL
#   Desejável: <200 | Limítrofe: 200–239 | Alto: ≥240
colesterol = np.clip(
    np.random.normal(loc=210, scale=40, size=N), 120, 350
).astype(int)

# ──────────────────────────────────────────────────────────────
# BLOCO 5 — REGRA DE CLASSIFICAÇÃO DO RISCO
#
# Cada variável recebe uma pontuação parcial (0, 1 ou 2)
# com base na faixa clínica em que se encontra.
# A soma dos cinco fatores gera um score de 0 a 10.
#
# Score  0–3  →  risco baixo  (classe 0)
# Score  4–6  →  risco médio  (classe 1)
# Score  7–10 →  risco alto   (classe 2)
#
# Esta regra é determinística: garante que a variável alvo
# seja coerente com os dados de entrada (evita ruído puro).
# ──────────────────────────────────────────────────────────────

def pontuar_glicose(g):
    """Score 0/1/2 para glicemia — critérios ADA."""
    if g < 100:   return 0   # normal
    elif g < 126: return 1   # pré-diabético
    else:         return 2   # diabético

def pontuar_pressao(p):
    """Score 0/1/2 para PA sistólica — diretrizes ESH."""
    if p < 120:   return 0   # ótima
    elif p < 140: return 1   # elevada/estágio 1
    else:         return 2   # hipertensão estágio 2

def pontuar_imc(i):
    """Score 0/1/2 para IMC — classificação OMS."""
    if i < 25:    return 0   # eutrófico
    elif i < 30:  return 1   # sobrepeso
    else:         return 2   # obeso

def pontuar_colesterol(c):
    """Score 0/1/2 para colesterol total — NCEP ATP III."""
    if c < 200:   return 0   # desejável
    elif c < 240: return 1   # limítrofe
    else:         return 2   # alto

def pontuar_idade(a):
    """Score 0/1/2 por faixa etária — risco cardiovascular."""
    if a < 45:    return 0   # adulto jovem
    elif a < 65:  return 1   # meia-idade
    else:         return 2   # idoso

# np.vectorize permite aplicar funções Python escalares em arrays
# NumPy sem precisar de loop explícito (mais legível e eficiente)
score_total = (
    np.vectorize(pontuar_glicose)(glicose)          +
    np.vectorize(pontuar_pressao)(pressao_arterial)  +
    np.vectorize(pontuar_imc)(imc)                   +
    np.vectorize(pontuar_colesterol)(colesterol)     +
    np.vectorize(pontuar_idade)(idade)
)

# np.where funciona como IF vetorizado:
# score ≤ 3 → 0 | score ≤ 6 → 1 | caso contrário → 2
risco = np.where(score_total <= 3, 0,
        np.where(score_total <= 6, 1, 2))

# ──────────────────────────────────────────────────────────────
# BLOCO 6 — MONTAGEM DO DATAFRAME E EXPORTAÇÃO
# ──────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "nome":             nomes,
    "idade":            idade,
    "glicose":          glicose,
    "pressao_arterial": pressao_arterial,
    "imc":              imc,
    "colesterol":       colesterol,
    "risco":            risco          # 0=baixo | 1=médio | 2=alto
})

# encoding="utf-8-sig" garante compatibilidade com Excel (BOM UTF-8)
df.to_csv("pacientes.csv", index=False, encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────
# BLOCO 7 — RELATÓRIO FINAL
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SCRIPT 01 — DATASET GERADO COM SUCESSO")
print("=" * 60)
print(f"  Arquivo    : pacientes.csv")
print(f"  Registros  : {len(df):,}")
print(f"  Colunas    : {list(df.columns)}")
print()
print("  Distribuição da variável alvo (risco):")
labels = {0: "Baixo  (0)", 1: "Médio  (1)", 2: "Alto   (2)"}
for k, v in df["risco"].value_counts().sort_index().items():
    barra = "█" * int(v / 25)
    print(f"    {labels[k]}  {v:5d}  ({v/N*100:.1f}%)  {barra}")
print()
print("  Estatísticas descritivas das variáveis clínicas:")
cols = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
print(df[cols].describe().round(2).to_string())
print("=" * 60)
print("  Execute agora: python 02_pipeline_ml.py")
print("=" * 60)
