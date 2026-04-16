"""
=============================================================
SCRIPT 01 – GERADOR DE DATASET SINTÉTICO
Sistema de Predição de Risco Clínico – Documentação v1.0
=============================================================
Referência: Seções 5, 6 e 7 da Documentação Técnica
Modelo de dados: Tabela Paciente + Tabela Resultado
Saída: pacientes.csv (2000 registros)
=============================================================
"""

import numpy as np
import pandas as pd

# ── Semente para reprodutibilidade ───────────────────────
np.random.seed(42)
N = 2000

# ── Nomes fictícios simples (sem sobrenomes) ─────────────
# Conforme Seção 5: "entrada de dados: nome, idade, ..."
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
nomes = np.random.choice(nomes_masculinos + nomes_femininos, size=N)

# ── Variáveis clínicas (faixas baseadas em diretrizes) ───
# Seção 6 – Modelo de Dados: id, nome, idade, glicose,
#            pressao_arterial, imc, colesterol

idade = np.random.randint(18, 100, size=N)

glicose = np.clip(
    np.random.normal(loc=105, scale=30, size=N), 60, 300
).astype(int)

pressao_arterial = np.clip(
    np.random.normal(loc=125, scale=20, size=N), 80, 200
).astype(int)

imc = np.clip(
    np.random.normal(loc=27.0, scale=5.5, size=N), 16.0, 50.0
).round(1)

colesterol = np.clip(
    np.random.normal(loc=210, scale=40, size=N), 120, 350
).astype(int)

# ── Regra clínica de pontuação ────────────────────────────
# Seção 7: modelo treinado para classificar baixo/médio/alto
# Cada variável contribui com score 0, 1 ou 2

def score_glicose(g):
    """<100 normal | 100-125 pré-diabético | ≥126 diabético"""
    if g < 100: return 0
    elif g < 126: return 1
    else: return 2

def score_pressao(p):
    """<120 normal | 120-139 elevada | ≥140 hipertensão"""
    if p < 120: return 0
    elif p < 140: return 1
    else: return 2

def score_imc(i):
    """<25 normal | 25-29.9 sobrepeso | ≥30 obeso"""
    if i < 25: return 0
    elif i < 30: return 1
    else: return 2

def score_colesterol(c):
    """<200 desejável | 200-239 limítrofe | ≥240 alto"""
    if c < 200: return 0
    elif c < 240: return 1
    else: return 2

def score_idade(a):
    """<45 baixo | 45-64 moderado | ≥65 elevado"""
    if a < 45: return 0
    elif a < 65: return 1
    else: return 2

vf = np.vectorize
total_score = (
    vf(score_glicose)(glicose) +
    vf(score_pressao)(pressao_arterial) +
    vf(score_imc)(imc) +
    vf(score_colesterol)(colesterol) +
    vf(score_idade)(idade)
)

# Classificação final (score máx = 10)
# 0–3 → baixo (0) | 4–6 → médio (1) | 7–10 → alto (2)
risco = np.where(total_score <= 3, 0, np.where(total_score <= 6, 1, 2))

# ── Probabilidades simuladas (Tabela Resultado – Seção 6) ─
# Probabilidade de risco alto, como um modelo retornaria
prob_base = total_score / 10.0
ruido = np.random.normal(0, 0.04, size=N)
probabilidade = np.clip(prob_base + ruido, 0.01, 0.99).round(4)

# ── Monta o DataFrame final ───────────────────────────────
df = pd.DataFrame({
    "id":               range(1, N + 1),         # Tabela Paciente
    "nome":             nomes,
    "idade":            idade,
    "glicose":          glicose,
    "pressao_arterial": pressao_arterial,
    "imc":              imc,
    "colesterol":       colesterol,
    "risco":            risco,                   # Tabela Resultado
    "probabilidade":    probabilidade,
})

# ── Salva CSV ─────────────────────────────────────────────
df.to_csv("pacientes.csv", index=False, encoding="utf-8-sig")

# ── Relatório no console ──────────────────────────────────
print("=" * 58)
print("  01 – DATASET GERADO  │  pacientes.csv")
print("=" * 58)
print(f"  Registros : {len(df):,}")
print(f"  Colunas   : {list(df.columns)}")
print()
print("  Distribuição da variável alvo (risco):")
labels = {0: "Baixo (0)", 1: "Médio (1)", 2: "Alto  (2)"}
for k, v in df["risco"].value_counts().sort_index().items():
    print(f"    {labels[k]}  →  {v:4d} registros  ({v/N*100:.1f}%)")
print()
print("  Estatísticas das variáveis clínicas:")
cols = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
print(df[cols].describe().round(2).to_string())
print("=" * 58)
