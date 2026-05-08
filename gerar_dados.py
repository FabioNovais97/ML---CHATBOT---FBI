# =============================================================
# gerar_dados.py
# Passo 1: Gera um dataset sintético de pacientes e salva em CSV
# Execução: python gerar_dados.py
# =============================================================

import pandas as pd
import numpy as np

SEED       = 42
N_PACIENTES = 500
ARQUIVO_CSV = "pacientes.csv"

np.random.seed(SEED)

# --- Gera indicadores de saúde ---
glicose    = np.random.randint(60,  200, N_PACIENTES)
pressao    = np.random.randint(60,  140, N_PACIENTES)
imc        = np.round(np.random.uniform(17.0, 45.0, N_PACIENTES), 1)
colesterol = np.random.randint(100, 300, N_PACIENTES)

# --- Regra de risco (qualquer indicador acima do limiar = risco 1) ---
risco = (
    (glicose    > 125) |
    (pressao    >  90) |
    (imc        >  30) |
    (colesterol > 200)
).astype(int)

# --- Nomes fictícios ---
base_nomes = [
    "Ana Lima", "Bruno Costa", "Carlos Souza", "Diana Ferreira",
    "Eduardo Alves", "Fernanda Rocha", "Gabriel Nunes", "Helena Martins",
    "Igor Pereira", "Julia Santos", "Kevin Dias", "Laura Gomes",
    "Marcos Ramos", "Natalia Cruz", "Otavio Vieira", "Patricia Lopes",
    "Rodrigo Melo", "Sabrina Torres", "Thiago Cardoso", "Ursula Barbosa"
]
nomes = [f"{base_nomes[i % len(base_nomes)]} {i + 1}" for i in range(N_PACIENTES)]

# --- Monta DataFrame ---
df = pd.DataFrame({
    "nome"      : nomes,
    "idade"     : np.random.randint(20, 80, N_PACIENTES),
    "glicose"   : glicose,
    "pressao"   : pressao,
    "imc"       : imc,
    "colesterol": colesterol,
    "risco"     : risco,
})

df.to_csv(ARQUIVO_CSV, index=False)

print(f"✅ Dataset gerado: {ARQUIVO_CSV}")
print(f"   Total de pacientes : {N_PACIENTES}")
print(f"   Distribuição de risco:")
print(f"     Normal : {(risco == 0).sum()}")
print(f"     Alerta : {(risco == 1).sum()}")
print(f"\n▶  Próximo passo: python treinar_modelo.py")
