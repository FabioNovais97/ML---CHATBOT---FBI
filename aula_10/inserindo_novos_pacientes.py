# ──────────────────────────────────────────────────────────────
# INSERIR DADOS DO NOVO PACIENTE AQUI
# ──────────────────────────────────────────────────────────────

# Modifique os valores abaixo para o novo paciente
novo_paciente_data = {
"nome": "Beatriz",
"idade": 26, # Idade em anos
"glicose": 100, # Glicose em mg/dL
"pressao_arterial": 200, # Pressão Arterial Sistólica em mmHg
"imc": 29.5, # Índice de Massa Corporal (kg/m²)
"colesterol": 280 # Colesterol Total em mg/dL
}

# Cria um DataFrame a partir dos dados do novo paciente
novo_paciente_df = pd.DataFrame([novo_paciente_data])

# Extrai apenas as features (mesmas usadas no treinamento)
X_novo_paciente = novo_paciente_df[FEATURES]

# Normaliza os dados do novo paciente usando o scaler JÁ TREINADO
X_novo_paciente_norm = scaler.transform(X_novo_paciente)

# Realiza a predição da classe de risco e das probabilidades
pred_classe_novo_paciente = melhor["modelo"].predict(X_novo_paciente_norm)[0]
pred_probas_novo_paciente = melhor["modelo"].predict_proba(X_novo_paciente_norm)[0]

# Mapeamento código de risco para rótulo textual
# (redefinido para garantir que esteja disponível)
rotulo = {0: "BAIXO 🟢", 1: "MÉDIO 🟡", 2: "ALTO 🔴"}

print(f"\n{'=' * 60}")
print(f" PREDIÇÃO PARA O PACIENTE: {novo_paciente_data['nome']}")
print(f"{'=' * 60}")
print(f" Risco Previsto: {rotulo[pred_classe_novo_paciente]}")
print(f" Probabilidades:")
for i, prob in enumerate(pred_probas_novo_paciente):
print(f" - {NOMES_CLASSES[i]}: {prob:.4f}")
print(f"{'=' * 60}")
