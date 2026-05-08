# =============================================================
# treinar_modelo.py
# Passo 2: Treina o modelo de ML e salva o pipeline em .pkl
# Execução: python treinar_modelo.py
# Depende de: pacientes.csv (rode gerar_dados.py antes)
# =============================================================

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV_PATH   = "pacientes.csv"
MODEL_PATH = "modelo_risco.pkl"

# Features que o modelo usa — DEVE ser idêntico ao formulário do app.py
FEATURES = ["glicose", "pressao", "imc", "colesterol"]
TARGET   = "risco"

# --- Carrega dados ---
print("📂 Carregando dataset...")
df = pd.read_csv(CSV_PATH)

X = df[FEATURES]
y = df[TARGET]

# --- Split estratificado (mantém proporção de risco em treino e teste) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Treino : {len(X_train)} amostras")
print(f"   Teste  : {len(X_test)} amostras")

# --- Pipeline: normalização + modelo ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",   # lida com possível desbalanceamento
    )),
])

print("\n🤖 Treinando modelo...")
pipeline.fit(X_train, y_train)

# --- Avaliação ---
y_pred = pipeline.predict(X_test)

print("\n📊 Resultados da Avaliação:")
print(f"   Acurácia : {accuracy_score(y_test, y_pred):.4f}")
print("\n   Relatório completo:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Alerta"]))
print("   Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# --- Salva o pipeline inteiro (não só o modelo) ---
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Pipeline salvo em: {MODEL_PATH}")
print(f"\n▶  Próximo passo: python criar_banco.py")
