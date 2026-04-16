"""
=============================================================
SCRIPT 02 – PIPELINE DE MACHINE LEARNING
Sistema de Predição de Risco Clínico – Documentação v1.0
=============================================================
Referência: Seção 7 – Machine Learning
  - Modelo: Random Forest Classifier
  - Etapas: Coleta → Limpeza → Treinamento (80/20) →
            Avaliação (acurácia, precisão, recall, F1) →
            Predição
Referência: Seção 5 – Especificação Funcional
  - "O sistema deve normalizar os dados antes da predição."
  - "O modelo deve retornar probabilidade e classificação."
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────
# ETAPA 1 – COLETA DE DADOS
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("  ETAPA 1 – COLETA DE DADOS")
print("=" * 60)

df = pd.read_csv("pacientes.csv")
print(f"  Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
print(f"  Colunas: {list(df.columns)}")

# ─────────────────────────────────────────────────────────
# ETAPA 2 – LIMPEZA E PREPARAÇÃO
# Seção 5: normalizar dados antes da predição
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ETAPA 2 – LIMPEZA E PREPARAÇÃO")
print("=" * 60)

# Verificação de nulos e duplicatas
nulos = df.isnull().sum().sum()
duplicatas = df.duplicated().sum()
print(f"  Valores nulos    : {nulos}")
print(f"  Registros duplicados: {duplicatas}")

# Features (X) e alvo (y)
FEATURES = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
TARGET   = "risco"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\n  Features utilizadas : {FEATURES}")
print(f"  Variável alvo       : {TARGET}")
print(f"  Classes             : {sorted(y.unique())}  (0=baixo, 1=médio, 2=alto)")

# Split estratificado 80/20 – Seção 7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Treino: {len(X_train)} amostras  |  Teste: {len(X_test)} amostras")

# ─────────────────────────────────────────────────────────
# ETAPA 3 – TREINAMENTO
# Seção 7: Random Forest Classifier
# Seção 5: normalizar dados (StandardScaler no pipeline)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ETAPA 3 – TREINAMENTO (Random Forest)")
print("=" * 60)

pipeline = Pipeline([
    ("scaler", StandardScaler()),          # Normalização – Seção 5
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",           # lida com desbalanceamento
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
print("  Modelo treinado com sucesso.")
print(f"  Hiperparâmetros RF: n_estimators=200, class_weight=balanced")

# Validação cruzada estratificada (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train,
                            cv=skf, scoring="accuracy")
print(f"\n  Cross-validation (5-fold) – Acurácia:")
print(f"    Scores : {cv_scores.round(4)}")
print(f"    Média  : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────
# ETAPA 4 – AVALIAÇÃO
# Seção 7: acurácia, precisão, recall, F1-score
# Seção 6/Resultado: probabilidade + classificação
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ETAPA 4 – AVALIAÇÃO NO CONJUNTO DE TESTE")
print("=" * 60)

y_pred      = pipeline.predict(X_test)
y_prob      = pipeline.predict_proba(X_test)          # probabilidades

acuracia    = accuracy_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"\n  Acurácia geral : {acuracia:.4f}  ({acuracia*100:.2f}%)")
print(f"  AUC-ROC (macro): {auc:.4f}")

print("\n  Relatório por classe (precisão, recall, F1-score):")
print("  " + "-" * 54)
target_names = ["Baixo (0)", "Médio (1)", "Alto  (2)"]
print(classification_report(y_test, y_pred,
                             target_names=target_names,
                             digits=4))

print("  Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':12}  Pred→Baixo  Pred→Médio  Pred→Alto")
for i, (row, label) in enumerate(zip(cm, ["Real→Baixo", "Real→Médio", "Real→Alto"])):
    print(f"  {label:12}  {row[0]:10d}  {row[1]:10d}  {row[2]:10d}")

# Importância das features
feat_imp = pd.Series(
    pipeline.named_steps["clf"].feature_importances_,
    index=FEATURES
).sort_values(ascending=False)
print("\n  Importância das Features (Random Forest):")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:20s} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────
# ETAPA 5 – PREDIÇÃO
# Seção 5: retornar probabilidade + classificação
# Seção 6: salvar resultado (risco + probabilidade)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ETAPA 5 – PREDIÇÃO DE NOVOS PACIENTES")
print("=" * 60)

# Exemplos de 3 perfis clínicos distintos
novos_pacientes = pd.DataFrame({
    "nome":             ["Fernanda", "Marcos", "Helena"],
    "idade":            [28,          62,        80],
    "glicose":          [85,         118,       145],
    "pressao_arterial": [110,        135,       160],
    "imc":              [22.1,       28.5,      34.8],
    "colesterol":       [175,        215,       270],
})

X_novos = novos_pacientes[FEATURES]
pred_classe = pipeline.predict(X_novos)
pred_prob   = pipeline.predict_proba(X_novos)

label_map = {0: "Baixo", 1: "Médio", 2: "Alto"}
print(f"\n  {'Nome':12} {'Idade':>5} {'Glicose':>8} {'PA':>4} {'IMC':>5} {'Colest':>7}  {'Risco':>6}  Probabilidades")
print("  " + "-" * 76)
for i, row in novos_pacientes.iterrows():
    classe = pred_classe[i]
    probs  = pred_prob[i]
    print(
        f"  {row['nome']:12} {row['idade']:>5} {row['glicose']:>8} "
        f"{row['pressao_arterial']:>4} {row['imc']:>5} {row['colesterol']:>7}  "
        f"{label_map[classe]:>6}  "
        f"[Baixo={probs[0]:.2f} Médio={probs[1]:.2f} Alto={probs[2]:.2f}]"
    )

# ─────────────────────────────────────────────────────────
# SALVAR MODELO (para integração com Backend – Seção 2)
# ─────────────────────────────────────────────────────────
joblib.dump(pipeline, "modelo_risco_clinico.pkl")
print("\n" + "=" * 60)
print("  Modelo salvo em: modelo_risco_clinico.pkl")
print("  Pronto para integração com a API (Seção 2 – Backend)")
print("=" * 60)
