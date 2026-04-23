"""
╔══════════════════════════════════════════════════════════════╗
║        SCRIPT 02 — PIPELINE COMPLETO DE MACHINE LEARNING     ║
║        Sistema de Predição de Risco Clínico                  ║
║        Projeto Acadêmico — Aula Prática de ML                ║
╚══════════════════════════════════════════════════════════════╝

OBJETIVO:
    Executar o pipeline completo de ML — do carregamento dos
    dados até a predição de um novo paciente — passando por:

    [1] Leitura e inspeção do dataset
    [2] Separação de features (X) e target (y)
    [3] Divisão treino/teste (80/20, estratificada)
    [4] Normalização com StandardScaler
    [5] Treinamento de 3 modelos:
          • Regressão Logística
          • Random Forest
          • KNN (K-Nearest Neighbors)
    [6] Avaliação: acurácia, precisão, recall, F1-score
    [7] Validação cruzada (k-fold, k=5) para cada modelo
    [8] Comparação e escolha do melhor modelo
    [9] Visualizações: gráfico de acurácia, matriz de confusão,
        curva ROC multiclasse
   [10] Predição final: novo paciente → probabilidade + classe

BIBLIOTECAS:
    pandas      — leitura e manipulação de dados
    numpy       — operações numéricas
    scikit-learn — modelos, avaliação, pré-processamento
    matplotlib   — visualizações gráficas
"""

# ══════════════════════════════════════════════════════════════
# IMPORTAÇÕES
# ══════════════════════════════════════════════════════════════

# Manipulação de dados
import pandas as pd
import numpy as np

# Pré-processamento
from sklearn.preprocessing import StandardScaler, label_binarize

# Divisão dos dados
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Modelos de classificação
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Visualização
import matplotlib
matplotlib.use("Agg")          # backend sem display (salva em arquivo)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suprimir avisos de convergência em datasets pequenos
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════════════════════

# Semente global: garante resultados idênticos em toda execução
SEMENTE      = 42
np.random.seed(SEMENTE)

# Nomes das classes para exibição nos gráficos e relatórios
NOMES_CLASSES = ["Baixo (0)", "Médio (1)", "Alto  (2)"]

# Paleta de cores consistente para os 3 modelos
CORES_MODELOS = {
    "Logística":     "#4C72B0",   # azul
    "Random Forest": "#55A868",   # verde
    "KNN":           "#C44E52",   # vermelho
}

# Separador visual para o console
SEP = "═" * 62


# ══════════════════════════════════════════════════════════════
# ETAPA 1 — LEITURA E INSPEÇÃO DO DATASET
# ══════════════════════════════════════════════════════════════
print(SEP)
print("  ETAPA 1 — LEITURA DO DATASET")
print(SEP)

# Lê o CSV gerado pelo script 01
df = pd.read_csv("pacientes.csv")

# Informações básicas sobre o dataset
print(f"  Shape          : {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"  Colunas        : {list(df.columns)}")
print(f"  Tipos de dados :\n{df.dtypes.to_string()}")
print(f"\n  Primeiros registros:")
print(df.head(5).to_string(index=False))

# Verificação de qualidade: nulos e duplicatas
n_nulos = df.isnull().sum().sum()
n_duplic = df.duplicated().sum()
print(f"\n  Valores nulos   : {n_nulos}")
print(f"  Duplicatas      : {n_duplic}")

# Distribuição da variável alvo
print(f"\n  Distribuição do alvo (risco):")
dist = df["risco"].value_counts().sort_index()
for k, v in dist.items():
    print(f"    Classe {k} → {v:5d} registros  ({v/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# ETAPA 2 — SEPARAÇÃO DE FEATURES (X) E TARGET (y)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 2 — FEATURES E TARGET")
print(SEP)

# Features: variáveis de entrada que o modelo usará para aprender
# A coluna "nome" é excluída por ser identificadora (não preditiva)
FEATURES = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
TARGET   = "risco"

X = df[FEATURES].copy()   # matriz de features (2000 × 5)
y = df[TARGET].copy()     # vetor alvo (2000,)

print(f"  Features (X)  : {FEATURES}")
print(f"  Target   (y)  : '{TARGET}'  |  classes: {sorted(y.unique())}")
print(f"  Shape de X    : {X.shape}")
print(f"  Shape de y    : {y.shape}")


# ══════════════════════════════════════════════════════════════
# ETAPA 3 — DIVISÃO TREINO / TESTE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 3 — DIVISÃO TREINO / TESTE (80% / 20%)")
print(SEP)

# stratify=y garante que a proporção de classes seja preservada
# tanto no conjunto de treino quanto no de teste.
# Sem isso, em datasets desbalanceados poderíamos ter uma classe
# sub-representada em um dos conjuntos.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.20,       # 20% para teste → 400 amostras
    random_state=SEMENTE,
    stratify=y            # mantém proporção de classes
)

print(f"  Total    : {len(X)} amostras")
print(f"  Treino   : {len(X_treino)} amostras  ({len(X_treino)/len(X)*100:.0f}%)")
print(f"  Teste    : {len(X_teste)} amostras   ({len(X_teste)/len(X)*100:.0f}%)")

# Confirma que as proporções foram preservadas
print(f"\n  Proporção das classes no treino:")
for k, v in y_treino.value_counts().sort_index().items():
    print(f"    Classe {k}: {v:4d}  ({v/len(y_treino)*100:.1f}%)")
print(f"\n  Proporção das classes no teste:")
for k, v in y_teste.value_counts().sort_index().items():
    print(f"    Classe {k}: {v:4d}  ({v/len(y_teste)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# ETAPA 4 — NORMALIZAÇÃO COM StandardScaler
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 4 — NORMALIZAÇÃO (StandardScaler)")
print(SEP)

# CONCEITO IMPORTANTE:
# StandardScaler transforma cada feature para ter média=0 e std=1.
# Fórmula: z = (x - média) / desvio_padrão
#
# POR QUE NORMALIZAR?
#   • Algoritmos baseados em distância (KNN) são sensíveis à escala.
#     Ex: "idade" vai de 18–99 e "IMC" vai de 16–50 — sem normalização,
#     a idade dominaria o cálculo de distância.
#   • Regressão Logística converge mais rápido com dados normalizados.
#   • Random Forest é invariante à escala, mas normalizamos por
#     consistência e para usar o mesmo pipeline em todos os modelos.
#
# REGRA DE OURO: fit() apenas no treino, transform() em ambos.
# Se fizermos fit() no teste, "vazamos" informação futura para o modelo.

scaler = StandardScaler()

# fit_transform: aprende média/std do treino E já transforma
X_treino_norm = scaler.fit_transform(X_treino)

# transform: aplica a mesma escala aprendida no treino
X_teste_norm  = scaler.transform(X_teste)

print(f"  Scaler fitado nos dados de treino.")
print(f"  Médias aprendidas (por feature):")
for feat, media, std in zip(FEATURES, scaler.mean_, scaler.scale_):
    print(f"    {feat:20s}  média={media:7.2f}  std={std:6.2f}")


# ══════════════════════════════════════════════════════════════
# ETAPA 5 — DEFINIÇÃO E TREINAMENTO DOS MODELOS
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 5 — TREINAMENTO DOS MODELOS")
print(SEP)

# ──────────────────────────────────────────────────────────────
# Instanciação dos modelos com hiperparâmetros comentados
# ──────────────────────────────────────────────────────────────

# REGRESSÃO LOGÍSTICA
# Modelo linear que estima a probabilidade de cada classe usando
# a função softmax (extensão logística para multiclasse).
# max_iter=1000: número máximo de iterações do otimizador.
# C=1.0: inverso da regularização L2 (padrão). Valores menores
#        aumentam regularização e reduzem overfitting.
modelo_lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=SEMENTE
    # multiclasse tratado automaticamente via "lbfgs" (padrão)
)

# RANDOM FOREST
# Ensemble de N árvores de decisão. Cada árvore é treinada em
# uma amostra bootstrap e considera um subconjunto aleatório de
# features em cada split (bagging + feature randomness).
# n_estimators=200: número de árvores.
# class_weight="balanced": compensa desbalanceamento de classes
#   ajustando os pesos inversamente proporcional à frequência.
modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,        # árvores crescem até pureza total
    min_samples_split=5,   # mínimo de amostras para fazer split
    min_samples_leaf=2,    # mínimo de amostras em cada folha
    class_weight="balanced",
    random_state=SEMENTE,
    n_jobs=-1              # usa todos os núcleos disponíveis
)

# KNN — K-Nearest Neighbors
# Classifica um ponto com base na maioria dos k vizinhos mais
# próximos (usando distância euclidiana por padrão).
# k=7: número de vizinhos. Valores ímpares evitam empates.
# weights="distance": vizinhos mais próximos têm mais influência.
modelo_knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",    # peso inversamente proporcional à dist.
    metric="euclidean"
)

# Dicionário central: organiza modelos para iterar depois
modelos = {
    "Logística":     modelo_lr,
    "Random Forest": modelo_rf,
    "KNN":           modelo_knn,
}

# Treinamento: cada modelo aprende a fronteira de decisão
# ajustando seus parâmetros internos aos dados de treino
for nome, modelo in modelos.items():
    modelo.fit(X_treino_norm, y_treino)
    print(f"  ✔  {nome:20s} treinado.")


# ══════════════════════════════════════════════════════════════
# ETAPA 6 — AVALIAÇÃO NO CONJUNTO DE TESTE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 6 — AVALIAÇÃO NO CONJUNTO DE TESTE")
print(SEP)

# Dicionário para armazenar resultados de todos os modelos
resultados = {}

for nome, modelo in modelos.items():

    # Predição das classes
    y_pred = modelo.predict(X_teste_norm)

    # MÉTRICAS
    # acurácia : fração de predições corretas sobre o total
    # precision: de todos os que previmos como classe X,
    #            quantos realmente eram X? (evita falsos positivos)
    # recall   : de todos os que eram classe X,
    #            quantos nós capturamos? (evita falsos negativos)
    # f1-score : média harmônica entre precision e recall;
    #            útil quando as classes são desbalanceadas

    acc  = accuracy_score(y_teste, y_pred)
    prec = precision_score(y_teste, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_teste, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_teste, y_pred, average="weighted", zero_division=0)

    resultados[nome] = {
        "modelo":    modelo,
        "y_pred":    y_pred,
        "acurácia":  acc,
        "precisão":  prec,
        "recall":    rec,
        "f1-score":  f1,
    }

    # Exibição formatada por modelo
    print(f"\n  ── {nome} ─────────────────────────────────────")
    print(f"     Acurácia : {acc:.4f}   ({acc*100:.2f}%)")
    print(f"     Precisão : {prec:.4f}")
    print(f"     Recall   : {rec:.4f}")
    print(f"     F1-Score : {f1:.4f}")
    print(f"\n  Relatório completo por classe:")
    print(classification_report(
        y_teste, y_pred,
        target_names=NOMES_CLASSES,
        digits=4
    ))


# ══════════════════════════════════════════════════════════════
# ETAPA 7 — VALIDAÇÃO CRUZADA (K-FOLD, K=5)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 7 — VALIDAÇÃO CRUZADA (K-FOLD, K=5)")
print(SEP)

# CONCEITO:
# A validação cruzada divide os dados de TREINO em k partes (folds).
# Em cada rodada, k-1 partes são usadas para treinar e 1 para validar.
# Isso é repetido k vezes. A média final é uma estimativa mais robusta
# do desempenho do modelo do que uma única divisão treino/teste.
#
# StratifiedKFold: mantém a proporção das classes em cada fold.
# Essencial para datasets multiclasse ou desbalanceados.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

print(f"  {'Modelo':20s}  {'Fold1':>6} {'Fold2':>6} {'Fold3':>6} {'Fold4':>6} {'Fold5':>6}  {'Média':>7}  {'Std':>6}")
print(f"  {'-'*70}")

for nome, modelo in modelos.items():
    # cross_val_score: realiza todo o processo de CV automaticamente
    # scoring="accuracy": métrica usada em cada fold
    scores = cross_val_score(
        modelo, X_treino_norm, y_treino,
        cv=skf, scoring="accuracy"
    )
    resultados[nome]["cv_media"] = scores.mean()
    resultados[nome]["cv_std"]   = scores.std()

    linha_scores = "  ".join([f"{s:.4f}" for s in scores])
    print(
        f"  {nome:20s}  {linha_scores}  "
        f"{scores.mean():.4f}   {scores.std():.4f}"
    )


# ══════════════════════════════════════════════════════════════
# ETAPA 8 — COMPARAÇÃO E ESCOLHA DO MELHOR MODELO
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 8 — COMPARAÇÃO DOS MODELOS")
print(SEP)

# Tabela comparativa com todas as métricas
print(f"\n  {'Modelo':20s}  {'Acurácia':>9}  {'Precisão':>9}  {'Recall':>7}  {'F1':>7}  {'CV Média':>9}  {'CV Std':>7}")
print(f"  {'-'*78}")

for nome, r in resultados.items():
    print(
        f"  {nome:20s}  "
        f"{r['acurácia']:>9.4f}  "
        f"{r['precisão']:>9.4f}  "
        f"{r['recall']:>7.4f}  "
        f"{r['f1-score']:>7.4f}  "
        f"{r['cv_media']:>9.4f}  "
        f"{r['cv_std']:>7.4f}"
    )

# Seleção do melhor modelo pelo F1-score ponderado
# (mais justo que acurácia pura em datasets com classes desbalanceadas)
melhor_nome = max(resultados, key=lambda n: resultados[n]["f1-score"])
melhor      = resultados[melhor_nome]

print(f"\n  ★  MELHOR MODELO : {melhor_nome}")
print(f"     F1-Score       : {melhor['f1-score']:.4f}")
print(f"     Acurácia       : {melhor['acurácia']:.4f}  ({melhor['acurácia']*100:.2f}%)")
print(f"     CV Média       : {melhor['cv_media']:.4f}  ±  {melhor['cv_std']:.4f}")


# ══════════════════════════════════════════════════════════════
# ETAPA 9 — VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 9 — GERANDO VISUALIZAÇÕES")
print(SEP)

# Paleta neutra para as figuras
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
})

# ──────────────────────────────────────────────────────────────
# FIGURA 1 — COMPARAÇÃO DE ACURÁCIA DOS MODELOS
# ──────────────────────────────────────────────────────────────

fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(
    "Comparação de Desempenho dos Modelos\nSistema de Predição de Risco Clínico",
    fontsize=13, fontweight="bold", y=1.01
)

nomes_mod = list(resultados.keys())
cores     = [CORES_MODELOS[n] for n in nomes_mod]

# ── Gráfico 1a: Barras com as 4 métricas ──────────────────────
ax = axes[0]
metricas_nomes  = ["Acurácia", "Precisão", "Recall", "F1-Score"]
metricas_chaves = ["acurácia", "precisão", "recall",  "f1-score"]

x    = np.arange(len(metricas_nomes))
larg = 0.22    # largura de cada barra

for i, (nome_m, cor) in enumerate(zip(nomes_mod, cores)):
    vals = [resultados[nome_m][mk] for mk in metricas_chaves]
    bars = ax.bar(x + i * larg, vals, larg, label=nome_m, color=cor,
                  edgecolor="white", linewidth=0.5)
    # Valor numérico acima de cada barra
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.005,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold"
        )

ax.set_xticks(x + larg)
ax.set_xticklabels(metricas_nomes, fontsize=10)
ax.set_ylim(0.70, 1.02)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Métricas por Modelo (Teste)", fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

# ── Gráfico 1b: Boxplot da validação cruzada ──────────────────
ax2 = axes[1]

# Re-executa CV para obter scores individuais (para o boxplot)
dados_cv = []
for nome_m, modelo in modelos.items():
    scores = cross_val_score(
        modelo, X_treino_norm, y_treino,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEMENTE),
        scoring="accuracy"
    )
    dados_cv.append(scores)

bp = ax2.boxplot(
    dados_cv,
    labels=nomes_mod,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markerfacecolor="gray", markersize=5)
)
for patch, cor in zip(bp["boxes"], cores):
    patch.set_facecolor(cor)
    patch.set_alpha(0.75)

ax2.set_ylabel("Acurácia", fontsize=10)
ax2.set_title("Validação Cruzada (5-Fold)", fontsize=11, fontweight="bold")
ax2.set_ylim(0.70, 1.02)
for i, (nm, cvd) in enumerate(zip(nomes_mod, dados_cv)):
    ax2.text(i + 1, cvd.mean() + 0.005, f"μ={cvd.mean():.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("figura1_comparacao_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  figura1_comparacao_modelos.png")

# ──────────────────────────────────────────────────────────────
# FIGURA 2 — MATRIZ DE CONFUSÃO DO MELHOR MODELO
# ──────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    f"Matriz de Confusão — {melhor_nome}\nSistema de Predição de Risco Clínico",
    fontsize=13, fontweight="bold"
)

cm = confusion_matrix(y_teste, melhor["y_pred"])

# ── Contagens absolutas ────────────────────────────────────────
ax = axes2[0]
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Baixo", "Médio", "Alto"], fontsize=10)
ax.set_yticklabels(["Baixo", "Médio", "Alto"], fontsize=10)
ax.set_xlabel("Predição", fontsize=11, fontweight="bold")
ax.set_ylabel("Valor Real", fontsize=11, fontweight="bold")
ax.set_title("Contagens Absolutas", fontsize=11)

thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        cor_txt = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=16, fontweight="bold", color=cor_txt)

# ── Proporções por linha (Recall visual) ──────────────────────
ax3 = axes2[1]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im2 = ax3.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                 vmin=0, vmax=1)
plt.colorbar(im2, ax=ax3, shrink=0.8)
ax3.set_xticks([0, 1, 2]); ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(["Baixo", "Médio", "Alto"], fontsize=10)
ax3.set_yticklabels(["Baixo", "Médio", "Alto"], fontsize=10)
ax3.set_xlabel("Predição", fontsize=11, fontweight="bold")
ax3.set_ylabel("Valor Real", fontsize=11, fontweight="bold")
ax3.set_title("Proporção por Classe (Recall)", fontsize=11)

thresh2 = 0.5
for i in range(3):
    for j in range(3):
        cor_txt = "white" if cm_norm[i, j] > thresh2 else "black"
        ax3.text(j, i, f"{cm_norm[i, j]:.2f}",
                 ha="center", va="center",
                 fontsize=14, fontweight="bold", color=cor_txt)

plt.tight_layout()
plt.savefig("figura2_matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  figura2_matriz_confusao.png")

# ──────────────────────────────────────────────────────────────
# FIGURA 3 — CURVA ROC MULTICLASSE (ONE-VS-REST)
# ──────────────────────────────────────────────────────────────
# A curva ROC mostra o tradeoff entre TPR (Recall) e FPR em
# diferentes limiares de decisão.
# Para multiclasse, usamos a estratégia OvR (One-vs-Rest):
# cada classe é tratada como positiva contra todas as outras.

fig3, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.set_title(
    f"Curva ROC Multiclasse (OvR) — Todos os Modelos\n"
    f"Sistema de Predição de Risco Clínico",
    fontsize=12, fontweight="bold"
)

# Binariza as classes para cálculo OvR
classes      = [0, 1, 2]
y_teste_bin  = label_binarize(y_teste, classes=classes)  # (400, 3)

# Estilos de linha distintos para separar modelos na mesma figura
estilos_linha = {
    "Logística":     "-",
    "Random Forest": "--",
    "KNN":           "-.",
}

for nome_m, modelo in modelos.items():
    y_score = modelo.predict_proba(X_teste_norm)   # probabilidades (400, 3)
    cor     = CORES_MODELOS[nome_m]
    estilo  = estilos_linha[nome_m]

    # AUC micro-médio: trata todas as classes juntas
    fpr_micro, tpr_micro, _ = roc_curve(
        y_teste_bin.ravel(), y_score.ravel()
    )
    auc_micro = auc(fpr_micro, tpr_micro)

    ax_roc.plot(
        fpr_micro, tpr_micro,
        lw=2, color=cor, linestyle=estilo,
        label=f"{nome_m}  (AUC micro = {auc_micro:.4f})"
    )

# Linha diagonal = classificador aleatório (baseline)
ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5,
            label="Aleatório (AUC = 0.50)")

ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.02])

plt.tight_layout()
plt.savefig("figura3_curva_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  figura3_curva_roc.png")


# ══════════════════════════════════════════════════════════════
# ETAPA 10 — PREDIÇÃO FINAL DE NOVO PACIENTE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ETAPA 10 — PREDIÇÃO DE NOVOS PACIENTES")
print(SEP)

# Três perfis clínicos distintos para demonstrar os 3 cenários
novos_pacientes = pd.DataFrame([
    # Perfil 1: adulto jovem, todos os indicadores normais
    {
        "nome": "Fernanda", "idade": 28,
        "glicose": 88, "pressao_arterial": 112,
        "imc": 22.4, "colesterol": 178
    },
    # Perfil 2: meia-idade, alguns indicadores no limiar
    {
        "nome": "Marcos", "idade": 55,
        "glicose": 118, "pressao_arterial": 134,
        "imc": 28.9, "colesterol": 225
    },
    # Perfil 3: idoso, múltiplos indicadores alterados
    {
        "nome": "Helena", "idade": 74,
        "glicose": 148, "pressao_arterial": 162,
        "imc": 35.2, "colesterol": 268
    },
])

# Extrai apenas as features e aplica o MESMO scaler do treino
# (NUNCA crie um novo scaler para novos dados — use o já fitado)
X_novos      = novos_pacientes[FEATURES]
X_novos_norm = scaler.transform(X_novos)   # transforma sem re-fittar

# Usa o melhor modelo para predição
pred_classes = melhor["modelo"].predict(X_novos_norm)
pred_probas  = melhor["modelo"].predict_proba(X_novos_norm)

# Mapeamento código → rótulo textual
rotulo = {0: "BAIXO 🟢", 1: "MÉDIO 🟡", 2: "ALTO  🔴"}

print(f"\n  Modelo utilizado: {melhor_nome}\n")
print(f"  {'Paciente':<10} {'Idade':>5} {'Glicose':>8} {'PA':>5} "
      f"{'IMC':>6} {'Colest':>7}  │  {'Risco':>10}  "
      f"{'P(baixo)':>9} {'P(médio)':>9} {'P(alto)':>8}")
print(f"  {'-'*100}")

for i, row in novos_pacientes.iterrows():
    cls   = pred_classes[i]
    probs = pred_probas[i]
    print(
        f"  {row['nome']:<10} {row['idade']:>5} {row['glicose']:>8} "
        f"{row['pressao_arterial']:>5} {row['imc']:>6.1f} {row['colesterol']:>7}  │  "
        f"{rotulo[cls]:>10}  "
        f"{probs[0]:>9.4f} {probs[1]:>9.4f} {probs[2]:>8.4f}"
    )

# Exibe a interpretação de cada caso
print(f"\n  Interpretação dos casos:")
interpretacoes = {
    "Fernanda": "Todos os indicadores normais, sem fatores de risco ativos.",
    "Marcos":   "Glicemia pré-diabética, PA elevada e sobrepeso. Atenção preventiva.",
    "Helena":   "Diabetes, hipertensão estágio 2 e obesidade. Acompanhamento urgente.",
}
for nome_p, txt in interpretacoes.items():
    print(f"    • {nome_p}: {txt}")


# ══════════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  PIPELINE CONCLUÍDO COM SUCESSO")
print(SEP)
print(f"  Melhor modelo   : {melhor_nome}")
print(f"  Acurácia (teste): {melhor['acurácia']:.4f}  ({melhor['acurácia']*100:.2f}%)")
print(f"  F1-Score        : {melhor['f1-score']:.4f}")
print(f"  CV 5-fold       : {melhor['cv_media']:.4f} ± {melhor['cv_std']:.4f}")
print()
print("  Arquivos gerados:")
print("    ✔  figura1_comparacao_modelos.png")
print("    ✔  figura2_matriz_confusao.png")
print("    ✔  figura3_curva_roc.png")
print(SEP)
