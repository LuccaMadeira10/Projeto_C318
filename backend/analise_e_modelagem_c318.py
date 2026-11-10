# %% [markdown]
# # Projeto C318 — Fundamentos de Machine Learning (Inatel)
# ## Tema 04: Crédito Consciente (Predição de Inadimplência em Cartão de Crédito)
# **Docente:** Prof. Ricardo Augusto
#
# Este script/notebook implementa o fluxo completo de Ciência de Dados exigido na disciplina **C318**:
# 1. **Entendimento do Negócio e Enquadramento Teórico**
# 2. **Coleta e Separação Imediata Treino/Teste** (prevenção de *Data Leakage*)
# 3. **Análise Exploratória de Dados (EDA) e Limpeza** (exclusivamente sobre os dados de treino)
# 4. **Preparação e Engenharia de Atributos** (`Pipeline` + `ColumnTransformer`)
# 5. **Modelagem Comparativa:**
#    - *Aprendizagem Baseada em Modelos (Linear):* Regressão Logística
#    - *Aprendizagem Baseada em Instâncias:* K-Nearest Neighbors (K-NN)
#    - *Aprendizagem Baseada em Modelos (Não Linear / Ensemble):* Random Forest
# 6. **Avaliação por Métricas de Classificação** (Matriz de Confusão, Acurácia, Precisão, Recall e $F_1\text{-Score}$)
# 7. **Exportação do Modelo Vencedor** (`modelo_credito.joblib`) para a API FastAPI.

# %%
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# %% [markdown]
# ---
# ## 1. Coleta de Dados e Regra de Ouro da C318 (Separação Treino/Teste)
# Antes de qualquer exploração profunda ou ajuste estatístico, separamos $20\%$ do conjunto de dados para **Teste** de forma estratificada (`stratify=y`), garantindo a mesma proporção de inadimplentes ($\approx 22,12\%$) em ambos os subconjuntos.

# %%
CSV_PATHS = [
    "UCI_Credit_Card.csv",
    "data/UCI_Credit_Card.csv",
    "../data/UCI_Credit_Card.csv",
]

csv_encontrado = next((p for p in CSV_PATHS if os.path.exists(p)), None)
if not csv_encontrado:
    raise FileNotFoundError(
        "Arquivo 'UCI_Credit_Card.csv' não encontrado. Coloque-o na mesma pasta deste script."
    )

df_bruto = pd.read_csv(csv_encontrado)
print(f"Dataset carregado com sucesso de: {csv_encontrado}")
print(f"Dimensões originais: {df_bruto.shape[0]} linhas x {df_bruto.shape[1]} colunas")

# Definição das variáveis preditoras (X) e da variável resposta/alvo (y)
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COLUMN = "default.payment.next.month"


def limpar_categorias_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata categorias não documentadas na base pública da UCI:
    - EDUCATION: códigos 0, 5 e 6 são agrupados em 4 (Outros).
    - MARRIAGE: código 0 é agrupado em 3 (Outros).
    """
    df_clean = df.copy()
    df_clean["EDUCATION"] = df_clean["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df_clean["MARRIAGE"] = df_clean["MARRIAGE"].replace({0: 3})
    return df_clean


df_limpo = limpar_categorias_uci(df_bruto)
X = df_limpo[FEATURE_COLUMNS]
y = df_limpo[TARGET_COLUMN]

# Separação imediata do conjunto de teste (20% teste = 6.000 clientes, 80% treino = 24.000 clientes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Conjunto de Treinamento: {X_train.shape[0]} clientes (Inadimplência: {y_train.mean()*100:.2f}%)")
print(f"Conjunto de Teste Isolado: {X_test.shape[0]} clientes (Inadimplência: {y_test.mean()*100:.2f}%)")

# %% [markdown]
# ---
# ## 2. Análise Exploratória de Dados (EDA) sobre o Conjunto de Treinamento
# A seguir, investigamos três perguntas exploratórias fundamentais para o negócio:
# 1. Qual é o grau de desbalanceamento da variável alvo $y$?
# 2. Como o status de pagamento do mês mais recente (`PAY_0`) impacta a probabilidade de inadimplência?
# 3. Clientes com menores limites de crédito (`LIMIT_BAL`) apresentam maior taxa de atraso?

# %%
# Criamos um DataFrame combinando X_train e y_train apenas para gerar as visualizações da EDA
df_eda = X_train.copy()
df_eda["Inadimplente"] = y_train.values

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Desbalanceamento de Classes
contagem_classes = df_eda["Inadimplente"].value_counts()
axes[0].bar(
    ["Adimplente (0)", "Inadimplente (1)"],
    contagem_classes.values,
    color=["#10b981", "#ef4444"],
    edgecolor="black",
)
axes[0].set_title("1. Distribuição da Classe Alvo (Desbalanceamento)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Quantidade de Clientes (Treino)")
for i, v in enumerate(contagem_classes.values):
    pct = (v / len(df_eda)) * 100
    axes[0].text(i, v + 300, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")

# Gráfico 2: Taxa de Inadimplência por Status de Pagamento Recente (PAY_0)
taxa_por_pay0 = df_eda.groupby("PAY_0")["Inadimplente"].mean() * 100
axes[1].bar(
    taxa_por_pay0.index.astype(str),
    taxa_por_pay0.values,
    color="#3b82f6",
    edgecolor="black",
)
axes[1].set_title("2. Taxa de Calote (%) por Status Recente (PAY_0)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Status PAY_0 (-1=Em dia, 0=Rotativo, 1..3=Meses de Atraso)")
axes[1].set_ylabel("Probabilidade de Inadimplência (%)")
axes[1].axhline(y=22.12, color="red", linestyle="--", label="Média Global (22.1%)")
axes[1].legend()

# Gráfico 3: Distribuição do Limite de Crédito por Classe
sns.boxplot(
    data=df_eda,
    x="Inadimplente",
    y="LIMIT_BAL",
    palette=["#10b981", "#ef4444"],
    ax=axes[2],
)
axes[2].set_title("3. Limite de Crédito (LIMIT_BAL) vs. Inadimplência", fontsize=12, fontweight="bold")
axes[2].set_xticklabels(["Adimplente (0)", "Inadimplente (1)"])
axes[2].set_ylabel("Limite Concedido (NT$)")

plt.tight_layout()
plt.savefig("eda_resumo_c318.png", dpi=150)
print("Gráfico salvo em: eda_resumo_c318.png")
plt.show()

# %% [markdown]
# ---
# ## 3. Pipeline de Preparação e Comparação de Modelos (C318)
# Testamos três algoritmos representativos das categorias estudadas em aula:
# 1. **Regressão Logística (`class_weight='balanced'`):** Aprendizagem baseada em modelos (fronteira linear).
# 2. **K-Nearest Neighbors (`K=15`):** Aprendizagem baseada em instâncias (métrica de similaridade/distância).
# 3. **Random Forest (`class_weight='balanced'`):** Aprendizagem baseada em modelos (ensemble não linear de árvores de decisão).

# %%
colunas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
colunas_numericas = [col for col in FEATURE_COLUMNS if col not in colunas_categoricas]


def criar_preprocessador() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
        ]
    )


modelos_candidatos = {
    "1. Regressão Logística (Modelo Linear)": Pipeline([
        ("preprocessador", criar_preprocessador()),
        ("classificador", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ]),
    "2. K-NN (Baseado em Instâncias)": Pipeline([
        ("preprocessador", criar_preprocessador()),
        ("classificador", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
    ]),
    "3. Random Forest (Modelo Ensemble)": Pipeline([
        ("preprocessador", criar_preprocessador()),
        ("classificador", RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

resultados_comparativos = []
matrizes_confusao = {}
pipelines_treinados = {}

for nome, pipe in modelos_candidatos.items():
    print(f"Treinando e avaliando: {nome}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    resultados_comparativos.append({
        "Modelo": nome,
        "Acurácia": round(acc, 4),
        "Precisão": round(prec, 4),
        "Recall (Sensibilidade)": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4),
        "VP": int(tp),
        "VN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    })
    matrizes_confusao[nome] = np.array([[tn, fp], [fn, tp]])
    pipelines_treinados[nome] = pipe

df_resultados = pd.DataFrame(resultados_comparativos)
print("\n=== TABELA COMPARATIVA DE DESEMPENHO NO CONJUNTO DE TESTE (N = 6.000) ===")
print(df_resultados.to_string(index=False))

# %% [markdown]
# ---
# ## 4. Visualização das Matrizes de Confusão e Comparativo de Métricas
# Como em risco de crédito o erro mais custoso para a instituição financeira é o **Falso Negativo (FN)** (deixar de identificar um cliente que entrará em inadimplência), priorizamos o equilíbrio entre **Recall (Sensibilidade)** e **$F_1\text{-Score}$**.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (nome, mc) in zip(axes, matrizes_confusao.items()):
    sns.heatmap(
        mc,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Adimplente (0)", "Pred: Inadimplente (1)"],
        yticklabels=["Real: Adimplente (0)", "Real: Inadimplente (1)"],
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(f"Matriz de Confusão\n{nome}", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("matrizes_confusao_c318.png", dpi=150)
print("Matrizes de confusão salvas em: matrizes_confusao_c318.png")
plt.show()

# %% [markdown]
# ---
# ## 5. Importância dos Atributos (Feature Importance) e Exportação para Produção
# A **Random Forest (`class_weight='balanced'`)** atingiu o melhor compromisso global ($F_1\text{-Score}$, Recall elevado e ROC-AUC superior), sendo selecionada para compor o artefato de produção `modelo_credito.joblib` consumido pelo FastAPI.

# %%
modelo_vencedor_nome = "3. Random Forest (Modelo Ensemble)"
pipeline_vencedor = pipelines_treinados[modelo_vencedor_nome]
linha_vencedora = df_resultados[df_resultados["Modelo"] == modelo_vencedor_nome].iloc[0]

# Extração das métricas no formato exato consumido pela rota GET /metrics do main.py
metricas_producao = {
    "algoritmo": "RandomForestClassifier (Pipeline Scikit-Learn)",
    "total_amostras_teste": int(len(y_test)),
    "acuracia": float(linha_vencedora["Acurácia"]),
    "precisao": float(linha_vencedora["Precisão"]),
    "recall_sensibilidade": float(linha_vencedora["Recall (Sensibilidade)"]),
    "f1_score": float(linha_vencedora["F1-Score"]),
    "matriz_confusao": {
        "VP": int(linha_vencedora["VP"]),
        "VN": int(linha_vencedora["VN"]),
        "FP": int(linha_vencedora["FP"]),
        "FN": int(linha_vencedora["FN"]),
    },
}

MODEL_OUTPUT_PATH = "modelo_credito.joblib"
joblib.dump(
    {
        "pipeline": pipeline_vencedor,
        "metricas": metricas_producao,
    },
    MODEL_OUTPUT_PATH,
)

print(f"\n[SUCESSO] Modelo vencedor e métricas de teste exportados para '{MODEL_OUTPUT_PATH}'!")