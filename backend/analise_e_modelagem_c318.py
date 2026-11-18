# %% [markdown]
# # Projeto C318 — Fundamentos de Machine Learning (Inatel)
# ## Tema 04: Crédito Consciente (Predição de Inadimplência em Cartão de Crédito)
# **Docente:** Prof. Ricardo Augusto
#
# Este script/notebook implementa o fluxo completo de Ciência de Dados exigido na disciplina **C318**:
# 1. **Coleta e separação imediata Treino/Teste** (o teste fica guardado até a avaliação final)
# 2. **Análise Exploratória de Dados (EDA)** (exclusivamente sobre os dados de treino)
# 3. **Comparação justa de modelos por validação cruzada** (somente no treino)
# 4. **Exportação do modelo** (`modelo_credito.joblib`) para a API FastAPI.
#
# **Regra metodológica central:** nenhuma informação do conjunto de teste influencia a escolha
# do modelo. O teste é usado uma única vez, no final.

# %%
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# Configuração de execução (reprodutível: semente fixa em tudo)
# ---------------------------------------------------------------------------
SEMENTE = 42
# C318_MODO_RAPIDO=1: grades mínimas e 3 folds (usado pelos testes automatizados)
MODO_RAPIDO = os.getenv("C318_MODO_RAPIDO") == "1"
N_FOLDS = 3 if MODO_RAPIDO else 5
N_JOBS = int(os.getenv("C318_N_JOBS", "-1"))


def _localizar_backend() -> Path:
    """Pasta backend/: a deste arquivo; em notebook (sem __file__), procura a partir da pasta atual."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    for pasta in [Path.cwd(), *Path.cwd().parents]:
        if (pasta / "analise_e_modelagem_c318.py").is_file():
            return pasta
        if (pasta / "backend" / "analise_e_modelagem_c318.py").is_file():
            return pasta / "backend"
    return Path.cwd()


BASE_DIR = _localizar_backend()
RAIZ_REPO = BASE_DIR.parent
# Destino das tabelas e gráficos (configurável por C318_RESULTADOS_DIR)
RESULTADOS_DIR = Path(os.getenv("C318_RESULTADOS_DIR") or RAIZ_REPO / "resultados")
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)


def exibir_caminho(caminho: Path) -> str:
    """Caminho relativo à raiz do projeto quando possível (não expõe a pasta do usuário)."""
    try:
        return Path(caminho).resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return str(caminho)


def salvar_tabela(df: pd.DataFrame, nome: str) -> None:
    caminho = RESULTADOS_DIR / nome
    df.to_csv(caminho, index=False, encoding="utf-8")
    print(f"Tabela salva em: {exibir_caminho(caminho)}")


def salvar_figura(fig, nome: str) -> None:
    caminho = RESULTADOS_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo em: {exibir_caminho(caminho)}")
    plt.show()
    plt.close(fig)


# %% [markdown]
# ---
# ## 1. Coleta de Dados e Regra de Ouro da C318 (Separação Treino/Teste)
# Antes de qualquer exploração ou ajuste estatístico, separamos $20\%$ do conjunto de dados para
# **Teste** de forma estratificada (`stratify=y`, `random_state=42`), garantindo a mesma proporção
# de inadimplentes ($\approx 22\%$) nos dois subconjuntos. Esses 6.000 clientes só serão usados na
# avaliação final.
#
# A limpeza de categorias abaixo é uma regra fixa linha a linha (não "aprende" nada dos dados),
# por isso pode ser aplicada antes da divisão sem vazamento.

# %%
CSV_PATHS = [
    BASE_DIR / "UCI_Credit_Card.csv",
    RAIZ_REPO / "data" / "UCI_Credit_Card.csv",
    BASE_DIR / "data" / "UCI_Credit_Card.csv",
]

csv_encontrado = next((p for p in CSV_PATHS if p.exists()), None)
if not csv_encontrado:
    raise FileNotFoundError(
        "Arquivo 'UCI_Credit_Card.csv' não encontrado. Locais procurados: "
        + ", ".join(str(p) for p in CSV_PATHS)
    )

df_bruto = pd.read_csv(csv_encontrado)
print(f"Dataset carregado de: {exibir_caminho(csv_encontrado)}")
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
    X, y, test_size=0.20, random_state=SEMENTE, stratify=y
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
taxa_media_treino = df_eda["Inadimplente"].mean() * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Desbalanceamento de Classes
contagem_classes = df_eda["Inadimplente"].value_counts().sort_index()
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
axes[1].set_xlabel("Status PAY_0 (-1=Em dia, 0=Rotativo, 1..8=Meses de Atraso)")
axes[1].set_ylabel("Probabilidade de Inadimplência (%)")
axes[1].axhline(y=taxa_media_treino, color="red", linestyle="--",
                label=f"Média no treino ({taxa_media_treino:.1f}%)")
axes[1].legend()

# Gráfico 3: Distribuição do Limite de Crédito por Classe
sns.boxplot(
    data=df_eda,
    x="Inadimplente",
    y="LIMIT_BAL",
    hue="Inadimplente",
    palette=["#10b981", "#ef4444"],
    legend=False,
    ax=axes[2],
)
axes[2].set_title("3. Limite de Crédito (LIMIT_BAL) vs. Inadimplência", fontsize=12, fontweight="bold")
axes[2].set_xticks([0, 1], ["Adimplente (0)", "Inadimplente (1)"])
axes[2].set_ylabel("Limite Concedido (NT$)")

plt.tight_layout()
salvar_figura(fig, "eda_resumo.png")

# %% [markdown]
# ---
# ## 3. Comparação justa de modelos por validação cruzada (somente treino)
# Todos os modelos usam **o mesmo pré-processamento dentro de um `Pipeline`** (padronização das
# variáveis numéricas e *one-hot* das categóricas). Como o `Pipeline` inteiro é ajustado dentro de
# cada *fold*, o `StandardScaler` e o `OneHotEncoder` nunca veem os dados de validação daquele fold
# (sem vazamento).
#
# Modelos comparados:
# 1. **Chute (referência):** `DummyClassifier(strategy="most_frequent")`, sempre prevê "vai pagar".
#    Qualquer modelo útil precisa superá-lo.
# 2. **Regressão Logística:** aprendizagem baseada em modelos (fronteira linear).
# 3. **K-NN:** aprendizagem baseada em instâncias (similaridade por distância).
# 4. **Árvore de Decisão:** modelo não linear interpretável (regras "se... então").
# 5. **Random Forest:** *ensemble* de árvores de decisão.
#
# Todos os que aceitam `class_weight` usam `"balanced"` (compensa os ~22% de inadimplentes).
#
# **Limitação do KNN:** o `KNeighborsClassifier` não aceita `class_weight`, então ele é o único sem
# compensação do desbalanceamento. Isso afeta principalmente as métricas que dependem do corte de
# 0,50 (recall, precisão e F1). Por isso o **critério principal de comparação é a ROC-AUC**, que
# avalia a ordenação dos clientes por risco em todos os cortes possíveis e não depende do corte
# nem do balanceamento das classes. Recall, precisão, F1 (classe 1) e PR-AUC são reportados como
# apoio.
#
# Validação: `StratifiedKFold(5, shuffle=True, random_state=42)` sobre `X_train`/`y_train`
# (3 folds no modo rápido).

# %%
COLUNAS_CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE"]
COLUNAS_NUMERICAS = [col for col in FEATURE_COLUMNS if col not in COLUNAS_CATEGORICAS]

CHUTE = "Chute (referência)"
LOGISTICA = "Regressão Logística"
KNN = "KNN"
ARVORE = "Árvore de Decisão"
FLORESTA = "Random Forest"
MODELOS = [CHUTE, LOGISTICA, KNN, ARVORE, FLORESTA]

CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
METRICAS_CV = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": make_scorer(recall_score, zero_division=0),
    "precisao": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def criar_preprocessador() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        ]
    )


def criar_classificador(nome: str):
    """Configuração inicial de cada modelo (antes do GridSearchCV)."""
    if nome == CHUTE:
        return DummyClassifier(strategy="most_frequent")
    if nome == LOGISTICA:
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEMENTE)
    if nome == KNN:
        # Não aceita class_weight (limitação registrada acima)
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if nome == ARVORE:
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEMENTE)
    if nome == FLORESTA:
        return RandomForestClassifier(
            n_estimators=120, max_depth=10, class_weight="balanced", random_state=SEMENTE
        )
    raise ValueError(f"modelo desconhecido: {nome}")


def criar_pipeline(nome: str) -> Pipeline:
    return Pipeline([
        ("preprocessador", criar_preprocessador()),
        ("classificador", criar_classificador(nome)),
    ])


def resumir_cv(nome: str, resultado: dict) -> dict:
    """Média e desvio-padrão (amostral, entre folds) de cada métrica de cross_validate."""
    linha = {"modelo": nome}
    for metrica in METRICAS_CV:
        valores = np.asarray(resultado[f"test_{metrica}"])
        linha[f"{metrica}_media"] = valores.mean()
        linha[f"{metrica}_dp"] = valores.std(ddof=1)
    if "train_roc_auc" in resultado:
        linha["roc_auc_treino_media"] = np.mean(resultado["train_roc_auc"])
    linha["tempo_ajuste_s"] = np.mean(resultado["fit_time"])
    linha["tempo_predicao_s"] = np.mean(resultado["score_time"])
    return linha


# T1: configuração inicial, mesmo pré-processamento e mesma CV para todos (somente treino)
resultados_cv_inicial = {}
for nome in MODELOS:
    print(f"Validação cruzada (configuração inicial): {nome}...")
    resultados_cv_inicial[nome] = cross_validate(
        criar_pipeline(nome), X_train, y_train, cv=CV, scoring=METRICAS_CV,
        n_jobs=N_JOBS, return_train_score=True, error_score="raise",
    )

df_cv_inicial = pd.DataFrame([resumir_cv(n, r) for n, r in resultados_cv_inicial.items()])
print(f"\n=== VALIDAÇÃO CRUZADA NO TREINO ({N_FOLDS} folds), CONFIGURAÇÃO INICIAL ===")
print(df_cv_inicial.round(4).to_string(index=False))
salvar_tabela(df_cv_inicial.round(6), "comparacao_modelos_cv_inicial.csv")

# %% [markdown]
# ---
# ## 4. Escolha do modelo (pela validação cruzada) e avaliação final no teste
# O vencedor é o modelo de maior ROC-AUC média na validação cruzada. Só depois dessa escolha o
# conjunto de teste é usado, uma única vez.

# %%
modelo_vencedor_nome = df_cv_inicial.sort_values("roc_auc_media", ascending=False).iloc[0]["modelo"]
print(f"Modelo vencedor na CV: {modelo_vencedor_nome}")

pipeline_vencedor = clone(criar_pipeline(modelo_vencedor_nome)).fit(X_train, y_train)
LIMIAR = 0.5
proba_teste = pipeline_vencedor.predict_proba(X_test)[:, 1]
y_pred = (proba_teste >= LIMIAR).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

metricas_producao = {
    "algoritmo": f"{type(pipeline_vencedor[-1]).__name__} (Pipeline Scikit-Learn)",
    "total_amostras_teste": int(len(y_test)),
    "acuracia": round(float(accuracy_score(y_test, y_pred)), 4),
    "precisao": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    "recall_sensibilidade": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, proba_teste)), 4),
    "matriz_confusao": {"VP": int(tp), "VN": int(tn), "FP": int(fp), "FN": int(fn)},
}
print(metricas_producao)

# Destino do artefato: backend/modelo_credito.joblib, ou a variável MODEL_OUTPUT_PATH.
# Contrato consumido pela API documentado em ARTEFATO_MODELO.md.
MODEL_OUTPUT_PATH = Path(os.getenv("MODEL_OUTPUT_PATH") or BASE_DIR / "modelo_credito.joblib")
csv_origem = exibir_caminho(csv_encontrado)

metadados_artefato = {
    "sklearn_version": sklearn.__version__,
    "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "csv_origem": csv_origem,
    "limiar_decisao": LIMIAR,
}

joblib.dump(
    {
        "pipeline": pipeline_vencedor,
        "metricas": metricas_producao,
        "metadados": metadados_artefato,
    },
    MODEL_OUTPUT_PATH,
)

print(f"\n[SUCESSO] Modelo vencedor e métricas de teste exportados para '{exibir_caminho(MODEL_OUTPUT_PATH)}'!")
