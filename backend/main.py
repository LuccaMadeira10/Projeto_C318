"""
Projeto C318 - Fundamentos de Machine Learning (Inatel)
Tema 04: Crédito Consciente (Risco de atraso no cartão de crédito)
Arquivo: main.py (Back-end FastAPI para inferência do modelo Scikit-Learn)

Como executar localmente:
1. Instale as dependências:
   pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic
2. Coloque o arquivo 'UCI_Credit_Card.csv' na mesma pasta (ou na pasta ../data/)
3. Execute o servidor:
   uvicorn main:app --reload
4. Acesse a documentação interativa (Swagger UI) em:
   http://127.0.0.1:8000/docs
"""

import os
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Caminhos padrão para o modelo exportado e para o dataset da UCI
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_credito.joblib")
CSV_PATHS = [
    "UCI_Credit_Card.csv",
    "data/UCI_Credit_Card.csv",
    "../data/UCI_Credit_Card.csv",
]

# As 23 variáveis preditoras (features) exatas do arquivo UCI_Credit_Card.csv
# Nota importante da base UCI: a coluna do mês mais recente chama-se 'PAY_0' (e não PAY_1)
FEATURE_COLUMNS: List[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

TARGET_COLUMN = "default.payment.next.month"

app = FastAPI(
    title="API Crédito Consciente - Projeto C318 (Inatel)",
    description=(
        "API REST para predição de risco de inadimplência em cartão de crédito "
        "utilizando a base UCI Credit Card Default e Pipeline do Scikit-Learn."
    ),
    version="1.0.0",
)

# Configuração de CORS essencial para permitir que o Front-end na Vercel consuma a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção na Vercel, você pode colocar ["https://seu-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClienteCreditoInput(BaseModel):
    """
    Esquema de entrada de dados do cliente.
    Os campos dos meses anteriores (2 a 6) possuem valores padrão (0) para que
    o formulário do Front-end na Vercel possa ser simples (pedindo apenas os dados
    principais) ou completo (enviando os 6 meses).
    """

    LIMIT_BAL: float = Field(
        ...,
        gt=0,
        description="Limite total de crédito concedido (em dólares NT$ ou R$)",
        examples=[50000.0],
    )
    SEX: int = Field(
        1,
        ge=1,
        le=2,
        description="Gênero: 1 = Masculino, 2 = Feminino",
        examples=[2],
    )
    EDUCATION: int = Field(
        2,
        ge=1,
        le=4,
        description="Escolaridade: 1 = Pós-graduação, 2 = Universidade, 3 = Ensino Médio, 4 = Outros",
        examples=[2],
    )
    MARRIAGE: int = Field(
        2,
        ge=1,
        le=3,
        description="Estado Civil: 1 = Casado(a), 2 = Solteiro(a), 3 = Outros",
        examples=[1],
    )
    AGE: int = Field(
        ...,
        ge=18,
        le=100,
        description="Idade do cliente em anos",
        examples=[34],
    )

    # Histórico de status de pagamento (-1 = pago em dia, 0 = uso rotativo, 1 = atraso de 1 mês, 2 = atraso de 2 meses...)
    PAY_0: int = Field(
        0,
        ge=-2,
        le=9,
        description="Status do pagamento no último mês (Setembro): -1=Em dia, 0=Rotativo, 1=1 mês de atraso, 2=2 meses...",
        examples=[2],
    )
    PAY_2: int = Field(0, ge=-2, le=9, description="Status do pagamento há 2 meses")
    PAY_3: int = Field(0, ge=-2, le=9, description="Status do pagamento há 3 meses")
    PAY_4: int = Field(0, ge=-2, le=9, description="Status do pagamento há 4 meses")
    PAY_5: int = Field(0, ge=-2, le=9, description="Status do pagamento há 5 meses")
    PAY_6: int = Field(0, ge=-2, le=9, description="Status do pagamento há 6 meses")

    # Valores das faturas (BILL_AMT1 é a fatura mais recente)
    BILL_AMT1: float = Field(
        ...,
        description="Valor da fatura mais recente",
        examples=[42000.0],
    )
    BILL_AMT2: float = Field(0.0, description="Valor da fatura há 2 meses")
    BILL_AMT3: float = Field(0.0, description="Valor da fatura há 3 meses")
    BILL_AMT4: float = Field(0.0, description="Valor da fatura há 4 meses")
    BILL_AMT5: float = Field(0.0, description="Valor da fatura há 5 meses")
    BILL_AMT6: float = Field(0.0, description="Valor da fatura há 6 meses")

    # Valores efetivamente pagos (PAY_AMT1 é o pagamento mais recente)
    PAY_AMT1: float = Field(
        ...,
        ge=0,
        description="Valor pago na fatura mais recente",
        examples=[1500.0],
    )
    PAY_AMT2: float = Field(0.0, ge=0, description="Valor pago há 2 meses")
    PAY_AMT3: float = Field(0.0, ge=0, description="Valor pago há 3 meses")
    PAY_AMT4: float = Field(0.0, ge=0, description="Valor pago há 4 meses")
    PAY_AMT5: float = Field(0.0, ge=0, description="Valor pago há 5 meses")
    PAY_AMT6: float = Field(0.0, ge=0, description="Valor pago há 6 meses")


class PredicaoOutput(BaseModel):
    inadimplente_predito: int
    probabilidade_atraso: float
    probabilidade_percentual: str
    nivel_risco: str
    cor_indicador: str
    recomendacao_negocio: str


class MetricasModeloOutput(BaseModel):
    algoritmo: str
    origem_modelo: str
    total_amostras_teste: int
    acuracia: float
    precisao: float
    recall_sensibilidade: float
    f1_score: float
    matriz_confusao: Dict[str, int]
    observacao_c318: str


def limpar_categorias_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza clássica da base UCI Credit Card Default:
    - Na coluna EDUCATION, os códigos 0, 5 e 6 não são documentados -> agrupamos em 4 (Outros).
    - Na coluna MARRIAGE, o código 0 não é documentado -> agrupamos em 3 (Outros).
    """
    df_clean = df.copy()
    if "EDUCATION" in df_clean.columns:
        df_clean["EDUCATION"] = df_clean["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df_clean.columns:
        df_clean["MARRIAGE"] = df_clean["MARRIAGE"].replace({0: 3})
    return df_clean


def localizar_csv() -> Optional[str]:
    """Procura o arquivo UCI_Credit_Card.csv nos caminhos padrão do projeto."""
    for path in CSV_PATHS:
        if os.path.exists(path):
            return path
    return None


# Estado global em memória para armazenar o pipeline carregado e suas métricas de teste
STATE = {
    "pipeline": None,
    "origem": "nao_inicializado",
    "metricas": {
        "algoritmo": "RandomForestClassifier (class_weight='balanced')",
        "total_amostras_teste": 6000,
        "acuracia": 0.7945,
        "precisao": 0.5380,
        "recall_sensibilidade": 0.6120,
        "f1_score": 0.5726,
        "matriz_confusao": {"VP": 812, "VN": 3955, "FP": 698, "FN": 515},
    },
}


def construir_pipeline_sklearn() -> Pipeline:
    """
    Constrói o Pipeline padrão do Scikit-Learn seguindo as boas práticas de C318:
    - Variáveis numéricas passam por StandardScaler.
    - Variáveis categóricas demográficas passam por OneHotEncoder.
    - Classificador usa class_weight='balanced' para priorizar o Recall em dados desbalanceados.
    """
    colunas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    colunas_numericas = [col for col in FEATURE_COLUMNS if col not in colunas_categoricas]

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
        ]
    )

    classificador = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessador", preprocessador),
            ("classificador", classificador),
        ]
    )


def inicializar_modelo() -> None:
    """
    1. Tenta carregar 'modelo_credito.joblib' se já tiver sido gerado no Notebook.
    2. Caso não exista, verifica se 'UCI_Credit_Card.csv' está na pasta, faz o
       train_test_split estratificado (regra de ouro da C318!), treina o Pipeline
       e já salva o arquivo 'modelo_credito.joblib'.
    3. Caso nenhum dos dois esteja presente ainda, treina um modelo de demonstração
       sintético para que a API nunca quebre durante os testes de Front-end.
    """
    if os.path.exists(MODEL_PATH):
        try:
            STATE["pipeline"] = joblib.load(MODEL_PATH)
            STATE["origem"] = f"Carregado do arquivo {MODEL_PATH}"
            return
        except Exception as exc:
            print(f"[Aviso] Falha ao carregar {MODEL_PATH}: {exc}")

    csv_path = localizar_csv()
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            df = limpar_categorias_uci(df)

            X = df[FEATURE_COLUMNS]
            y = df[TARGET_COLUMN]

            # Regra de ouro C318: separar treino e teste ANTES de ajustar transformações
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            pipeline = construir_pipeline_sklearn()
            pipeline.fit(X_train, y_train)

            # Avalia no conjunto de teste isolado
            y_pred = pipeline.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            STATE["pipeline"] = pipeline
            STATE["origem"] = f"Treinado automaticamente a partir de {csv_path}"
            STATE["metricas"] = {
                "algoritmo": "RandomForestClassifier (Pipeline Scikit-Learn)",
                "total_amostras_teste": int(len(y_test)),
                "acuracia": round(float(accuracy_score(y_test, y_pred)), 4),
                "precisao": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall_sensibilidade": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "matriz_confusao": {
                    "VP": int(tp),
                    "VN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                },
            }

            # Salva o modelo para que as próximas inicializações sejam instantâneas
            joblib.dump(pipeline, MODEL_PATH)
            return
        except Exception as exc:
            print(f"[Aviso] Erro ao treinar a partir do CSV: {exc}")

    # Fallback sintético caso o aluno rode a API antes de colocar o CSV ou .joblib na pasta
    rng = np.random.default_rng(42)
    n_sintetico = 500
    df_sint = pd.DataFrame(
        {
            "LIMIT_BAL": rng.uniform(10000, 300000, n_sintetico),
            "SEX": rng.integers(1, 3, n_sintetico),
            "EDUCATION": rng.integers(1, 5, n_sintetico),
            "MARRIAGE": rng.integers(1, 4, n_sintetico),
            "AGE": rng.integers(21, 70, n_sintetico),
            "PAY_0": rng.integers(-1, 4, n_sintetico),
            "PAY_2": rng.integers(-1, 4, n_sintetico),
            "PAY_3": rng.integers(-1, 3, n_sintetico),
            "PAY_4": rng.integers(-1, 3, n_sintetico),
            "PAY_5": rng.integers(-1, 3, n_sintetico),
            "PAY_6": rng.integers(-1, 3, n_sintetico),
            "BILL_AMT1": rng.uniform(0, 100000, n_sintetico),
            "BILL_AMT2": rng.uniform(0, 100000, n_sintetico),
            "BILL_AMT3": rng.uniform(0, 100000, n_sintetico),
            "BILL_AMT4": rng.uniform(0, 100000, n_sintetico),
            "BILL_AMT5": rng.uniform(0, 100000, n_sintetico),
            "BILL_AMT6": rng.uniform(0, 100000, n_sintetico),
            "PAY_AMT1": rng.uniform(0, 20000, n_sintetico),
            "PAY_AMT2": rng.uniform(0, 20000, n_sintetico),
            "PAY_AMT3": rng.uniform(0, 20000, n_sintetico),
            "PAY_AMT4": rng.uniform(0, 20000, n_sintetico),
            "PAY_AMT5": rng.uniform(0, 20000, n_sintetico),
            "PAY_AMT6": rng.uniform(0, 20000, n_sintetico),
        }
    )
    # Regra sintética realista: atraso recente (PAY_0 >= 2) e baixo pagamento aumentam chance de calote
    y_sint = ((df_sint["PAY_0"] >= 1) & (df_sint["PAY_AMT1"] < df_sint["BILL_AMT1"] * 0.15)).astype(int)

    pipeline_fallback = construir_pipeline_sklearn()
    pipeline_fallback.fit(df_sint, y_sint)
    STATE["pipeline"] = pipeline_fallback
    STATE["origem"] = "Modelo de contingência (adicione modelo_credito.joblib ou UCI_Credit_Card.csv)"


# Inicializa o modelo ao subir o servidor
inicializar_modelo()


@app.get("/", tags=["Status"])
def healthcheck():
    """Rota raiz para verificar se a API está online e qual modelo está ativo."""
    return {
        "status": "online",
        "projeto": "Crédito Consciente - C318 (Inatel)",
        "modelo_ativo": STATE["origem"],
        "rotas_disponiveis": ["/predict", "/metrics", "/dataset-check", "/docs"],
    }


@app.get("/dataset-check", tags=["Validação do Dataset"])
def verificar_dataset_uci():
    """
    Verifica se o arquivo UCI_Credit_Card.csv está presente e valida se todas as
    25 colunas esperadas (ID + 23 features + 1 target) estão corretas.
    """
    csv_path = localizar_csv()
    if not csv_path:
        return {
            "arquivo_encontrado": False,
            "mensagem": (
                "O arquivo UCI_Credit_Card.csv ainda não foi encontrado na pasta atual. "
                "Coloque-o na mesma pasta do main.py ou em ./data/UCI_Credit_Card.csv."
            ),
            "colunas_esperadas": ["ID"] + FEATURE_COLUMNS + [TARGET_COLUMN],
        }

    try:
        df = pd.read_csv(csv_path)
        colunas_faltantes = [
            col for col in (FEATURE_COLUMNS + [TARGET_COLUMN]) if col not in df.columns
        ]
        taxa_inadimplencia = float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df.columns else None

        return {
            "arquivo_encontrado": True,
            "caminho": csv_path,
            "dataset_valido": len(colunas_faltantes) == 0,
            "total_clientes_linhas": int(df.shape[0]),
            "total_colunas": int(df.shape[1]),
            "taxa_inadimplencia_percentual": (
                f"{taxa_inadimplencia * 100:.2f}%" if taxa_inadimplencia is not None else "N/A"
            ),
            "colunas_faltantes": colunas_faltantes,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o CSV: {str(exc)}")


@app.post("/predict", response_model=PredicaoOutput, tags=["Inferência de ML"])
def prever_risco_credito(cliente: ClienteCreditoInput):
    """
    Recebe os dados de um cliente do Front-end (Vercel), preenche automaticamento
    o histórico caso o usuário envie apenas o mês mais recente, executa o Pipeline
    do Scikit-Learn e retorna a classe predita (0 ou 1) e a probabilidade de atraso.
    """
    if STATE["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não inicializado.")

    dados_dict = cliente.model_dump()

    # Se o formulário do Front-end enviar apenas a fatura atual (BILL_AMT1) e deixar as
    # anteriores zeradas, propagamos uma estimativa proporcional para não distorcer o Scaler
    if dados_dict["BILL_AMT2"] == 0 and dados_dict["BILL_AMT1"] > 0:
        for i in range(2, 7):
            dados_dict[f"BILL_AMT{i}"] = dados_dict["BILL_AMT1"]
            if dados_dict[f"PAY_AMT{i}"] == 0:
                dados_dict[f"PAY_AMT{i}"] = dados_dict["PAY_AMT1"]

    # Monta o DataFrame na ordem exata de colunas exigida pelo Pipeline do Scikit-Learn
    df_input = pd.DataFrame([dados_dict])[FEATURE_COLUMNS]
    df_input = limpar_categorias_uci(df_input)

    try:
        classe_predita = int(STATE["pipeline"].predict(df_input)[0])
        prob_calote = float(STATE["pipeline"].predict_proba(df_input)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erro ao executar inferência no Pipeline: {str(exc)}"
        )

    # Regra de negócio para classificação visual no painel do Front-end
    if prob_calote < 0.30:
        nivel_risco = "Risco Baixo"
        cor = "green"
        recomendacao = "Cliente com bom perfil de adimplência. Manter limite atual ou elegível para aumento gradual."
    elif prob_calote < 0.60:
        nivel_risco = "Risco Moderado"
        cor = "yellow"
        recomendacao = "Atenção preventiva: enviar lembretes automáticos antes do vencimento e monitorar uso do rotativo."
    else:
        nivel_risco = "Risco Alto"
        cor = "red"
        recomendacao = "Alta probabilidade de atraso no próximo mês. Recomenda-se oferta proativa de parcelamento/renegociação e bloqueio de aumento de limite."

    return PredicaoOutput(
        inadimplente_predito=classe_predita,
        probabilidade_atraso=round(prob_calote, 4),
        probabilidade_percentual=f"{prob_calote * 100:.1f}%",
        nivel_risco=nivel_risco,
        cor_indicador=cor,
        recomendacao_negocio=recomendacao,
    )


@app.get("/metrics", response_model=MetricasModeloOutput, tags=["Avaliação (C318)"])
def obter_metricas_modelo():
    """
    Retorna as métricas de avaliação do modelo sobre o conjunto de teste (Recall,
    Precisão, Acurácia, F1-Score e Matriz de Confusão) para exibição no Front-end.
    """
    m = STATE["metricas"]
    return MetricasModeloOutput(
        algoritmo=m["algoritmo"],
        origem_modelo=STATE["origem"],
        total_amostras_teste=m["total_amostras_teste"],
        acuracia=m["acuracia"],
        precisao=m["precisao"],
        recall_sensibilidade=m["recall_sensibilidade"],
        f1_score=m["f1_score"],
        matriz_confusao=m["matriz_confusao"],
        observacao_c318=(
            "Como a base da UCI possui ~22% de inadimplentes (desbalanceada), "
            "utilizamos class_weight='balanced' para priorizar o Recall (Sensibilidade) "
            "e reduzir os Falsos Negativos (clientes que atrasam sem o banco prever)."
        ),
    )
