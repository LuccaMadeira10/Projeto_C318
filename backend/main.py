"""
Projeto C318 - Fundamentos de Machine Learning (Inatel)
Tema 04: Crédito Consciente (Risco de atraso no cartão de crédito)
Arquivo: main.py (Back-end FastAPI para inferência do modelo Scikit-Learn)

A API apenas CARREGA o artefato modelo_credito.joblib gerado pelo script de treino
(contrato em ARTEFATO_MODELO.md). Ela não treina modelos nem inventa métricas: sem
artefato válido, /predict e /metrics retornam HTTP 503.

Como executar localmente (comandos a partir da pasta backend/):
1. Ambiente e dependências (versões fixadas):
   python -m venv ../.venv
   ../.venv/Scripts/activate        (Linux/macOS: source ../.venv/bin/activate)
   pip install -r requirements.txt              # só a API
   pip install -r requirements-dev.txt          # API + script de treino + testes
2. Treinar e gerar o artefato (requer data/UCI_Credit_Card.csv):
   python analise_e_modelagem_c318.py
3. Executar a API (os caminhos independem do diretório atual):
   uvicorn main:app --reload
   # da raiz do projeto: uvicorn main:app --app-dir backend
   # outro artefato:     defina MODEL_PATH=/caminho/modelo.joblib antes de iniciar
4. Documentação interativa (Swagger UI): http://127.0.0.1:8000/docs
5. Testes:
   python -m pytest                 # suíte completa (inclui os testes slow)
   python -m pytest -m "not slow"   # sem executar o script de treino
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("credito_consciente")

# Caminhos relativos a este arquivo, independentes do diretório de execução
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_PADRAO = BASE_DIR / "modelo_credito.joblib"
CSV_PATHS: List[Path] = [
    BASE_DIR / "UCI_Credit_Card.csv",
    BASE_DIR.parent / "data" / "UCI_Credit_Card.csv",
    BASE_DIR / "data" / "UCI_Credit_Card.csv",
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carrega o modelo no startup (e não durante o import do módulo)."""
    inicializar_modelo()
    yield


app = FastAPI(
    title="API Crédito Consciente - Projeto C318 (Inatel)",
    description=(
        "API REST para predição de risco de inadimplência em cartão de crédito "
        "utilizando a base UCI Credit Card Default e Pipeline do Scikit-Learn."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Configuração de CORS essencial para permitir que o Front-end na Vercel consuma a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção na Vercel, você pode restringir ao domínio do seu app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClienteCreditoInput(BaseModel):
    """
    Esquema de entrada de dados do cliente.
    Os campos dos meses anteriores (2 a 6) possuem valores padrão (None) para que
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
    PAY_2: Optional[int] = Field(None, ge=-2, le=9, description="Status do pagamento há 2 meses")
    PAY_3: Optional[int] = Field(None, ge=-2, le=9, description="Status do pagamento há 3 meses")
    PAY_4: Optional[int] = Field(None, ge=-2, le=9, description="Status do pagamento há 4 meses")
    PAY_5: Optional[int] = Field(None, ge=-2, le=9, description="Status do pagamento há 5 meses")
    PAY_6: Optional[int] = Field(None, ge=-2, le=9, description="Status do pagamento há 6 meses")

    # Valores das faturas (BILL_AMT1 é a fatura mais recente)
    BILL_AMT1: float = Field(
        ...,
        description="Valor da fatura mais recente",
        examples=[42000.0],
    )
    BILL_AMT2: Optional[float] = Field(None, description="Valor da fatura há 2 meses")
    BILL_AMT3: Optional[float] = Field(None, description="Valor da fatura há 3 meses")
    BILL_AMT4: Optional[float] = Field(None, description="Valor da fatura há 4 meses")
    BILL_AMT5: Optional[float] = Field(None, description="Valor da fatura há 5 meses")
    BILL_AMT6: Optional[float] = Field(None, description="Valor da fatura há 6 meses")

    # Valores efetivamente pagos (PAY_AMT1 é o pagamento mais recente)
    PAY_AMT1: float = Field(
        ...,
        ge=0,
        description="Valor pago na fatura mais recente",
        examples=[1500.0],
    )
    PAY_AMT2: Optional[float] = Field(None, ge=0, description="Valor pago há 2 meses")
    PAY_AMT3: Optional[float] = Field(None, ge=0, description="Valor pago há 3 meses")
    PAY_AMT4: Optional[float] = Field(None, ge=0, description="Valor pago há 4 meses")
    PAY_AMT5: Optional[float] = Field(None, ge=0, description="Valor pago há 5 meses")
    PAY_AMT6: Optional[float] = Field(None, ge=0, description="Valor pago há 6 meses")


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


def localizar_csv() -> Optional[Path]:
    """Procura o arquivo UCI_Credit_Card.csv nos caminhos candidatos do projeto."""
    for path in CSV_PATHS:
        if path.is_file():
            return path
    return None


def caminho_exibicao(path: Path) -> str:
    """Caminho relativo à raiz do projeto (evita expor o caminho absoluto do servidor)."""
    try:
        return path.relative_to(BASE_DIR.parent).as_posix()
    except ValueError:
        return str(path)


# Estado global em memória. Só é preenchido no startup a partir do artefato gerado
# pelo script de treino; sem artefato válido, permanece indisponível (sem fallback).
STATE: Dict[str, Any] = {
    "pipeline": None,
    "metricas": None,
    "metadados": {},
    "origem": "indisponivel",
    "erro": None,
}

CHAVES_METRICAS = [
    "algoritmo",
    "total_amostras_teste",
    "acuracia",
    "precisao",
    "recall_sensibilidade",
    "f1_score",
    "matriz_confusao",
]
CHAVES_MATRIZ = ["VP", "VN", "FP", "FN"]
LIMIAR_PADRAO = 0.5


class ArtefatoInvalidoError(Exception):
    """O artefato do modelo está ausente, ilegível ou fora do contrato esperado."""


def caminho_modelo() -> Path:
    """Caminho do artefato: variável MODEL_PATH (lida no momento da chamada) ou o padrão."""
    return Path(os.getenv("MODEL_PATH") or MODEL_PATH_PADRAO)


def carregar_modelo(caminho) -> Dict[str, Any]:
    """
    Carrega e valida o artefato gerado por analise_e_modelagem_c318.py.

    Contrato (ver ARTEFATO_MODELO.md): dicionário com "pipeline" (Pipeline com
    predict_proba e feature_names_in_ igual a FEATURE_COLUMNS), "metricas" e,
    opcionalmente, "metadados". Campos adicionais são ignorados.
    Nunca treina nem grava arquivos. Lança ArtefatoInvalidoError explicando o problema.
    """
    caminho = str(caminho)
    if not os.path.isfile(caminho):
        raise ArtefatoInvalidoError(f"arquivo do modelo não encontrado em '{caminho}'")
    if os.path.getsize(caminho) == 0:
        raise ArtefatoInvalidoError(f"arquivo do modelo '{caminho}' está vazio")
    try:
        artefato = joblib.load(caminho)
    except Exception as exc:
        raise ArtefatoInvalidoError(
            f"não foi possível ler '{caminho}' (arquivo corrompido ou gerado com outra "
            f"versão do scikit-learn): {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(artefato, dict):
        raise ArtefatoInvalidoError(
            "formato inválido: esperado um dicionário com as chaves 'pipeline' e 'metricas' "
            f"(encontrado {type(artefato).__name__})"
        )
    faltantes = [chave for chave in ("pipeline", "metricas") if chave not in artefato]
    if faltantes:
        raise ArtefatoInvalidoError(f"artefato sem as chaves obrigatórias: {faltantes}")

    pipeline = artefato["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise ArtefatoInvalidoError("o pipeline do artefato não possui predict_proba")
    colunas = list(getattr(pipeline, "feature_names_in_", []))
    if colunas != FEATURE_COLUMNS:
        raise ArtefatoInvalidoError(
            f"colunas do pipeline incompatíveis com a API: esperado {FEATURE_COLUMNS}, "
            f"encontrado {colunas}"
        )
    classes = [int(c) for c in getattr(pipeline, "classes_", [0, 1])]
    if classes != [0, 1]:
        raise ArtefatoInvalidoError(f"classes do pipeline devem ser [0, 1], encontrado {classes}")

    metricas = artefato["metricas"]
    if not isinstance(metricas, dict) or any(chave not in metricas for chave in CHAVES_METRICAS):
        raise ArtefatoInvalidoError(f"'metricas' deve conter as chaves {CHAVES_METRICAS}")
    matriz = metricas["matriz_confusao"]
    if not isinstance(matriz, dict) or any(chave not in matriz for chave in CHAVES_MATRIZ):
        raise ArtefatoInvalidoError(f"'matriz_confusao' deve conter as chaves {CHAVES_MATRIZ}")

    metadados = artefato.get("metadados") or {}
    if not isinstance(metadados, dict):
        raise ArtefatoInvalidoError("'metadados', quando presente, deve ser um dicionário")
    limiar = metadados.get("limiar_decisao", LIMIAR_PADRAO)
    if isinstance(limiar, bool) or not isinstance(limiar, (int, float)) or not 0.0 < limiar < 1.0:
        raise ArtefatoInvalidoError(f"limiar_decisao inválido: {limiar!r} (esperado entre 0 e 1)")

    return {"pipeline": pipeline, "metricas": metricas, "metadados": metadados}


def descrever_origem(caminho, metadados: Dict[str, Any]) -> str:
    """Texto de origem do modelo montado a partir dos metadados do artefato."""
    nome = os.path.basename(str(caminho))
    if not metadados:
        return f"Artefato {nome} (sem metadados de treino)"
    return (
        f"Artefato {nome} gerado pelo script de treino em "
        f"{metadados.get('treinado_em', 'data desconhecida')} "
        f"(scikit-learn {metadados.get('sklearn_version', 'versão desconhecida')})"
    )


def obter_limiar() -> float:
    """Limiar de decisão do artefato carregado (padrão 0,50)."""
    return float(STATE["metadados"].get("limiar_decisao", LIMIAR_PADRAO))


def inicializar_modelo() -> None:
    """
    Executado no startup: carrega o artefato apontado por MODEL_PATH.
    Em caso de falha a API continua no ar, mas /predict e /metrics retornam 503.
    """
    STATE.update(pipeline=None, metricas=None, metadados={}, origem="indisponivel", erro=None)
    caminho = caminho_modelo()
    try:
        artefato = carregar_modelo(caminho)
    except ArtefatoInvalidoError as exc:
        STATE["erro"] = str(exc)
        logger.error("Falha ao carregar o modelo: %s", exc)
        return
    STATE.update(artefato)
    STATE["origem"] = descrever_origem(caminho, artefato["metadados"])
    logger.info("Modelo carregado: %s", STATE["origem"])


def exigir_modelo() -> None:
    """Interrompe a requisição com HTTP 503 quando não há modelo válido carregado."""
    if STATE["pipeline"] is None or STATE["metricas"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo preditivo indisponível: {STATE['erro'] or 'modelo não carregado'}. "
                "Gere o artefato executando 'python analise_e_modelagem_c318.py' na pasta "
                "backend/ (ou defina MODEL_PATH com o caminho de um artefato válido) e "
                "reinicie a API."
            ),
        )


@app.get("/", tags=["Status"])
def healthcheck():
    """Rota raiz para verificar se a API está online e qual modelo está ativo."""
    carregado = STATE["pipeline"] is not None
    return {
        "status": "online" if carregado else "modelo_indisponivel",
        "projeto": "Crédito Consciente - C318 (Inatel)",
        "modelo_ativo": STATE["origem"],
        "modelo_carregado": carregado,
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
                "O arquivo UCI_Credit_Card.csv não foi encontrado. Locais procurados: "
                + ", ".join(caminho_exibicao(p) for p in CSV_PATHS)
                + "."
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
            "caminho": caminho_exibicao(csv_path),
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
    Recebe os dados de um cliente do Front-end (Vercel), preenche automaticamente
    o histórico caso o usuário envie apenas o mês mais recente, executa o Pipeline
    do Scikit-Learn e retorna a classe predita (0 ou 1) e a probabilidade de atraso.
    """
    exigir_modelo()

    dados_dict = cliente.model_dump()

    # Preenche de forma inteligente os meses 2 a 6 caso o Front-end envie apenas o resumo do mês 1
    status_base_antigo = 0 if dados_dict["PAY_0"] <= 0 else max(0, dados_dict["PAY_0"] - 1)
    for i in range(2, 7):
        if dados_dict[f"PAY_{i}"] is None:
            dados_dict[f"PAY_{i}"] = status_base_antigo if i <= 3 else 0
        if dados_dict[f"BILL_AMT{i}"] is None:
            dados_dict[f"BILL_AMT{i}"] = dados_dict["BILL_AMT1"]
        if dados_dict[f"PAY_AMT{i}"] is None:
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
    if prob_calote < 0.35:
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
    exigir_modelo()
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
