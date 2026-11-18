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

import csv
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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

# Resultados do script de treino (Passo 2) usados por GET /comparacao-modelos.
# Pasta padrão: <raiz do projeto>/resultados; sobrescrevível por C318_RESULTADOS_DIR.
RESULTADOS_DIR_PADRAO = BASE_DIR.parent / "resultados"
ARQUIVO_COMPARACAO_CV = "comparacao_modelos_cv.csv"
ARQUIVO_HIPERPARAMETROS = "melhores_hiperparametros.json"
COLUNAS_METRICAS_CV = [
    "roc_auc_media", "roc_auc_dp",
    "pr_auc_media", "pr_auc_dp",
    "recall_media", "recall_dp",
    "precisao_media", "precisao_dp",
    "f1_media", "f1_dp",
]

# Histórico dos meses 2 a 6: opcional no payload (preenchido automaticamente se ausente)
CAMPOS_OPCIONAIS: List[str] = (
    [f"PAY_{i}" for i in range(2, 7)]
    + [f"BILL_AMT{i}" for i in range(2, 7)]
    + [f"PAY_AMT{i}" for i in range(2, 7)]
)


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


AVISO_ESCORE = (
    "Escore do modelo entre 0 e 1, não necessariamente uma probabilidade calibrada. "
    "Modelos treinados com class_weight='balanced' tendem a superestimar a chance real de "
    "inadimplência, cuja frequência observada na base é de aproximadamente 22%. Use o valor "
    "para ordenar clientes por risco, não como percentual literal de chance."
)


class PredicaoOutput(BaseModel):
    inadimplente_predito: int = Field(
        ..., description="Classe prevista: 1 quando o escore é maior ou igual ao limiar de decisão do artefato."
    )
    probabilidade_atraso: float = Field(
        ..., description=AVISO_ESCORE + " Truncado para baixo em 4 casas decimais."
    )
    probabilidade_percentual: str = Field(
        ...,
        description=(
            "Escore em percentual, truncado para baixo em 1 casa decimal (mesma ressalva de "
            "calibração)."
        ),
    )
    nivel_risco: str
    cor_indicador: str
    recomendacao_negocio: str
    campos_preenchidos_automaticamente: List[str] = Field(
        default_factory=list,
        description=(
            "Campos dos meses 2 a 6 que não foram enviados e foram estimados pela API a "
            "partir do mês mais recente. Lista vazia quando todos foram informados."
        ),
    )
    aviso: Optional[str] = Field(
        None,
        description="Presente somente quando houve preenchimento automático de campos.",
    )


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
    # Campos opcionais lidos do artefato: null quando o artefato não os traz (nunca inventados)
    roc_auc: Optional[float] = Field(None, description="ROC-AUC no conjunto de teste.")
    pr_auc: Optional[float] = Field(None, description="PR-AUC (precisão média) no conjunto de teste.")
    roc_auc_chute: Optional[float] = Field(
        None, description="ROC-AUC do modelo de referência (chute) no mesmo teste."
    )
    roc_auc_cv_treino: Optional[float] = Field(
        None, description="ROC-AUC média do modelo final na validação cruzada do treino."
    )
    modelo: Optional[str] = Field(None, description="Nome do modelo vencedor (metadados.modelo).")
    hiperparametros: Optional[Dict[str, Any]] = Field(
        None, description="Hiperparâmetros do modelo final (metadados.hiperparametros)."
    )
    limiar_decisao: Optional[float] = Field(
        None,
        description=(
            "Corte de decisão gravado no artefato (metadados.limiar_decisao). null quando o "
            "artefato não o traz; nesse caso a API usa o padrão de 0,50."
        ),
    )
    atributos_sensiveis_usados: Optional[bool] = Field(
        None, description="false: SEX, AGE e MARRIAGE são descartados pelo pipeline."
    )
    treinado_em: Optional[str] = Field(None, description="Data/hora do treino (ISO 8601).")


class MetricaCVOutput(BaseModel):
    """Uma linha de resultados/comparacao_modelos_cv.csv (validação cruzada, só treino)."""

    modelo: str
    roc_auc_media: float
    roc_auc_dp: float
    pr_auc_media: float
    pr_auc_dp: float
    recall_media: float
    recall_dp: float
    precisao_media: float
    precisao_dp: float
    f1_media: float
    f1_dp: float
    vencedor: bool
    melhores_parametros: Optional[Dict[str, Any]] = Field(
        None, description="Melhores hiperparâmetros do GridSearch (null para o Chute)."
    )
    grade: Optional[Dict[str, List[Any]]] = Field(
        None, description="Valores testados no GridSearch (null para o Chute)."
    )


class ComparacaoModelosOutput(BaseModel):
    criterio: Optional[str] = Field(None, description="Métrica usada para escolher o vencedor.")
    n_folds: Optional[int] = Field(None, description="Número de folds da validação cruzada.")
    modelos: List[MetricaCVOutput]
    vencedor: str = Field(..., description="Modelo marcado como vencedor no CSV.")
    final: Dict[str, Any] = Field(
        ..., description="Modelo de produção e seus parâmetros (bloco 'final' do JSON)."
    )
    modelo_carregado: Optional[str] = Field(
        None, description="metadados.modelo do artefato carregado na API (null sem modelo)."
    )
    coerente_com_modelo_carregado: bool = Field(
        ...,
        description=(
            "true quando o vencedor do CSV, final.modelo do JSON e metadados.modelo do "
            "artefato carregado são o mesmo modelo."
        ),
    )


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
# Campos informativos do artefato expostos em /metrics (null quando ausentes)
CHAVES_METRICAS_OPCIONAIS = ["roc_auc", "pr_auc", "roc_auc_chute", "roc_auc_cv_treino"]
CHAVES_METADADOS_EXPOSTOS = [
    "modelo",
    "hiperparametros",
    "limiar_decisao",
    "atributos_sensiveis_usados",
    "treinado_em",
]
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


def descrever_observacao() -> str:
    """
    Observação de /metrics montada a partir do modelo carregado (sem hiperparâmetros
    escritos à mão): menciona class_weight='balanced' somente se o classificador o usa.
    """
    pipeline = STATE["pipeline"]
    classificador = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    balanceado = getattr(classificador, "class_weight", None) == "balanced"
    texto = (
        "Métricas calculadas pelo script de treino no conjunto de teste e lidas do artefato. "
        "A base da UCI possui ~22% de inadimplentes (desbalanceada), por isso o Recall "
        "(Sensibilidade) e os Falsos Negativos são priorizados na avaliação. "
    )
    if balanceado:
        texto += (
            "O classificador foi treinado com class_weight='balanced', o que tende a "
            "superestimar a chance real de inadimplência. "
        )
    return texto + (
        "Por isso, probabilidade_atraso é um escore do modelo, não uma probabilidade "
        f"calibrada: use-o para ordenar riscos. Limiar de decisão: {obter_limiar():.2f}."
    )


CORTE_RISCO_MODERADO = 0.35
RECOMENDACAO_BAIXO = (
    "Cliente com bom perfil de adimplência. Manter limite atual ou elegível para aumento gradual."
)
RECOMENDACAO_MODERADO = (
    "Atenção preventiva: enviar lembretes automáticos antes do vencimento e monitorar uso do rotativo."
)
RECOMENDACAO_ALTO = (
    "Alta probabilidade de atraso no próximo mês. Recomenda-se oferta proativa de "
    "parcelamento/renegociação e bloqueio de aumento de limite."
)


def classificar_risco(prob: float, limiar: float) -> Tuple[int, str, str, str]:
    """
    Regra única para classe e faixa de risco (sem contradições entre as duas):
    - classe 1 (Risco Alto, red) quando prob >= limiar;
    - Risco Moderado (yellow) quando min(0.35, limiar) <= prob < limiar;
    - Risco Baixo (green) quando prob < min(0.35, limiar).
    Retorna (classe, nivel, cor, recomendacao).
    """
    if prob >= limiar:
        return 1, "Risco Alto", "red", RECOMENDACAO_ALTO
    if prob >= min(CORTE_RISCO_MODERADO, limiar):
        return 0, "Risco Moderado", "yellow", RECOMENDACAO_MODERADO
    return 0, "Risco Baixo", "green", RECOMENDACAO_BAIXO


CASAS_GARANTIA_LIMIAR = 3


def formatar_probabilidade(prob: float) -> Tuple[float, str]:
    """
    Valores exibidos do escore: probabilidade_atraso com 4 casas e o percentual com 1 casa,
    ambos TRUNCADOS para baixo (nunca arredondados para cima). Assim o valor exibido fica do
    mesmo lado do corte que o escore bruto (ex.: 0,49998 aparece como 0.4999 e "49.9%", e não
    "50.0%" com classe 0). A classe continua calculada com o escore bruto.

    Usa a representação decimal mais curta do float (repr), e não sua expansão binária
    exata: 0.57 é truncado para 0.57, e não para 0.5699.

    Limite da garantia: vale para cortes com no máximo 3 casas decimais (o percentual tem 1
    casa, isto é, 3 casas na probabilidade). Ver ARTEFATO_MODELO.md.
    """
    decimal = Decimal(repr(float(prob)))
    prob_4 = decimal.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    percentual = (decimal * 100).quantize(Decimal("0.1"), rounding=ROUND_DOWN)
    return float(prob_4), f"{percentual}%"


def casas_decimais(valor: float) -> int:
    """Número de casas decimais da representação mais curta do float (0.5037 -> 4)."""
    return max(0, -Decimal(repr(float(valor))).normalize().as_tuple().exponent)


def avisar_limiar_com_muitas_casas(limiar: float) -> None:
    """Registra um aviso (sem bloquear a API) quando o corte tem mais de 3 casas decimais."""
    if casas_decimais(limiar) > CASAS_GARANTIA_LIMIAR:
        logger.warning(
            "limiar_decisao=%s tem mais de %d casas decimais: o escore exibido (truncado em 4 "
            "casas e o percentual em 1 casa) pode aparecer do lado oposto do corte em relação "
            "à classe prevista (ex.: corte 0.5037 e escore 0.50375 = classe 1 exibida como "
            "50.3%%). A classe continua correta. Ver ARTEFATO_MODELO.md.",
            limiar,
            CASAS_GARANTIA_LIMIAR,
        )


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
        # O log registra o caminho completo; a resposta HTTP mostra apenas o nome do arquivo
        STATE["erro"] = str(exc).replace(str(caminho), caminho.name)
        logger.error("Falha ao carregar o modelo: %s", exc)
        return
    STATE.update(artefato)
    avisar_limiar_com_muitas_casas(obter_limiar())
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
        "rotas_disponiveis": [
            "/predict", "/metrics", "/comparacao-modelos", "/dataset-check", "/docs",
        ],
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
def prever_risco_credito(
    cliente: ClienteCreditoInput,
    estrito: bool = Query(
        False,
        description=(
            "Se true, exige os 15 campos dos meses 2 a 6 (PAY_2..6, BILL_AMT2..6, "
            "PAY_AMT2..6) e retorna 422 listando os ausentes, sem preenchimento automático."
        ),
    ),
):
    """
    Recebe os dados de um cliente do Front-end (Vercel), preenche automaticamente
    o histórico caso o usuário envie apenas o mês mais recente, executa o Pipeline
    do Scikit-Learn e retorna a classe predita (0 ou 1) e o escore de risco.

    Atenção: probabilidade_atraso é o escore de predict_proba, não uma probabilidade
    calibrada; serve para ordenar riscos, não como percentual literal de chance.
    Classe e faixa de risco usam o limiar_decisao do artefato (padrão 0,50).
    Os campos estimados são listados em campos_preenchidos_automaticamente e
    sinalizados em aviso.
    """
    dados_dict = cliente.model_dump()
    campos_preenchidos = [col for col in CAMPOS_OPCIONAIS if dados_dict[col] is None]
    if estrito and campos_preenchidos:
        raise HTTPException(
            status_code=422,
            detail={
                "mensagem": (
                    "Modo estrito (estrito=true): todos os campos dos meses 2 a 6 devem ser "
                    "informados."
                ),
                "campos_faltantes": campos_preenchidos,
            },
        )

    exigir_modelo()

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
        prob_calote = float(STATE["pipeline"].predict_proba(df_input)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erro ao executar inferência no Pipeline: {str(exc)}"
        )

    # Classe e faixa de risco vêm da mesma regra, com o limiar definido no artefato
    classe_predita, nivel_risco, cor, recomendacao = classificar_risco(prob_calote, obter_limiar())
    # Exibição truncada para baixo: coerente com a classe (calculada acima com o escore bruto)
    prob_exibida, percentual_exibido = formatar_probabilidade(prob_calote)

    return PredicaoOutput(
        inadimplente_predito=classe_predita,
        probabilidade_atraso=prob_exibida,
        probabilidade_percentual=percentual_exibido,
        nivel_risco=nivel_risco,
        cor_indicador=cor,
        recomendacao_negocio=recomendacao,
        campos_preenchidos_automaticamente=campos_preenchidos,
        aviso=(
            f"{len(campos_preenchidos)} campo(s) do histórico (meses 2 a 6) não foram "
            "informados e foram estimados automaticamente a partir do mês mais recente. "
            "Esses valores não são dados reais do cliente, o que reduz a confiança na previsão."
            if campos_preenchidos
            else None
        ),
    )


@app.get("/metrics", response_model=MetricasModeloOutput, tags=["Avaliação (C318)"])
def obter_metricas_modelo():
    """
    Retorna as métricas de avaliação do modelo sobre o conjunto de teste (Recall,
    Precisão, Acurácia, F1-Score e Matriz de Confusão) para exibição no Front-end.
    Todos os valores são lidos do artefato gerado pelo script de treino.
    Também traz, quando o artefato os tem, ROC-AUC, PR-AUC, ROC-AUC do chute e da
    validação cruzada, e os metadados do modelo (nome, hiperparâmetros, corte, uso dos
    atributos sensíveis e data do treino). Campo ausente no artefato vem null.
    """
    exigir_modelo()
    m = STATE["metricas"]
    meta = STATE["metadados"]
    return MetricasModeloOutput(
        # Opcionais: copiados do artefato como estão; ausentes viram null
        **{chave: m.get(chave) for chave in CHAVES_METRICAS_OPCIONAIS},
        **{chave: meta.get(chave) for chave in CHAVES_METADADOS_EXPOSTOS},
        algoritmo=m["algoritmo"],
        origem_modelo=STATE["origem"],
        total_amostras_teste=m["total_amostras_teste"],
        acuracia=m["acuracia"],
        precisao=m["precisao"],
        recall_sensibilidade=m["recall_sensibilidade"],
        f1_score=m["f1_score"],
        matriz_confusao=m["matriz_confusao"],
        observacao_c318=descrever_observacao(),
    )


class ResultadosIndisponiveisError(Exception):
    """Arquivo de resultados do Passo 2 ausente, ilegível ou fora do formato esperado."""


def pasta_resultados() -> Path:
    """Pasta dos resultados: variável C318_RESULTADOS_DIR (lida na chamada) ou a padrão."""
    return Path(os.getenv("C318_RESULTADOS_DIR") or RESULTADOS_DIR_PADRAO)


def exibir_resultado(caminho: Path) -> str:
    """Caminho do arquivo de resultados para mensagens, sem expor caminho absoluto do servidor."""
    try:
        return caminho.relative_to(BASE_DIR.parent).as_posix()
    except ValueError:
        return f"C318_RESULTADOS_DIR/{caminho.name}"


def _exigir_arquivo(caminho: Path) -> None:
    if not caminho.is_file():
        raise ResultadosIndisponiveisError(
            f"arquivo de resultados não encontrado: {exibir_resultado(caminho)}"
        )


def ler_comparacao_cv(caminho: Path) -> List[Dict[str, Any]]:
    """
    Lê as métricas de validação cruzada de comparacao_modelos_cv.csv.
    A coluna melhores_parametros é ignorada de propósito: está no formato de dicionário do
    Python (aspas simples, None) e não é JSON; os hiperparâmetros vêm do JSON.
    """
    _exigir_arquivo(caminho)
    try:
        with open(caminho, encoding="utf-8-sig", newline="") as arquivo:
            linhas = list(csv.DictReader(arquivo))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ResultadosIndisponiveisError(
            f"não foi possível ler {exibir_resultado(caminho)}: {type(exc).__name__}"
        ) from exc
    if not linhas:
        raise ResultadosIndisponiveisError(f"{exibir_resultado(caminho)} não tem modelos")
    faltantes = [c for c in ["modelo", *COLUNAS_METRICAS_CV, "vencedor"] if c not in linhas[0]]
    if faltantes:
        raise ResultadosIndisponiveisError(
            f"{exibir_resultado(caminho)} sem as colunas {faltantes}"
        )
    modelos = []
    for linha in linhas:
        try:
            metricas = {c: float(linha[c]) for c in COLUNAS_METRICAS_CV}
        except (TypeError, ValueError) as exc:
            raise ResultadosIndisponiveisError(
                f"valor não numérico em {exibir_resultado(caminho)} (modelo {linha['modelo']!r})"
            ) from exc
        if not all(math.isfinite(v) for v in metricas.values()):
            raise ResultadosIndisponiveisError(
                f"valor não finito em {exibir_resultado(caminho)} (modelo {linha['modelo']!r})"
            )
        vencedor = (linha["vencedor"] or "").strip().lower()
        if vencedor not in ("true", "false"):
            raise ResultadosIndisponiveisError(
                f"coluna vencedor inválida em {exibir_resultado(caminho)}: {linha['vencedor']!r}"
            )
        modelos.append({"modelo": linha["modelo"], **metricas, "vencedor": vencedor == "true"})
    if sum(m["vencedor"] for m in modelos) != 1:
        raise ResultadosIndisponiveisError(
            f"{exibir_resultado(caminho)} deve marcar exatamente um vencedor"
        )
    return modelos


def ler_hiperparametros(caminho: Path) -> Dict[str, Any]:
    """Lê melhores_hiperparametros.json (melhores parâmetros, grade e modelo final)."""
    _exigir_arquivo(caminho)
    try:
        with open(caminho, encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultadosIndisponiveisError(
            f"não foi possível ler {exibir_resultado(caminho)}: {type(exc).__name__}"
        ) from exc
    if (
        not isinstance(dados, dict)
        or not isinstance(dados.get("modelos"), dict)
        or not isinstance(dados.get("final"), dict)
        or not isinstance(dados["final"].get("modelo"), str)
    ):
        raise ResultadosIndisponiveisError(
            f"{exibir_resultado(caminho)} sem as chaves 'modelos' e 'final.modelo'"
        )
    return dados


@app.get(
    "/comparacao-modelos", response_model=ComparacaoModelosOutput, tags=["Avaliação (C318)"]
)
def comparar_modelos():
    """
    Comparação dos modelos na validação cruzada (só treino), lida dos resultados do script
    de treino: métricas (média e desvio) de comparacao_modelos_cv.csv e hiperparâmetros
    (melhores e grade testada) de melhores_hiperparametros.json. O Chute não tem
    hiperparâmetros (null). coerente_com_modelo_carregado indica se o vencedor do CSV,
    final.modelo do JSON e o modelo do artefato carregado coincidem.

    Os arquivos são lidos a cada requisição: são pequenos, a rota é chamada poucas vezes e
    assim a resposta reflete os resultados atuais, sem reiniciar a API. Arquivo ausente ou
    ilegível: HTTP 503.
    """
    pasta = pasta_resultados()
    try:
        modelos = ler_comparacao_cv(pasta / ARQUIVO_COMPARACAO_CV)
        hiper = ler_hiperparametros(pasta / ARQUIVO_HIPERPARAMETROS)
    except ResultadosIndisponiveisError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Comparação dos modelos indisponível: {exc}. Gere os resultados executando "
                "'python analise_e_modelagem_c318.py' na pasta backend/ (ou defina "
                "C318_RESULTADOS_DIR com a pasta dos resultados)."
            ),
        )

    for m in modelos:
        info = hiper["modelos"].get(m["modelo"])
        m["melhores_parametros"] = info.get("melhores_parametros") if isinstance(info, dict) else None
        m["grade"] = info.get("grade") if isinstance(info, dict) else None

    vencedor = next(m["modelo"] for m in modelos if m["vencedor"])
    modelo_carregado = STATE["metadados"].get("modelo") if STATE["pipeline"] is not None else None
    return ComparacaoModelosOutput(
        criterio=hiper.get("criterio"),
        n_folds=hiper.get("n_folds"),
        modelos=modelos,
        vencedor=vencedor,
        final=hiper["final"],
        modelo_carregado=modelo_carregado,
        coerente_com_modelo_carregado=(
            modelo_carregado is not None and vencedor == hiper["final"]["modelo"] == modelo_carregado
        ),
    )
