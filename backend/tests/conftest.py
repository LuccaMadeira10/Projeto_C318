"""
Fixtures compartilhadas pelos testes da API Crédito Consciente (C318).

Os testes funcionam tanto a partir de backend/ (python -m pytest) quanto da raiz
(python -m pytest backend/tests), pois todos os caminhos são relativos a este arquivo.
"""

import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
CSV_REAL = REPO_DIR / "data" / "UCI_Credit_Card.csv"

# Payload exatamente como o front-end (frontend/index.html) envia com os valores padrão
# do formulário: os campos avançados (PAY_2, PAY_3, BILL_AMT2, PAY_AMT2) só são enviados
# quando preenchidos.
PAYLOAD_FRONT = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "BILL_AMT1": 0,
    "PAY_AMT1": 0,
}

# Payload do front com os campos avançados preenchidos
PAYLOAD_FRONT_AVANCADO = {
    "LIMIT_BAL": 80000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 1,
    "PAY_2": 1,
    "PAY_3": 0,
    "BILL_AMT1": 40000,
    "BILL_AMT2": 38000,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 1500,
}

# Cenário A — bom pagador (esperado: classe 0 e Risco Baixo)
CENARIO_A = {
    "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 38,
    "PAY_0": -1, "BILL_AMT1": 12000, "PAY_AMT1": 12000,
}

# Cenário B — atraso recente (esperado: classe 1 e Risco Alto)
CENARIO_B = {
    "LIMIT_BAL": 30000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 25,
    "PAY_0": 2, "BILL_AMT1": 29500, "PAY_AMT1": 0,
}


@pytest.fixture(scope="session")
def modelo_real() -> Path:
    """
    Caminho do artefato real usado nos testes de integração.

    Por padrão é o artefato versionado (backend/modelo_credito.joblib). A variável
    C318_MODELO_REAL permite apontar para outro artefato (por exemplo, um candidato
    gerado no Passo 2) sem tocar no arquivo versionado.
    """
    caminho = Path(os.environ.get("C318_MODELO_REAL", BACKEND_DIR / "modelo_credito.joblib"))
    if not caminho.is_file():
        pytest.fail(f"Artefato real não encontrado em {caminho}. Gere-o com o script de treino.")
    return caminho


def abrir_cliente(monkeypatch, caminho_modelo):
    """
    Configura MODEL_PATH e (re)importa o main. O carregamento do modelo acontece no
    startup, isto é, dentro de `with TestClient(main.app)`.

    O reload garante isolamento entre testes que usam artefatos diferentes.
    """
    monkeypatch.setenv("MODEL_PATH", str(caminho_modelo))
    main = importlib.reload(importlib.import_module("main"))
    return main, TestClient(main.app)


@pytest.fixture
def cliente(monkeypatch, modelo_real):
    """TestClient com o startup executado e o modelo real carregado."""
    _, tc = abrir_cliente(monkeypatch, modelo_real)
    with tc as c:
        yield c


# ---------------------------------------------------------------------------
# Fábrica de artefatos pequenos (somente para testes; a produção nunca treina)
# ---------------------------------------------------------------------------

def _treinar_pipeline_pequeno(renomear_pay0: bool = False):
    """
    Treina um Pipeline pequeno com 1.500 registros reais da base UCI, apenas para
    exercitar o carregamento e a inferência da API. Retorna (pipeline, metricas).
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from test_contrato_caracterizacao import FEATURES_ESPERADAS

    df = pd.read_csv(CSV_REAL).sample(n=2000, random_state=7)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    colunas = list(FEATURES_ESPERADAS)
    if renomear_pay0:
        df = df.rename(columns={"PAY_0": "PAY_1"})
        colunas = ["PAY_1" if c == "PAY_0" else c for c in colunas]
    X, y = df[colunas], df["default.payment.next.month"]
    X_tr, X_te, y_tr, y_te = X.iloc[:1500], X.iloc[1500:], y.iloc[:1500], y.iloc[1500:]

    categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    pipeline = Pipeline([
        ("preprocessador", ColumnTransformer([
            ("num", StandardScaler(), [c for c in colunas if c not in categoricas]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ])),
        ("classificador", RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)),
    ])
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    metricas = {
        "algoritmo": "RandomForestClassifier (artefato de teste)",
        "total_amostras_teste": int(len(y_te)),
        "acuracia": round(float(accuracy_score(y_te, y_pred)), 4),
        "precisao": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall_sensibilidade": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "matriz_confusao": {"VP": int(tp), "VN": int(tn), "FP": int(fp), "FN": int(fn)},
    }
    return pipeline, metricas


@pytest.fixture(scope="session")
def _pipelines_pequenos():
    return {
        "valido": _treinar_pipeline_pequeno(),
        "colunas_incompativeis": _treinar_pipeline_pequeno(renomear_pay0=True),
    }


VARIANTES_DEFEITUOSAS = [
    "ausente",
    "corrompido",
    "vazio",
    "sem_pipeline",
    "sem_metricas",
    "colunas_incompativeis",
    "pipeline_puro",
]


@pytest.fixture
def fabrica_artefato(tmp_path, _pipelines_pequenos):
    """
    Retorna uma função criar(variante="valido", limiar=None, metadados=True) que grava
    um artefato em tmp_path e devolve o caminho. Variantes: "valido" e as listadas em
    VARIANTES_DEFEITUOSAS.
    """
    import joblib
    import sklearn

    def criar(variante: str = "valido", limiar=None, metadados: bool = True) -> Path:
        caminho = tmp_path / f"modelo_{variante}.joblib"
        pipeline, metricas = _pipelines_pequenos[
            "colunas_incompativeis" if variante == "colunas_incompativeis" else "valido"
        ]
        artefato = {"pipeline": pipeline, "metricas": dict(metricas)}
        if metadados:
            artefato["metadados"] = {
                "sklearn_version": sklearn.__version__,
                "treinado_em": "2026-11-18T00:00:00+00:00",
                "csv_origem": "data/UCI_Credit_Card.csv",
                "limiar_decisao": 0.5 if limiar is None else limiar,
            }
        if variante == "ausente":
            return caminho
        if variante == "corrompido":
            caminho.write_bytes(b"isto nao e um arquivo joblib \x00\x01\x02" * 50)
            return caminho
        if variante == "vazio":
            caminho.write_bytes(b"")
            return caminho
        if variante == "sem_pipeline":
            artefato.pop("pipeline")
        elif variante == "sem_metricas":
            artefato.pop("metricas")
        elif variante == "pipeline_puro":
            artefato = pipeline
        joblib.dump(artefato, caminho)
        return caminho

    return criar
