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


def _abrir_cliente(monkeypatch, caminho_modelo):
    """Configura MODEL_PATH e importa o main (o startup roda dentro do TestClient)."""
    monkeypatch.setenv("MODEL_PATH", str(caminho_modelo))
    main = importlib.import_module("main")
    return main, TestClient(main.app)


@pytest.fixture
def cliente(monkeypatch, modelo_real):
    """TestClient com o startup executado e o modelo real carregado."""
    _, tc = _abrir_cliente(monkeypatch, modelo_real)
    with tc as c:
        yield c
