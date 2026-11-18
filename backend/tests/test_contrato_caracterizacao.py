"""
Bloco C — Testes de caracterização do contrato da API.

Estes testes descrevem o comportamento que o front-end já consome e devem passar
tanto no main.py antigo quanto no corrigido. Verificam a presença e o tipo dos
campos, sem exigir igualdade integral dos dicionários (mudanças aditivas são
permitidas).
"""

import re

import pandas as pd
import pytest

from conftest import CENARIO_A, CENARIO_B, PAYLOAD_FRONT, PAYLOAD_FRONT_AVANCADO

NIVEIS_VALIDOS = {"Risco Baixo", "Risco Moderado", "Risco Alto"}
COR_POR_NIVEL = {"Risco Baixo": "green", "Risco Moderado": "yellow", "Risco Alto": "red"}
CAMPOS_PREDICT = {
    "inadimplente_predito": int,
    "probabilidade_atraso": float,
    "probabilidade_percentual": str,
    "nivel_risco": str,
    "cor_indicador": str,
    "recomendacao_negocio": str,
}

FEATURES_ESPERADAS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def _prever(cliente, payload):
    resp = cliente.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _validar_saida_predict(dados):
    for campo, tipo in CAMPOS_PREDICT.items():
        assert campo in dados, f"campo {campo} ausente"
        if tipo is float:
            assert isinstance(dados[campo], (int, float)) and not isinstance(dados[campo], bool)
        else:
            assert isinstance(dados[campo], tipo), f"{campo} deveria ser {tipo.__name__}"
    assert dados["inadimplente_predito"] in (0, 1)
    assert 0.0 <= dados["probabilidade_atraso"] <= 1.0
    assert re.fullmatch(r"\d{1,3}\.\d%", dados["probabilidade_percentual"])
    assert dados["nivel_risco"] in NIVEIS_VALIDOS
    assert dados["cor_indicador"] in {"green", "yellow", "red"}
    assert dados["recomendacao_negocio"].strip()


def test_raiz_contrato(cliente):
    resp = cliente.get("/")
    assert resp.status_code == 200
    dados = resp.json()
    for campo in ("status", "projeto", "modelo_ativo", "rotas_disponiveis"):
        assert campo in dados
    assert isinstance(dados["status"], str)
    assert isinstance(dados["modelo_ativo"], str)
    for rota in ("/predict", "/metrics", "/dataset-check", "/docs"):
        assert rota in dados["rotas_disponiveis"]


@pytest.mark.parametrize("payload", [PAYLOAD_FRONT, PAYLOAD_FRONT_AVANCADO], ids=["minimo", "avancado"])
def test_predict_payload_do_front(cliente, payload):
    _validar_saida_predict(_prever(cliente, payload))


@pytest.mark.parametrize("payload", [PAYLOAD_FRONT, CENARIO_A, CENARIO_B, PAYLOAD_FRONT_AVANCADO])
def test_predict_formato_percentual(cliente, payload):
    dados = _prever(cliente, payload)
    prob = dados["probabilidade_atraso"]
    # probabilidade_atraso é arredondada em 4 casas; o percentual usa o valor bruto.
    # Aceita-se a diferença de arredondamento de no máximo 0,1 ponto percentual.
    esperado = float(dados["probabilidade_percentual"].rstrip("%"))
    assert abs(esperado - prob * 100) <= 0.05 + 1e-9
    assert dados["probabilidade_percentual"] == f"{esperado:.1f}%"


@pytest.mark.parametrize("payload", [PAYLOAD_FRONT, CENARIO_A, CENARIO_B, PAYLOAD_FRONT_AVANCADO])
def test_predict_cor_e_nivel_mapeados(cliente, payload):
    dados = _prever(cliente, payload)
    assert COR_POR_NIVEL[dados["nivel_risco"]] == dados["cor_indicador"]


def test_cenario_a_bom_pagador(cliente):
    dados = _prever(cliente, CENARIO_A)
    assert dados["inadimplente_predito"] == 0
    assert dados["nivel_risco"] == "Risco Baixo"


def test_cenario_b_atraso(cliente):
    dados_b = _prever(cliente, CENARIO_B)
    dados_a = _prever(cliente, CENARIO_A)
    assert dados_b["inadimplente_predito"] == 1
    assert dados_b["nivel_risco"] == "Risco Alto"
    assert dados_b["probabilidade_atraso"] > dados_a["probabilidade_atraso"]


@pytest.mark.parametrize("base", [CENARIO_A, PAYLOAD_FRONT], ids=["cenario_a", "front"])
def test_atraso_aumenta_risco(cliente, base):
    em_dia = dict(base, PAY_0=-1)
    atrasado = dict(base, PAY_0=2)
    p_em_dia = _prever(cliente, em_dia)["probabilidade_atraso"]
    p_atrasado = _prever(cliente, atrasado)["probabilidade_atraso"]
    assert p_atrasado >= p_em_dia


@pytest.mark.parametrize(
    "alteracao",
    [
        {"LIMIT_BAL": 0},
        {"LIMIT_BAL": -1},
        {"AGE": 17},
        {"AGE": 101},
        {"SEX": 3},
        {"MARRIAGE": 4},
        {"PAY_0": 10},
        {"PAY_AMT1": -1},
    ],
    ids=lambda d: "-".join(f"{k}={v}" for k, v in d.items()),
)
def test_predict_validacao_422(cliente, alteracao):
    resp = cliente.post("/predict", json=dict(CENARIO_A, **alteracao))
    assert resp.status_code == 422


@pytest.mark.parametrize("campo", ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"])
def test_predict_campo_obrigatorio_ausente_422(cliente, campo):
    payload = {k: v for k, v in CENARIO_A.items() if k != campo}
    assert cliente.post("/predict", json=payload).status_code == 422


def test_categorias_nao_documentadas(cliente):
    """
    Contrato atual (preservado):
    - PAY_x = -2 e PAY_x = 0 não constam na documentação original da UCI, mas existem
      na base e são aceitos pelo schema: a API deve responder normalmente.
    - EDUCATION 0/5/6 e MARRIAGE 0 também não são documentados, porém o schema atual
      os rejeita (EDUCATION 1..4, MARRIAGE 1..3) com 422. A normalização
      0/5/6 -> 4 e 0 -> 3 ocorre em limpar_categorias_uci (ver teste abaixo).
    """
    for valor in (-2, 0):
        payload = dict(CENARIO_A, PAY_0=valor, PAY_2=valor, PAY_3=valor)
        _validar_saida_predict(_prever(cliente, payload))
    for alteracao in ({"EDUCATION": 0}, {"EDUCATION": 5}, {"EDUCATION": 6}, {"MARRIAGE": 0}):
        assert cliente.post("/predict", json=dict(CENARIO_A, **alteracao)).status_code == 422


def test_limpar_categorias_uci():
    import main

    original = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6], "MARRIAGE": [0, 1, 2, 3, 1, 2, 3]})
    copia = original.copy()
    limpo = main.limpar_categorias_uci(original)
    assert limpo["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert limpo["MARRIAGE"].tolist() == [3, 1, 2, 3, 1, 2, 3]
    pd.testing.assert_frame_equal(original, copia)


def test_metrics_contrato(cliente):
    resp = cliente.get("/metrics")
    assert resp.status_code == 200
    m = resp.json()
    for campo in ("algoritmo", "origem_modelo", "observacao_c318"):
        assert isinstance(m[campo], str) and m[campo]
    assert isinstance(m["total_amostras_teste"], int)
    for campo in ("acuracia", "precisao", "recall_sensibilidade", "f1_score"):
        assert isinstance(m[campo], (int, float))
        assert 0.0 <= m[campo] <= 1.0
    assert set(m["matriz_confusao"]) >= {"VP", "VN", "FP", "FN"}
    assert all(isinstance(v, int) for v in m["matriz_confusao"].values())


def test_dataset_check_real(cliente):
    resp = cliente.get("/dataset-check")
    assert resp.status_code == 200
    d = resp.json()
    assert d["arquivo_encontrado"] is True
    assert isinstance(d["caminho"], str) and d["caminho"].endswith("UCI_Credit_Card.csv")
    assert d["dataset_valido"] is True
    assert d["total_clientes_linhas"] == 30000
    assert d["total_colunas"] == 25
    assert d["taxa_inadimplencia_percentual"] == "22.12%"
    assert d["colunas_faltantes"] == []


def test_cors_preflight(cliente):
    resp = cliente.options(
        "/predict",
        headers={
            "Origin": "https://exemplo-front.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_openapi_e_docs(cliente):
    caminhos = cliente.get("/openapi.json").json()["paths"]
    for rota in ("/predict", "/metrics", "/dataset-check"):
        assert rota in caminhos
    assert cliente.get("/docs").status_code == 200


def test_feature_columns():
    import main

    assert main.FEATURE_COLUMNS == FEATURES_ESPERADAS
    assert len(main.FEATURE_COLUMNS) == 23
    assert "PAY_0" in main.FEATURE_COLUMNS and "PAY_1" not in main.FEATURE_COLUMNS
