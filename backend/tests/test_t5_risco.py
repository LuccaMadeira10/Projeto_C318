"""
T5 — Classe prevista e faixa de risco calculadas pela mesma regra (sem contradições).
"""

import random

import pytest

from conftest import CENARIO_A, abrir_cliente

LIMIARES = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
COR = {"Risco Baixo": "green", "Risco Moderado": "yellow", "Risco Alto": "red"}


# Respostas da API trazem probabilidade_atraso arredondada em 4 casas, mas a classe e a faixa
# usam o escore bruto. A até meia unidade da 4ª casa de um corte, as duas leituras podem
# divergir (ex.: escore 0,49998 é exibido como 0.5 e corretamente classificado como 0).
TOLERANCIA_ARREDONDAMENTO = 5e-5


def _verificar(prob, limiar, classe, nivel, cor, recomendacao, arredondado=False):
    assert classe in (0, 1)
    assert COR[nivel] == cor
    assert isinstance(recomendacao, str) and recomendacao
    assert (classe == 1) == (nivel == "Risco Alto")
    if not arredondado or abs(prob - limiar) > TOLERANCIA_ARREDONDAMENTO:
        assert (classe == 1) == (prob >= limiar)
    corte_moderado = min(0.35, limiar)
    if classe == 0 and (not arredondado or abs(prob - corte_moderado) > TOLERANCIA_ARREDONDAMENTO):
        esperado = "Risco Baixo" if prob < corte_moderado else "Risco Moderado"
        assert nivel == esperado


def test_classificar_risco_propriedade():
    import main

    for limiar in LIMIARES:
        for milesimo in range(0, 1001):
            prob = milesimo / 1000
            _verificar(prob, limiar, *main.classificar_risco(prob, limiar))


def test_classificar_risco_preserva_recomendacoes():
    import main

    textos = {main.classificar_risco(p, 0.5)[3] for p in (0.1, 0.4, 0.9)}
    assert len(textos) == 3
    assert any(t.startswith("Cliente com bom perfil de adimplência") for t in textos)
    assert any(t.startswith("Atenção preventiva") for t in textos)
    assert any(t.startswith("Alta probabilidade de atraso") for t in textos)


def _payload_aleatorio(rng: random.Random):
    payload = {
        "LIMIT_BAL": rng.choice([10000, 20000, 30000, 50000, 80000, 120000, 200000, 360000, 500000]),
        "SEX": rng.randint(1, 2),
        "EDUCATION": rng.randint(1, 4),
        "MARRIAGE": rng.randint(1, 3),
        "AGE": rng.randint(21, 70),
        "PAY_0": rng.randint(-2, 8),
        "BILL_AMT1": round(rng.uniform(-1000, 200000), 2),
        "PAY_AMT1": round(rng.uniform(0, 30000), 2),
    }
    for i in range(2, 7):
        if rng.random() < 0.4:
            payload[f"PAY_{i}"] = rng.randint(-2, 8)
        if rng.random() < 0.4:
            payload[f"BILL_AMT{i}"] = round(rng.uniform(0, 200000), 2)
        if rng.random() < 0.4:
            payload[f"PAY_AMT{i}"] = round(rng.uniform(0, 30000), 2)
    return payload


def test_predict_classe_coerente_com_faixa(cliente):
    import main

    limiar = main.STATE.get("metadados", {}).get("limiar_decisao", 0.5)
    rng = random.Random(42)
    faixas = set()
    for _ in range(200):
        resp = cliente.post("/predict", json=_payload_aleatorio(rng))
        assert resp.status_code == 200, resp.text
        d = resp.json()
        faixas.add(d["nivel_risco"])
        _verificar(d["probabilidade_atraso"], limiar, d["inadimplente_predito"],
                   d["nivel_risco"], d["cor_indicador"], d["recomendacao_negocio"], arredondado=True)
    assert faixas == set(COR), f"a amostra deveria cobrir as três faixas: {faixas}"


@pytest.mark.parametrize("limiar", [0.3, 0.7])
def test_limiar_do_artefato_e_respeitado(monkeypatch, fabrica_artefato, limiar):
    """O limiar vem de metadados.limiar_decisao, sem alteração na API."""
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido", limiar=limiar))
    rng = random.Random(7)
    with tc as c:
        for _ in range(60):
            d = c.post("/predict", json=_payload_aleatorio(rng)).json()
            _verificar(d["probabilidade_atraso"], limiar, d["inadimplente_predito"],
                       d["nivel_risco"], d["cor_indicador"], d["recomendacao_negocio"], arredondado=True)
        assert c.post("/predict", json=CENARIO_A).status_code == 200
