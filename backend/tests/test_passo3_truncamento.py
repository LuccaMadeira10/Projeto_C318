"""
Passo 3, T3 — Probabilidade exibida coerente com a classe.

probabilidade_atraso (4 casas) e probabilidade_percentual (1 casa) são truncados para baixo;
a classe continua calculada com o escore bruto. Com cortes de até 3 casas decimais, o valor
exibido é maior ou igual ao corte se e somente se a classe é 1. Acima disso, a API avisa no log.
"""

import logging
import math
from decimal import Decimal

import numpy as np
import pytest

from conftest import CENARIO_A, abrir_cliente

CORTES_PASSO2 = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0,10 a 0,90, passo 0,05
CORTES_3_CASAS = [0.101, 0.255, 0.333, 0.355, 0.499, 0.501, 0.649, 0.777, 0.899]
CORTES = CORTES_PASSO2 + CORTES_3_CASAS


def _escores_perto(corte):
    escores = {corte + k * 1e-5 for k in range(-150, 151)}
    escores |= {corte + k * 1e-6 for k in range(-20, 21)}
    escores |= {corte - 1e-9, corte + 1e-9, corte - 1e-12, corte + 1e-12,
                math.nextafter(corte, 0.0), math.nextafter(corte, 1.0), corte}
    return sorted(e for e in escores if 0.0 <= e <= 1.0)


def _lado_exibido(prob_4, percentual, corte):
    """(exibido_4_casas >= corte, percentual_exibido >= corte), comparando em decimal exato."""
    corte_dec = Decimal(repr(corte))
    pct = Decimal(percentual.rstrip("%")) / 100
    return Decimal(repr(prob_4)) >= corte_dec, pct >= corte_dec


def test_cortes_do_passo2_tem_no_maximo_3_casas():
    import main

    assert CORTES_PASSO2[0] == 0.10 and CORTES_PASSO2[-1] == 0.90
    assert all(main.casas_decimais(c) <= 3 for c in CORTES)


@pytest.mark.parametrize("corte", CORTES)
def test_exibido_do_mesmo_lado_do_corte_que_a_classe(corte):
    import main

    for prob in _escores_perto(corte):
        classe, nivel, _, _ = main.classificar_risco(prob, corte)
        assert classe == int(prob >= corte)  # a classe usa o escore bruto
        prob_4, percentual = main.formatar_probabilidade(prob)
        assert prob_4 <= prob and Decimal(percentual.rstrip("%")) <= Decimal(repr(prob)) * 100
        lado_4, lado_pct = _lado_exibido(prob_4, percentual, corte)
        assert lado_4 == (classe == 1), (corte, prob, prob_4)
        assert lado_pct == (classe == 1), (corte, prob, percentual)
        # A faixa Moderado (corte 0,35) também fica coerente com o valor exibido
        if classe == 0:
            moderado_4, moderado_pct = _lado_exibido(prob_4, percentual, min(0.35, corte))
            assert moderado_4 == moderado_pct == (nivel == "Risco Moderado"), (corte, prob)


def test_caso_049998():
    import main

    assert main.classificar_risco(0.49998, 0.5)[0] == 0
    assert main.formatar_probabilidade(0.49998) == (0.4999, "49.9%")


@pytest.mark.parametrize("prob, esperado", [
    (0.57, (0.57, "57.0%")),        # repr curto: não vira 0.5699 pela expansão binária
    (0.5, (0.5, "50.0%")),
    (0.123456789, (0.1234, "12.3%")),
    (0.99999, (0.9999, "99.9%")),
    (1.0, (1.0, "100.0%")),
    (0.0, (0.0, "0.0%")),
])
def test_formatar_probabilidade_trunca(prob, esperado):
    import main

    assert main.formatar_probabilidade(prob) == esperado


class _PipelineFixo:
    """Substitui o pipeline carregado para devolver um escore escolhido pelo teste."""

    def __init__(self, escore):
        self.escore = escore

    def predict_proba(self, X):
        return np.array([[1.0 - self.escore, self.escore]] * len(X))


@pytest.mark.parametrize("escore", [0.49998, 0.499999999, 0.5, 0.50001, 0.34999, 0.35,
                                    math.nextafter(0.5, 0.0)])
def test_predict_exibe_do_lado_da_classe(cliente, monkeypatch, escore):
    import main

    monkeypatch.setitem(main.STATE, "pipeline", _PipelineFixo(escore))
    corte = main.obter_limiar()
    d = cliente.post("/predict", json=CENARIO_A).json()
    assert d["inadimplente_predito"] == int(escore >= corte)
    lado_4, lado_pct = _lado_exibido(d["probabilidade_atraso"], d["probabilidade_percentual"], corte)
    assert lado_4 == lado_pct == (d["inadimplente_predito"] == 1)
    if escore == 0.49998:
        assert d["probabilidade_percentual"] == "49.9%" and d["probabilidade_atraso"] == 0.4999


def test_limite_corte_com_4_casas_gera_aviso_no_log(monkeypatch, fabrica_artefato, caplog):
    import main

    # O contraexemplo que motiva o aviso: classe 1, mas exibido abaixo do corte
    assert main.classificar_risco(0.50375, 0.5037)[0] == 1
    assert main.formatar_probabilidade(0.50375)[1] == "50.3%"

    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido", limiar=0.5037))
    with caplog.at_level(logging.WARNING, logger="credito_consciente"):
        with tc as c:
            assert c.get("/").json()["modelo_carregado"] is True  # não bloqueia a API
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("0.5037" in r.getMessage() and "casas decimais" in r.getMessage() for r in avisos)


@pytest.mark.parametrize("limiar", [0.5, 0.355])
def test_corte_com_ate_3_casas_nao_gera_aviso(monkeypatch, fabrica_artefato, caplog, limiar):
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido", limiar=limiar))
    with caplog.at_level(logging.WARNING, logger="credito_consciente"):
        with tc:
            pass
    assert not [r for r in caplog.records if "casas decimais" in r.getMessage()]
