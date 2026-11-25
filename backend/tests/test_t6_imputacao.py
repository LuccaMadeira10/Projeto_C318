"""
T6 — Transparência no preenchimento automático dos meses 2 a 6.
"""

import pytest

from conftest import CENARIO_A, CENARIO_B, PAYLOAD_FRONT, PAYLOAD_FRONT_AVANCADO

OPCIONAIS = (
    [f"PAY_{i}" for i in range(2, 7)]
    + [f"BILL_AMT{i}" for i in range(2, 7)]
    + [f"PAY_AMT{i}" for i in range(2, 7)]
)


def _expandir_manual(payload):
    """Reproduz explicitamente a regra de preenchimento usada pela API."""
    completo = dict(payload)
    base = 0 if payload["PAY_0"] <= 0 else max(0, payload["PAY_0"] - 1)
    for i in range(2, 7):
        completo.setdefault(f"PAY_{i}", base if i <= 3 else 0)
        completo.setdefault(f"BILL_AMT{i}", payload["BILL_AMT1"])
        completo.setdefault(f"PAY_AMT{i}", payload["PAY_AMT1"])
    return completo


def test_imputacao_transparente(cliente):
    minimo = cliente.post("/predict", json=PAYLOAD_FRONT).json()
    assert minimo["campos_preenchidos_automaticamente"] == OPCIONAIS
    assert isinstance(minimo["aviso"], str)
    assert "estimad" in minimo["aviso"] and "confiança" in minimo["aviso"]

    avancado = cliente.post("/predict", json=PAYLOAD_FRONT_AVANCADO).json()
    informados = {"PAY_2", "PAY_3", "BILL_AMT2", "PAY_AMT2"}
    assert avancado["campos_preenchidos_automaticamente"] == [c for c in OPCIONAIS if c not in informados]
    assert avancado["aviso"]

    completo = cliente.post("/predict", json=_expandir_manual(CENARIO_A)).json()
    assert completo["campos_preenchidos_automaticamente"] == []
    assert completo["aviso"] is None


@pytest.mark.parametrize("payload", [PAYLOAD_FRONT, CENARIO_A, CENARIO_B, PAYLOAD_FRONT_AVANCADO])
def test_imputacao_nao_mudou_resultado(cliente, payload):
    automatico = cliente.post("/predict", json=payload).json()
    manual = cliente.post("/predict", json=_expandir_manual(payload)).json()
    assert automatico["probabilidade_atraso"] == manual["probabilidade_atraso"]
    assert automatico["inadimplente_predito"] == manual["inadimplente_predito"]
    assert automatico["nivel_risco"] == manual["nivel_risco"]


def test_modo_estrito(cliente):
    incompleto = cliente.post("/predict?estrito=true", json=PAYLOAD_FRONT)
    assert incompleto.status_code == 422
    detalhe = incompleto.json()["detail"]
    assert detalhe["campos_faltantes"] == OPCIONAIS
    assert "estrito" in detalhe["mensagem"]

    parcial = cliente.post("/predict?estrito=true", json=PAYLOAD_FRONT_AVANCADO)
    assert parcial.status_code == 422
    assert "PAY_2" not in parcial.json()["detail"]["campos_faltantes"]

    completo = cliente.post("/predict?estrito=true", json=_expandir_manual(CENARIO_B))
    assert completo.status_code == 200
    assert completo.json()["campos_preenchidos_automaticamente"] == []

    # Padrão (estrito=false) continua aceitando o payload mínimo
    assert cliente.post("/predict?estrito=false", json=PAYLOAD_FRONT).status_code == 200
