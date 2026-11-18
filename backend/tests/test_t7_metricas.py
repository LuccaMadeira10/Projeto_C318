"""
T7 — /metrics reproduz o artefato e a documentação explica a limitação do escore.
"""

import json

import pytest

from conftest import CENARIO_A, abrir_cliente

CAMPOS = ["algoritmo", "total_amostras_teste", "acuracia", "precisao", "recall_sensibilidade", "f1_score"]
# (acurácia, recall, F1) que NÃO podem reaparecer
FICTICIAS_ANTIGAS = (0.7945, 0.6120, 0.5726)   # escritas à mão no main.py antigo
JOBLIB_INCOMPATIVEL = (0.788, 0.5599, 0.5388)  # artefato antigo (scikit-learn 1.2.2)


def test_metrics_fiel_ao_artefato(cliente, modelo_real):
    import joblib

    artefato = joblib.load(modelo_real)
    m = cliente.get("/metrics").json()
    for campo in CAMPOS:
        assert m[campo] == artefato["metricas"][campo], campo
    assert m["matriz_confusao"] == artefato["metricas"]["matriz_confusao"]
    meta = artefato.get("metadados", {})
    if meta:
        assert meta["sklearn_version"] in m["origem_modelo"]
        assert meta["treinado_em"] in m["origem_modelo"]
    assert cliente.get("/").json()["modelo_ativo"] == m["origem_modelo"]


def test_metrics_internamente_consistentes(cliente):
    m = cliente.get("/metrics").json()
    mc = m["matriz_confusao"]
    vp, vn, fp, fn = mc["VP"], mc["VN"], mc["FP"], mc["FN"]
    assert vp + vn + fp + fn == m["total_amostras_teste"] == 6000
    precisao = vp / (vp + fp)
    recall = vp / (vp + fn)
    assert m["acuracia"] == pytest.approx((vp + vn) / 6000, abs=1e-3)
    assert m["precisao"] == pytest.approx(precisao, abs=1e-3)
    assert m["recall_sensibilidade"] == pytest.approx(recall, abs=1e-3)
    assert m["f1_score"] == pytest.approx(2 * precisao * recall / (precisao + recall), abs=1e-3)


def test_metrics_nao_sao_as_antigas(cliente):
    m = cliente.get("/metrics").json()
    trio = (m["acuracia"], m["recall_sensibilidade"], m["f1_score"])
    for antigo in (FICTICIAS_ANTIGAS, JOBLIB_INCOMPATIVEL):
        assert trio != pytest.approx(antigo, abs=5e-4)


def test_metrics_vem_do_artefato_de_teste(monkeypatch, fabrica_artefato):
    """Com outro artefato, /metrics muda junto: nada está escrito na API."""
    import joblib

    caminho = fabrica_artefato("valido")
    esperado = joblib.load(caminho)["metricas"]
    _, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        m = c.get("/metrics").json()
    assert {k: m[k] for k in CAMPOS} == {k: esperado[k] for k in CAMPOS}
    assert "2026-11-18T00:00:00+00:00" in m["origem_modelo"]


def test_openapi_diz_que_nao_calibrada(cliente):
    schema = cliente.get("/openapi.json").json()
    campo = schema["components"]["schemas"]["PredicaoOutput"]["properties"]["probabilidade_atraso"]
    descricao = campo.get("description", "").lower()
    assert "escore" in descricao
    assert "calibrad" in descricao
    assert "22%" in descricao
    texto_predict = json.dumps(schema["paths"]["/predict"], ensure_ascii=False).lower()
    assert "calibrad" in texto_predict


def test_observacao_explica_limitacao_do_escore(cliente):
    obs = cliente.get("/metrics").json()["observacao_c318"].lower()
    assert "calibrad" in obs
    assert "escore" in obs
    assert "balanced" in obs  # o artefato real usa class_weight='balanced'


def test_observacao_sem_balanced_quando_o_modelo_nao_usa(monkeypatch, fabrica_artefato):
    """A observação descreve o modelo carregado, sem texto fixo sobre hiperparâmetros."""
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido"))  # RF sem class_weight
    with tc as c:
        obs = c.get("/metrics").json()["observacao_c318"].lower()
        assert c.post("/predict", json=CENARIO_A).status_code == 200
    assert "calibrad" in obs
    assert "balanced" not in obs
