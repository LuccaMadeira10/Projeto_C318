"""
Passo 3, T1 — /metrics expõe ROC-AUC, PR-AUC e os metadados do modelo lidos do artefato.
Campo ausente no artefato vem null (nunca inventado).
"""

import joblib

from conftest import abrir_cliente

METRICAS_OPCIONAIS = ["roc_auc", "pr_auc", "roc_auc_chute", "roc_auc_cv_treino"]
METADADOS_EXPOSTOS = [
    "modelo", "hiperparametros", "limiar_decisao", "atributos_sensiveis_usados", "treinado_em",
]


def test_metrics_campos_novos_iguais_ao_artefato_real(cliente, modelo_real):
    artefato = joblib.load(modelo_real)
    dados = cliente.get("/metrics").json()
    for chave in METRICAS_OPCIONAIS:
        assert chave in artefato["metricas"], f"artefato real deveria ter {chave}"
        assert dados[chave] == artefato["metricas"][chave], chave
    for chave in METADADOS_EXPOSTOS:
        assert chave in artefato["metadados"], f"artefato real deveria ter {chave}"
        assert dados[chave] == artefato["metadados"][chave], chave


def test_metrics_campos_ausentes_vem_null(monkeypatch, fabrica_artefato):
    # A fábrica não grava as métricas informativas nem modelo/hiperparâmetros/atributos
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido"))
    with tc as c:
        dados = c.get("/metrics").json()
    for chave in METRICAS_OPCIONAIS + ["modelo", "hiperparametros", "atributos_sensiveis_usados"]:
        assert chave in dados and dados[chave] is None, chave
    assert dados["limiar_decisao"] == 0.5
    assert dados["treinado_em"] == "2026-11-18T00:00:00+00:00"


def test_metrics_sem_metadados_limiar_e_data_null(monkeypatch, fabrica_artefato):
    """Sem metadados, a API usa o corte padrão internamente, mas não o apresenta como do artefato."""
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("valido", metadados=False))
    with tc as c:
        dados = c.get("/metrics").json()
    for chave in METADADOS_EXPOSTOS:
        assert dados[chave] is None, chave


def test_metrics_campos_parciais(monkeypatch, fabrica_artefato):
    caminho = fabrica_artefato(
        "valido",
        metricas_extra={"roc_auc": 0.7, "pr_auc": 0.4},
        metadados_extra={"modelo": "KNN", "atributos_sensiveis_usados": True},
    )
    _, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        dados = c.get("/metrics").json()
    assert dados["roc_auc"] == 0.7 and dados["pr_auc"] == 0.4
    assert dados["roc_auc_chute"] is None and dados["roc_auc_cv_treino"] is None
    assert dados["modelo"] == "KNN" and dados["atributos_sensiveis_usados"] is True
    assert dados["hiperparametros"] is None


def test_metrics_continua_503_sem_modelo(monkeypatch, fabrica_artefato):
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("ausente"))
    with tc as c:
        assert c.get("/metrics").status_code == 503
