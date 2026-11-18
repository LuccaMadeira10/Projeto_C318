"""
Passo 2 — Consistência entre o artefato versionado, os resultados em resultados/ e o notebook.

Não executa o script (rápido): confere que o que foi gravado é coerente e reproduzível.
O teste de vazamento e a execução isolada ficam em test_script_treino.py (slow).
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import BACKEND_DIR, CSV_REAL, REPO_DIR, verificar_metricas_recomputadas
from test_contrato_caracterizacao import FEATURES_ESPERADAS

RESULTADOS = REPO_DIR / "resultados"
NOTEBOOK = REPO_DIR / "notebooks" / "analise_e_modelagem_c318.ipynb"
SCRIPT = BACKEND_DIR / "analise_e_modelagem_c318.py"
MODELOS = {"Chute (referência)", "Regressão Logística", "KNN", "Árvore de Decisão", "Random Forest"}
SENSIVEIS = ["SEX", "AGE", "MARRIAGE"]


@pytest.fixture(scope="module")
def artefato(modelo_real):
    import joblib

    return joblib.load(modelo_real)


@pytest.fixture(scope="module")
def hiperparametros():
    return json.loads((RESULTADOS / "melhores_hiperparametros.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- contrato e métricas

def test_artefato_versionado_e_da_execucao_completa(artefato):
    meta = artefato["metadados"]
    assert meta["modo_rapido"] is False and meta["n_folds_cv"] == 5
    assert meta["csv_origem"] == "data/UCI_Credit_Card.csv"
    assert list(artefato["pipeline"].feature_names_in_) == FEATURES_ESPERADAS


def test_metricas_do_artefato_batem_com_recomputacao(modelo_real):
    verificar_metricas_recomputadas(modelo_real)


# ---------------------------------------------------------------- consistência dos resultados

def test_comparacao_tem_os_5_modelos_e_chute_e_referencia():
    df = pd.read_csv(RESULTADOS / "comparacao_modelos_cv.csv").set_index("modelo")
    assert set(df.index) == MODELOS
    chute = df.loc["Chute (referência)"]
    assert chute["recall_media"] == 0
    assert chute["roc_auc_media"] == pytest.approx(0.5, abs=1e-9)  # constante por construção
    # Abaixo de 0,50 é sinal de bug (por exemplo, usar a coluna errada do predict_proba)
    assert (df.drop(index="Chute (referência)")["roc_auc_media"] > 0.5).all()
    assert df["vencedor"].sum() == 1


def test_vencedor_do_csv_e_o_do_artefato(artefato, hiperparametros):
    df = pd.read_csv(RESULTADOS / "comparacao_modelos_cv.csv")
    vencedor = df.loc[df["vencedor"], "modelo"].item()
    assert vencedor == artefato["metadados"]["modelo"] == hiperparametros["final"]["modelo"]


def test_hiperparametros_do_pipeline_batem_com_o_json(artefato, hiperparametros):
    final = hiperparametros["final"]
    params = artefato["pipeline"].get_params()
    for chave, valor in final["parametros_pipeline"].items():
        assert params[chave] == valor, chave
    assert artefato["metadados"]["hiperparametros"] == final["parametros"]
    assert artefato["metadados"]["atributos_sensiveis_usados"] == final["atributos_sensiveis_usados"]


def test_limiar_do_artefato_e_o_registrado(artefato):
    df = pd.read_csv(RESULTADOS / "limiar_custo.csv")
    assert df["escolhido"].sum() == 1
    assert df.loc[df["escolhido"], "limiar"].item() == pytest.approx(artefato["metadados"]["limiar_decisao"])
    sens = pd.read_csv(RESULTADOS / "limiar_sensibilidade_custo.csv")
    assert set(sens["razao_custo_fn_fp"]) == {1, 2, 3, 5, 10}


def test_avaliacao_teste_registrada_bate_com_o_artefato(artefato):
    registro = json.loads((RESULTADOS / "avaliacao_teste.json").read_text(encoding="utf-8"))
    assert registro["metricas"] == artefato["metricas"]
    assert {k: v for k, v in registro["metadados"].items() if k != "treinado_em"} == {
        k: v for k, v in artefato["metadados"].items() if k != "treinado_em"
    }


# ---------------------------------------------------------------- atributos sensíveis

def test_atributos_sensiveis_descartados(artefato):
    if artefato["metadados"]["atributos_sensiveis_usados"]:
        pytest.skip("o modelo de produção usa SEX, AGE e MARRIAGE (atributos_sensiveis_usados=true)")
    pipeline = artefato["pipeline"]
    transformadores = pipeline.named_steps["preprocessador"].transformers_
    usados = {c for nome, t, cols in transformadores if t != "drop" and nome != "remainder" for c in cols}
    descartados = {c for nome, t, cols in transformadores if t == "drop" for c in cols}
    assert usados.isdisjoint(SENSIVEIS)
    assert descartados == set(SENSIVEIS)

    # Variar só esses campos não muda o escore
    amostra = pd.read_csv(CSV_REAL).sample(n=200, random_state=1)
    amostra["EDUCATION"] = amostra["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X = amostra[FEATURES_ESPERADAS].copy()
    base = pipeline.predict_proba(X)[:, 1]
    rng = np.random.default_rng(3)
    variado = X.assign(SEX=rng.integers(1, 3, len(X)), AGE=rng.integers(18, 80, len(X)),
                       MARRIAGE=rng.integers(1, 4, len(X)))
    np.testing.assert_array_equal(pipeline.predict_proba(variado)[:, 1], base)


# ---------------------------------------------------------------- importância

def test_importancia_uma_linha_por_variavel_usada(artefato):
    df = pd.read_csv(RESULTADOS / "importancia_atributos.csv")
    esperadas = [c for c in FEATURES_ESPERADAS
                 if artefato["metadados"]["atributos_sensiveis_usados"] or c not in SENSIVEIS]
    assert sorted(df["variavel"]) == sorted(esperadas)
    assert df["variavel"].is_unique
    if df["medida_nativa"].iloc[0].startswith("feature_importances_"):
        # O CSV grava 6 casas decimais: a soma de ~20 valores arredondados difere de 1 em < 1e-5
        assert df["importancia_nativa"].sum() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------- risco

def test_invariante_classe_e_risco_alto_com_limiar_real(artefato):
    import main

    limiar = artefato["metadados"]["limiar_decisao"]
    df = main.limpar_categorias_uci(pd.read_csv(CSV_REAL).sample(n=1000, random_state=5))
    escores = list(artefato["pipeline"].predict_proba(df[FEATURES_ESPERADAS])[:, 1])
    escores += [i / 1000 for i in range(1001)] + [limiar, np.nextafter(limiar, 0)]
    faixas = set()
    for p in escores:
        classe, nivel, cor, _ = main.classificar_risco(float(p), limiar)
        assert (classe == 1) == (nivel == "Risco Alto") == (p >= limiar)
        faixas.add(nivel)
    assert faixas == {"Risco Baixo", "Risco Moderado", "Risco Alto"}


# ---------------------------------------------------------------- notebook

def test_notebook_executado_e_igual_ao_script():
    jupytext = pytest.importorskip("jupytext")
    nbformat = pytest.importorskip("nbformat")
    if not NOTEBOOK.is_file():
        pytest.skip("notebook não gerado")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    codigo_nb = [c for c in nb.cells if c.cell_type == "code"]
    assert sum(bool(c.outputs) for c in codigo_nb) >= len(codigo_nb) - 2, "notebook sem saídas"
    assert not [o for c in codigo_nb for o in c.outputs if o.output_type == "error"]
    assert sum("image/png" in o.get("data", {}) for c in codigo_nb for o in c.outputs) >= 8
    codigo_py = [c.source for c in jupytext.read(SCRIPT).cells if c.cell_type == "code"]
    assert [c.source for c in codigo_nb] == codigo_py, "notebook desatualizado: regenere a partir do .py"
