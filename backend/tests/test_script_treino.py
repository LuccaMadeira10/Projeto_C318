"""
Testes slow: executam o script de treino em subprocess, no modo rápido (C318_MODO_RAPIDO=1:
grades mínimas e 3 folds).

- Execução isolada: cwd externo ao repositório, MODEL_OUTPUT_PATH e C318_RESULTADOS_DIR
  temporários, MPLBACKEND=Agg. O artefato e os resultados versionados não podem mudar.
- Sem vazamento: um CSV igual ao original, mas com os valores de cada coluna preditora
  embaralhados SÓ entre as linhas do conjunto de teste. Se nada do teste influencia as escolhas,
  modelo, hiperparâmetros, atributos sensíveis e limiar saem idênticos; a AUC de teste despenca,
  o que prova que a perturbação chegou ao script.

Para pular: python -m pytest -m "not slow"
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import BACKEND_DIR, CSV_REAL, REPO_DIR, verificar_metricas_recomputadas
from test_contrato_caracterizacao import FEATURES_ESPERADAS

pytestmark = pytest.mark.slow

ARTEFATO_REPO = BACKEND_DIR / "modelo_credito.joblib"
RESULTADOS_REPO = REPO_DIR / "resultados"
ALVO = "default.payment.next.month"
ARQUIVOS_RESULTADOS = [
    "comparacao_modelos_cv.csv", "comparacao_modelos_cv_inicial.csv", "grid_search_resultados.csv",
    "melhores_hiperparametros.json", "sensiveis_com_vs_sem.csv", "limiar_custo.csv",
    "limiar_sensibilidade_custo.csv", "avaliacao_teste.csv", "avaliacao_teste.json",
    "importancia_atributos.csv",
    "eda_resumo.png", "eda_correlacao.png", "eda_status_pagamento.png", "eda_uso_limite.png",
    "comparacao_modelos_cv.png", "importancia_atributos.png", "curva_cortes.png",
    "matriz_confusao_final.png",
]


def _hashes_versionados() -> dict:
    arquivos = [ARTEFATO_REPO, *sorted(RESULTADOS_REPO.glob("*"))]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in arquivos if p.is_file()}


def _rodar_script(pasta, csv=None) -> dict:
    env = dict(
        os.environ,
        C318_MODO_RAPIDO="1",
        MODEL_OUTPUT_PATH=str(pasta / "modelo.joblib"),
        C318_RESULTADOS_DIR=str(pasta / "resultados"),
        MPLBACKEND="Agg",
        PYTHONIOENCODING="utf-8",
    )
    env.pop("C318_CSV_PATH", None)
    if csv is not None:
        env["C318_CSV_PATH"] = str(csv)
    proc = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "analise_e_modelagem_c318.py")],
        cwd=pasta, env=env, capture_output=True, text=True, encoding="utf-8", timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-4000:]
    return {"artefato": pasta / "modelo.joblib", "resultados": pasta / "resultados", "stdout": proc.stdout}


def _indices_teste(df: pd.DataFrame):
    """Mesmo train_test_split do script: a divisão estratificada depende só do alvo."""
    from sklearn.model_selection import train_test_split

    _, idx_teste = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df[ALVO])
    return idx_teste


@pytest.fixture(scope="module")
def execucao_original(tmp_path_factory):
    antes = _hashes_versionados()
    resultado = _rodar_script(tmp_path_factory.mktemp("treino_externo"))
    resultado["hashes_antes"], resultado["hashes_depois"] = antes, _hashes_versionados()
    return resultado


@pytest.fixture(scope="module")
def execucao_teste_embaralhado(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("treino_teste_embaralhado")
    original = pd.read_csv(CSV_REAL)
    alterado = original.copy()
    idx_teste = _indices_teste(original)
    rng = np.random.default_rng(2026)
    for coluna in FEATURES_ESPERADAS:
        alterado.loc[idx_teste, coluna] = rng.permutation(alterado.loc[idx_teste, coluna].to_numpy())

    # Sanidade da perturbação: treino idêntico, teste alterado, alvo intacto
    treino = original.index.difference(idx_teste)
    pd.testing.assert_frame_equal(original.loc[treino], alterado.loc[treino])
    assert (original.loc[idx_teste, FEATURES_ESPERADAS] != alterado.loc[idx_teste, FEATURES_ESPERADAS]).any(axis=None)
    assert original[ALVO].equals(alterado[ALVO])

    csv = pasta / "UCI_teste_embaralhado.csv"
    alterado.to_csv(csv, index=False)
    resultado = _rodar_script(pasta, csv=csv)
    resultado["csv"] = csv
    return resultado


def test_execucao_isolada_gera_artefato_e_resultados(execucao_original):
    import joblib
    import main
    import sklearn

    assert execucao_original["hashes_antes"] == execucao_original["hashes_depois"], (
        "o script alterou arquivos versionados"
    )
    artefato = main.carregar_modelo(execucao_original["artefato"])  # contrato completo
    meta = artefato["metadados"]
    assert meta["sklearn_version"] == sklearn.__version__
    assert 0 < meta["limiar_decisao"] < 1
    assert isinstance(meta["atributos_sensiveis_usados"], bool)
    assert meta["csv_origem"] == "data/UCI_Credit_Card.csv"
    assert meta["modo_rapido"] is True
    assert meta["treinado_em"]
    assert artefato["metricas"]["total_amostras_teste"] == 6000
    assert list(joblib.load(execucao_original["artefato"])["pipeline"].feature_names_in_) == FEATURES_ESPERADAS
    for nome in ARQUIVOS_RESULTADOS:
        caminho = execucao_original["resultados"] / nome
        assert caminho.is_file() and caminho.stat().st_size > 0, nome


def test_numeros_batem_com_o_codigo(execucao_original):
    """Métricas do artefato gerado = recomputação independente (split, pipeline e limiar)."""
    verificar_metricas_recomputadas(execucao_original["artefato"])


def _sem_tempos(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if "tempo" in c or "time" in c])


def test_sem_vazamento_do_conjunto_de_teste(execucao_original, execucao_teste_embaralhado):
    import joblib

    a = joblib.load(execucao_original["artefato"])
    b = joblib.load(execucao_teste_embaralhado["artefato"])

    # 1) A perturbação chegou ao script: CSV lido é o alterado e a AUC de teste despenca
    assert b["metadados"]["csv_origem"] == str(execucao_teste_embaralhado["csv"].resolve())
    assert b["metadados"]["csv_origem"] != a["metadados"]["csv_origem"]
    queda = a["metricas"]["roc_auc"] - b["metricas"]["roc_auc"]
    assert queda >= 0.10, (a["metricas"]["roc_auc"], b["metricas"]["roc_auc"])

    # 2) Decisões idênticas: modelo, hiperparâmetros, atributos sensíveis e limiar
    for chave in ("modelo", "hiperparametros", "atributos_sensiveis_usados", "limiar_decisao"):
        assert a["metadados"][chave] == b["metadados"][chave], chave
    assert a["pipeline"].get_params(deep=True).keys() == b["pipeline"].get_params(deep=True).keys()
    ra, rb = execucao_original["resultados"], execucao_teste_embaralhado["resultados"]
    assert json.loads((ra / "melhores_hiperparametros.json").read_text(encoding="utf-8")) == json.loads(
        (rb / "melhores_hiperparametros.json").read_text(encoding="utf-8")
    )

    # 3) Toda a etapa de seleção (CV, grade, sensíveis, cortes) é numericamente idêntica
    for nome in ("comparacao_modelos_cv.csv", "comparacao_modelos_cv_inicial.csv",
                 "grid_search_resultados.csv", "sensiveis_com_vs_sem.csv", "limiar_custo.csv",
                 "limiar_sensibilidade_custo.csv"):
        pd.testing.assert_frame_equal(_sem_tempos(pd.read_csv(ra / nome)), _sem_tempos(pd.read_csv(rb / nome)),
                                      obj=nome)
