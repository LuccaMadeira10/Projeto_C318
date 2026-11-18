"""
T3 — Reprodutibilidade: dependências fixadas e artefato compatível com elas.
"""

import re
import warnings

import pandas as pd

from conftest import BACKEND_DIR, CENARIO_B
from test_contrato_caracterizacao import FEATURES_ESPERADAS

ARQUIVOS_REQUIREMENTS = ["requirements.txt", "requirements-treino.txt", "requirements-dev.txt"]
RUNTIME = {"fastapi", "uvicorn", "scikit-learn", "pandas", "numpy", "joblib", "pydantic"}
INCLUSOES = {
    "requirements-treino.txt": "-r requirements.txt",
    "requirements-dev.txt": "-r requirements-treino.txt",
}
PACOTES_EXTRAS = {
    "requirements-treino.txt": {"matplotlib", "seaborn"},
    "requirements-dev.txt": {"pytest", "httpx"},
}


def _ler(nome):
    return (BACKEND_DIR / nome).read_text(encoding="utf-8").splitlines()


def _pacotes(nome):
    """Retorna {pacote: versão} das linhas de requisito (ignora comentários e -r)."""
    pacotes = {}
    for linha in _ler(nome):
        linha = linha.split("#", 1)[0].strip()
        if not linha or linha.startswith("-r"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[0-9][0-9A-Za-z.\-+]*", linha), (
            f"{nome}: '{linha}' não está fixado com =="
        )
        nome_pacote, versao = linha.split("==")
        pacotes[nome_pacote.lower()] = versao
    return pacotes


def test_requirements_todos_fixados():
    for nome in ARQUIVOS_REQUIREMENTS:
        linhas = _ler(nome)
        assert re.match(r"#.*Python 3\.\d+", linhas[0]), f"{nome}: 1ª linha deve registrar a versão do Python"
        _pacotes(nome)  # valida que toda linha de requisito usa ==
    assert set(_pacotes("requirements.txt")) == RUNTIME
    for nome, inclusao in INCLUSOES.items():
        assert inclusao in [l.strip() for l in _ler(nome)]
        assert set(_pacotes(nome)) == PACOTES_EXTRAS[nome]


def test_versao_sklearn_coerente():
    import joblib
    import sklearn

    fixada = _pacotes("requirements.txt")["scikit-learn"]
    metadados = joblib.load(BACKEND_DIR / "modelo_credito.joblib").get("metadados", {})
    assert sklearn.__version__ == fixada == metadados.get("sklearn_version")


def test_versoes_instaladas_iguais_as_fixadas():
    from importlib.metadata import version

    for nome in ARQUIVOS_REQUIREMENTS:
        for pacote, versao in _pacotes(nome).items():
            assert version(pacote) == versao, f"{pacote}: instalado {version(pacote)}, fixado {versao}"


def test_artefato_valido_do_repo():
    """O artefato versionado abre sem nenhum warning (equivalente a -W error) e segue o contrato."""
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        artefato = joblib.load(BACKEND_DIR / "modelo_credito.joblib")
    assert isinstance(artefato, dict)
    assert {"pipeline", "metricas", "metadados"} <= set(artefato)
    pipeline = artefato["pipeline"]
    assert list(pipeline.feature_names_in_) == FEATURES_ESPERADAS
    completo = dict(
        CENARIO_B, PAY_2=1, PAY_3=1, PAY_4=0, PAY_5=0, PAY_6=0,
        BILL_AMT2=29500, BILL_AMT3=29500, BILL_AMT4=29500, BILL_AMT5=29500, BILL_AMT6=29500,
        PAY_AMT2=0, PAY_AMT3=0, PAY_AMT4=0, PAY_AMT5=0, PAY_AMT6=0,
    )
    proba = pipeline.predict_proba(pd.DataFrame([completo])[FEATURES_ESPERADAS])
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-9
