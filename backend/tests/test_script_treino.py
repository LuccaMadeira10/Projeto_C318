"""
Testes slow: executam o script de treino completo em subprocess.

- cwd externo ao repositório (prova que os caminhos independem do diretório atual);
- MODEL_OUTPUT_PATH temporário (o artefato versionado nunca é tocado);
- MPLBACKEND=Agg (sem janelas de gráfico).

Para pular: python -m pytest -m "not slow"
"""

import hashlib
import os
import subprocess
import sys

import pytest

from conftest import BACKEND_DIR

pytestmark = pytest.mark.slow

ARTEFATO_REPO = BACKEND_DIR / "modelo_credito.joblib"


def _sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def execucao_script(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("treino_externo")
    saida = pasta / "modelo_gerado.joblib"
    sha_antes = _sha256(ARTEFATO_REPO) if ARTEFATO_REPO.exists() else None
    env = dict(os.environ, MODEL_OUTPUT_PATH=str(saida), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "analise_e_modelagem_c318.py")],
        cwd=pasta,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    sha_depois = _sha256(ARTEFATO_REPO) if ARTEFATO_REPO.exists() else None
    return {"proc": proc, "saida": saida, "sha_antes": sha_antes, "sha_depois": sha_depois}


def test_script_treino_gera_artefato(execucao_script):
    import main
    import sklearn

    proc = execucao_script["proc"]
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert execucao_script["saida"].is_file()
    # O artefato versionado não pode ser alterado pelo teste
    assert execucao_script["sha_antes"] == execucao_script["sha_depois"]

    artefato = main.carregar_modelo(execucao_script["saida"])  # valida o contrato completo
    meta = artefato["metadados"]
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["limiar_decisao"] == 0.5
    assert meta["csv_origem"].endswith("UCI_Credit_Card.csv")
    assert meta["treinado_em"]
    assert artefato["metricas"]["total_amostras_teste"] == 6000


def test_numeros_batem_com_o_codigo(execucao_script):
    """Referência da revisão (código executado): acurácia ~0,777, recall ~0,593, F1 ~0,541."""
    import main

    assert execucao_script["proc"].returncode == 0
    m = main.carregar_modelo(execucao_script["saida"])["metricas"]
    assert m["acuracia"] == pytest.approx(0.777, abs=0.01)
    assert m["recall_sensibilidade"] == pytest.approx(0.593, abs=0.01)
    assert m["f1_score"] == pytest.approx(0.541, abs=0.01)
