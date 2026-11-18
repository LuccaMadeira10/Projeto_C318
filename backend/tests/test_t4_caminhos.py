"""
T4 — Caminhos independentes do diretório de execução.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BACKEND_DIR, CENARIO_A, REPO_DIR

CODIGO_SUBPROCESS = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])  # equivalente a uvicorn --app-dir\n"
    "from fastapi.testclient import TestClient\n"
    "import main\n"
    "with TestClient(main.app) as c:\n"
    "    print(json.dumps({\n"
    "        'raiz': c.get('/').json(),\n"
    "        'metrics': c.get('/metrics').status_code,\n"
    "        'predict': c.post('/predict', json=json.loads(sys.argv[2])).status_code,\n"
    "        'dataset': c.get('/dataset-check').json(),\n"
    "    }))\n"
)


@pytest.mark.parametrize("onde", ["externo", "raiz", "backend"])
def test_caminhos_independem_do_cwd(tmp_path, modelo_real, onde):
    cwd = {"externo": tmp_path, "raiz": REPO_DIR, "backend": BACKEND_DIR}[onde]
    env = dict(os.environ)
    env.pop("MODEL_PATH", None)
    env.pop("PYTHONPATH", None)
    if modelo_real != BACKEND_DIR / "modelo_credito.joblib":
        env["MODEL_PATH"] = str(modelo_real)  # override explícito via C318_MODELO_REAL
    proc = subprocess.run(
        [sys.executable, "-c", CODIGO_SUBPROCESS, str(BACKEND_DIR), json.dumps(CENARIO_A)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    saida = json.loads(proc.stdout.strip().splitlines()[-1])
    assert saida["raiz"]["modelo_carregado"] is True
    assert saida["metrics"] == 200
    assert saida["predict"] == 200
    assert saida["dataset"]["arquivo_encontrado"] is True
    assert saida["dataset"]["total_clientes_linhas"] == 30000


def test_model_path_padrao_e_sobrescrita(monkeypatch):
    import main

    monkeypatch.delenv("MODEL_PATH", raising=False)
    assert os.path.samefile(main.caminho_modelo(), BACKEND_DIR / "modelo_credito.joblib")
    monkeypatch.setenv("MODEL_PATH", "/outro/caminho.joblib")
    assert Path(main.caminho_modelo()) == Path("/outro/caminho.joblib")


def test_csv_candidatos_relativos_ao_arquivo():
    import main

    esperados = [
        BACKEND_DIR / "UCI_Credit_Card.csv",
        REPO_DIR / "data" / "UCI_Credit_Card.csv",
        BACKEND_DIR / "data" / "UCI_Credit_Card.csv",
    ]
    assert [os.path.abspath(p) for p in main.CSV_PATHS] == [os.path.abspath(p) for p in esperados]
