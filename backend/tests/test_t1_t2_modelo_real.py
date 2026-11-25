"""
T1/T2 — A API apenas carrega o artefato gerado pelo script de treino.

- Sem modelo válido: /predict e /metrics retornam 503 e GET / informa modelo_carregado=false.
- Não existe fallback sintético nem métricas escritas à mão.
- A API nunca treina nem grava o artefato.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BACKEND_DIR, CENARIO_A, CENARIO_B, VARIANTES_DEFEITUOSAS, abrir_cliente

# Métricas que estavam escritas à mão no main.py antigo
NUMEROS_FICTICIOS = ("0.7945", "0.6120", "0.5726")


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


@pytest.mark.parametrize("variante", VARIANTES_DEFEITUOSAS)
def test_sem_modelo_da_503(monkeypatch, fabrica_artefato, variante):
    caminho = fabrica_artefato(variante)
    _, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        raiz = c.get("/")
        assert raiz.status_code == 200
        dados = raiz.json()
        assert dados["modelo_carregado"] is False
        assert dados["status"] == "modelo_indisponivel"
        assert dados["modelo_ativo"] == "indisponivel"

        for resp in (c.post("/predict", json=CENARIO_B), c.get("/metrics")):
            assert resp.status_code == 503, resp.text
            detalhe = resp.json()["detail"]
            assert isinstance(detalhe, str)
            assert "indisponível" in detalhe
            assert "analise_e_modelagem_c318.py" in detalhe
            for numero in NUMEROS_FICTICIOS:
                assert numero not in resp.text


def test_falha_de_carregamento_gera_log_error(monkeypatch, fabrica_artefato, caplog):
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("corrompido"))
    with caplog.at_level("ERROR"), tc:
        pass
    assert any(r.levelname == "ERROR" and "modelo" in r.getMessage().lower() for r in caplog.records)


def test_artefato_valido_da_fabrica_carrega(monkeypatch, fabrica_artefato):
    """Controle positivo: um artefato no contrato é carregado e servido."""
    caminho = fabrica_artefato("valido")
    _, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        assert c.get("/").json()["modelo_carregado"] is True
        assert c.get("/").json()["status"] == "online"
        assert c.post("/predict", json=CENARIO_A).status_code == 200
        assert c.get("/metrics").json()["algoritmo"] == "RandomForestClassifier (artefato de teste)"


def test_artefato_sem_metadados_usa_padroes(monkeypatch, fabrica_artefato):
    caminho = fabrica_artefato("valido", metadados=False)
    main, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        assert c.get("/").json()["modelo_carregado"] is True
        assert c.post("/predict", json=CENARIO_B).status_code == 200
        assert main.obter_limiar() == 0.5


def test_modelo_carregado_no_startup_e_nao_no_import(monkeypatch, fabrica_artefato):
    """MODEL_PATH é lido no startup: alterar a variável depois do import tem efeito."""
    main, _ = abrir_cliente(monkeypatch, fabrica_artefato("ausente"))
    assert main.STATE["pipeline"] is None
    monkeypatch.setenv("MODEL_PATH", str(fabrica_artefato("valido")))
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        assert c.get("/").json()["modelo_carregado"] is True


def test_nao_existe_fallback_sintetico(tmp_path):
    """
    Cópia isolada do main.py, sem artefato e sem CSV em nenhum caminho candidato:
    nenhum modelo pode ser inventado.
    """
    pasta = tmp_path / "isolado" / "backend"
    pasta.mkdir(parents=True)
    shutil.copy(BACKEND_DIR / "main.py", pasta / "main.py")
    codigo = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "import main\n"
        "with TestClient(main.app) as c:\n"
        "    r = c.get('/'); m = c.get('/metrics'); p = c.post('/predict', json=%r)\n"
        "    d = c.get('/dataset-check')\n"
        "    print(json.dumps({'raiz': r.json(), 'metrics': m.status_code, 'predict': p.status_code,"
        " 'dataset': d.json(), 'texto': m.text + p.text}))\n"
    ) % CENARIO_B
    env = dict(os.environ, MODEL_PATH=str(tmp_path / "nao_existe.joblib"))
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", codigo], cwd=pasta, env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    saida = json.loads(proc.stdout.strip().splitlines()[-1])
    assert saida["dataset"]["arquivo_encontrado"] is False
    assert saida["raiz"].get("modelo_carregado") is False
    assert saida["metrics"] == 503
    assert saida["predict"] == 503
    for numero in NUMEROS_FICTICIOS:
        assert numero not in saida["texto"]
    assert not (tmp_path / "nao_existe.joblib").exists()


def test_main_nao_treina_nem_inventa():
    fonte = (BACKEND_DIR / "main.py").read_text(encoding="utf-8")
    proibidos = [
        ".fit(",
        "train_test_split",
        "RandomForestClassifier",
        "np.random",
        "default_rng",
        "joblib.dump",
        "0.7945",
        "0.6120",
        "0.5726",
    ]
    encontrados = [p for p in proibidos if p in fonte]
    assert encontrados == [], f"main.py contém trechos proibidos: {encontrados}"


def test_api_nao_sobrescreve_artefato(monkeypatch, modelo_real, fabrica_artefato):
    antes = _sha256(modelo_real)
    mtime = modelo_real.stat().st_mtime_ns
    _, tc = abrir_cliente(monkeypatch, modelo_real)
    with tc as c:
        c.get("/")
        c.get("/metrics")
        c.post("/predict", json=CENARIO_A)
    assert _sha256(modelo_real) == antes
    assert modelo_real.stat().st_mtime_ns == mtime

    ausente = fabrica_artefato("ausente")
    _, tc = abrir_cliente(monkeypatch, ausente)
    with tc as c:
        c.get("/")
        c.post("/predict", json=CENARIO_A)
    assert not ausente.exists(), "a API não pode criar o artefato"
