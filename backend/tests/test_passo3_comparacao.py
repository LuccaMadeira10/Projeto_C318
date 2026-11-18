"""
Passo 3, T2 — GET /comparacao-modelos: métricas de comparacao_modelos_cv.csv e
hiperparâmetros de melhores_hiperparametros.json, com o flag de coerência com o modelo
carregado e 503 quando os arquivos faltam ou estão ilegíveis.
"""

import csv
import json
import shutil

import pytest

from conftest import REPO_DIR, abrir_cliente

RESULTADOS = REPO_DIR / "resultados"
CSV_CV = RESULTADOS / "comparacao_modelos_cv.csv"
JSON_HIPER = RESULTADOS / "melhores_hiperparametros.json"
COLUNAS = [
    "roc_auc_media", "roc_auc_dp", "pr_auc_media", "pr_auc_dp", "recall_media", "recall_dp",
    "precisao_media", "precisao_dp", "f1_media", "f1_dp",
]


@pytest.fixture
def cliente_resultados_reais(monkeypatch, cliente):
    monkeypatch.delenv("C318_RESULTADOS_DIR", raising=False)
    return cliente


def _linhas_csv():
    with open(CSV_CV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_comparacao_metricas_iguais_ao_csv(cliente_resultados_reais):
    resp = cliente_resultados_reais.get("/comparacao-modelos")
    assert resp.status_code == 200, resp.text
    dados = resp.json()
    linhas = _linhas_csv()
    assert len(dados["modelos"]) == len(linhas) == 5
    for linha, modelo in zip(linhas, dados["modelos"]):
        assert modelo["modelo"] == linha["modelo"]
        for coluna in COLUNAS:
            assert modelo[coluna] == float(linha[coluna]), (linha["modelo"], coluna)
        assert modelo["vencedor"] == (linha["vencedor"] == "True")
    vencedores = [m["modelo"] for m in dados["modelos"] if m["vencedor"]]
    assert len(vencedores) == 1
    assert dados["vencedor"] == vencedores[0]


def test_comparacao_hiperparametros_iguais_ao_json(cliente_resultados_reais):
    dados = cliente_resultados_reais.get("/comparacao-modelos").json()
    hiper = json.loads(JSON_HIPER.read_text(encoding="utf-8"))
    sem_hiper = []
    for modelo in dados["modelos"]:
        info = hiper["modelos"].get(modelo["modelo"])
        if info is None:
            sem_hiper.append(modelo["modelo"])
            assert modelo["melhores_parametros"] is None and modelo["grade"] is None
        else:
            assert modelo["melhores_parametros"] == info["melhores_parametros"]
            assert modelo["grade"] == info["grade"]
    assert len(sem_hiper) == 1 and sem_hiper[0].startswith("Chute")
    assert dados["final"] == hiper["final"]
    assert dados["criterio"] == hiper["criterio"]
    assert dados["n_folds"] == hiper["n_folds"]


def test_coerente_com_artefato_real(cliente_resultados_reais):
    dados = cliente_resultados_reais.get("/comparacao-modelos").json()
    assert dados["modelo_carregado"] == dados["vencedor"]
    assert dados["coerente_com_modelo_carregado"] is True


def test_incoerente_com_artefato_de_outro_modelo(monkeypatch, fabrica_artefato):
    monkeypatch.delenv("C318_RESULTADOS_DIR", raising=False)
    caminho = fabrica_artefato("valido", metadados_extra={"modelo": "KNN"})
    _, tc = abrir_cliente(monkeypatch, caminho)
    with tc as c:
        dados = c.get("/comparacao-modelos").json()
    assert dados["modelo_carregado"] == "KNN"
    assert dados["coerente_com_modelo_carregado"] is False


def test_incoerente_sem_modelo_carregado(monkeypatch, fabrica_artefato):
    monkeypatch.delenv("C318_RESULTADOS_DIR", raising=False)
    _, tc = abrir_cliente(monkeypatch, fabrica_artefato("ausente"))
    with tc as c:
        resp = c.get("/comparacao-modelos")
    assert resp.status_code == 200
    assert resp.json()["modelo_carregado"] is None
    assert resp.json()["coerente_com_modelo_carregado"] is False


def test_incoerente_quando_json_diverge_do_csv(monkeypatch, cliente, tmp_path):
    shutil.copy(CSV_CV, tmp_path / CSV_CV.name)
    hiper = json.loads(JSON_HIPER.read_text(encoding="utf-8"))
    hiper["final"]["modelo"] = "KNN"
    (tmp_path / JSON_HIPER.name).write_text(json.dumps(hiper), encoding="utf-8")
    monkeypatch.setenv("C318_RESULTADOS_DIR", str(tmp_path))
    assert cliente.get("/comparacao-modelos").json()["coerente_com_modelo_carregado"] is False


@pytest.mark.parametrize("presentes", [[], ["csv"], ["json"]], ids=["nenhum", "so_csv", "so_json"])
def test_503_sem_arquivos(monkeypatch, cliente, tmp_path, presentes):
    if "csv" in presentes:
        shutil.copy(CSV_CV, tmp_path / CSV_CV.name)
    if "json" in presentes:
        shutil.copy(JSON_HIPER, tmp_path / JSON_HIPER.name)
    monkeypatch.setenv("C318_RESULTADOS_DIR", str(tmp_path))
    resp = cliente.get("/comparacao-modelos")
    assert resp.status_code == 503
    detalhe = resp.json()["detail"]
    assert isinstance(detalhe, str) and "não encontrado" in detalhe
    assert "analise_e_modelagem_c318.py" in detalhe
    assert str(tmp_path) not in detalhe  # sem caminho absoluto do servidor


@pytest.mark.parametrize("defeito", ["json_invalido", "csv_sem_coluna", "csv_nao_numerico",
                                     "csv_dois_vencedores", "json_sem_final"])
def test_503_arquivos_ilegiveis(monkeypatch, cliente, tmp_path, defeito):
    texto_csv = CSV_CV.read_text(encoding="utf-8")
    texto_json = JSON_HIPER.read_text(encoding="utf-8")
    if defeito == "json_invalido":
        texto_json = texto_json[: len(texto_json) // 2]
    elif defeito == "csv_sem_coluna":
        texto_csv = texto_csv.replace("roc_auc_media", "auc", 1)
    elif defeito == "csv_nao_numerico":
        texto_csv = texto_csv.replace("0.785697", "abc", 1)
    elif defeito == "csv_dois_vencedores":
        texto_csv = texto_csv.replace(",False\n", ",True\n", 1)
    elif defeito == "json_sem_final":
        dados = json.loads(texto_json)
        dados.pop("final")
        texto_json = json.dumps(dados)
    (tmp_path / CSV_CV.name).write_text(texto_csv, encoding="utf-8")
    (tmp_path / JSON_HIPER.name).write_text(texto_json, encoding="utf-8")
    monkeypatch.setenv("C318_RESULTADOS_DIR", str(tmp_path))
    resp = cliente.get("/comparacao-modelos")
    assert resp.status_code == 503, resp.text
    assert "Comparação dos modelos indisponível" in resp.json()["detail"]


def test_raiz_lista_a_rota(cliente):
    assert "/comparacao-modelos" in cliente.get("/").json()["rotas_disponiveis"]
