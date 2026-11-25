"""
Passo 3 — Verificações estáticas do front (frontend/index.html e frontend/app.js):
nenhum número do modelo escrito à mão, nenhum simulador local e campos de métricas
começando com "—" (preenchidos só com a resposta da API).
"""

import re

import pytest

from conftest import REPO_DIR

FRONTEND = REPO_DIR / "frontend"
INDEX = FRONTEND / "index.html"
APP_JS = FRONTEND / "app.js"

NUMEROS_ANTIGOS = [
    "79.5", "61.2", "53.8", "0.573", "3955", "698", "515", "812",
    "68.9", "64.5", "80.6", "61.8", "34.1", "0.439", "14.2",
]
TERMOS_PROIBIDOS = [
    "inferenciaCalibradaLocal", "pesos reais", "motor calibrado", "Vercel", "Random Forest (Balanced)",
]
# Elementos preenchidos só com dados da API (métricas, matriz, escore, rodapé e fatos do modelo)
CAMPOS_DO_MODELO = [
    "mRecall", "mAcc", "mAccSempre", "mPrec", "mF1", "mAuc", "mAucChute", "mPrAuc", "mTotal",
    "mcModelo", "mcVN", "mcFP", "mcFN", "mcVP",
    "probValue", "probCorte", "riskTitle", "riskClassLabel", "recText",
    "rodapeModelo", "rodapeCorte",
    "fatoModelo", "fatoHiperparametros", "fatoCorte", "fatoSensiveis", "fatoTreino", "fatoAucCv",
    "textoJustificativa", "textoAvaliacaoTeste", "textoAcuracia",
]


@pytest.fixture(scope="module")
def fontes():
    return {"index.html": INDEX.read_text(encoding="utf-8"), "app.js": APP_JS.read_text(encoding="utf-8")}


@pytest.mark.parametrize("numero", NUMEROS_ANTIGOS)
def test_nenhum_numero_antigo(fontes, numero):
    for nome, texto in fontes.items():
        assert numero not in texto, f"{numero} aparece em {nome}"


@pytest.mark.parametrize("termo", TERMOS_PROIBIDOS)
def test_nenhum_termo_do_simulador(fontes, termo):
    for nome, texto in fontes.items():
        assert termo.lower() not in texto.lower(), f"'{termo}' aparece em {nome}"


@pytest.mark.parametrize("campo", CAMPOS_DO_MODELO)
def test_campos_do_modelo_comecam_com_traco(fontes, campo):
    html = fontes["index.html"]
    m = re.search(rf'<(\w+)[^>]*\bid="{campo}"[^>]*>(.*?)</\1>', html, re.S)
    assert m, f"elemento #{campo} não encontrado"
    assert m.group(2).strip() == "—", f"#{campo} começa com {m.group(2).strip()!r}"


def test_matriz_e_tabela_sem_valores_iniciais(fontes):
    html = fontes["index.html"]
    corpo = re.search(r'<tbody id="comparacaoCorpo"[^>]*>(.*?)</tbody>', html, re.S).group(1)
    assert re.sub(r"<[^>]+>", "", corpo).strip() == "—"


def test_app_js_carregado_por_script_src(fontes):
    assert '<script src="app.js"></script>' in fontes["index.html"]
    assert "module.exports" in fontes["app.js"]


def test_index_nao_tem_percentuais_ou_decimais_de_modelo(fontes):
    """Fora dos textos de metodologia, nenhum percentual com casa decimal (ex.: 22.12%)."""
    html = fontes["index.html"]
    assert not re.findall(r"\d+[.,]\d+\s*%", html)


def test_aviso_dos_atributos_sensiveis(fontes):
    html = fontes["index.html"].lower()
    assert "não são usados pelo modelo; mantidos por compatibilidade da api" in html
    assert html.count('class="aviso-sensivel') == 3
