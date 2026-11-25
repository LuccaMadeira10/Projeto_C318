// Testes da lógica do front (frontend/app.js), sem dependências: node --test frontend/tests
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const app = require(path.join(__dirname, "..", "app.js"));

// Resposta de exemplo no formato de GET /comparacao-modelos (valores fictícios de teste)
const COMPARACAO = {
  criterio: "roc_auc",
  n_folds: 5,
  vencedor: "Floresta",
  final: { modelo: "Floresta", parametros: { n_estimators: 300 } },
  modelo_carregado: "Floresta",
  coerente_com_modelo_carregado: true,
  modelos: [
    { modelo: "Chute (referência)", roc_auc_media: 0.5, roc_auc_dp: 0, pr_auc_media: 0.2, pr_auc_dp: 0.001,
      recall_media: 0, recall_dp: 0, precisao_media: 0, precisao_dp: 0, f1_media: 0, f1_dp: 0,
      vencedor: false, melhores_parametros: null, grade: null },
    { modelo: "KNN", roc_auc_media: 0.7412, roc_auc_dp: 0.0123, pr_auc_media: 0.47, pr_auc_dp: 0.02,
      recall_media: 0.2468, recall_dp: 0.01, precisao_media: 0.6543, precisao_dp: 0.03, f1_media: 0.3579, f1_dp: 0.01,
      vencedor: false, melhores_parametros: { n_neighbors: 101, weights: "uniform" },
      grade: { n_neighbors: [11, 101], weights: ["uniform", "distance"] } },
    { modelo: "Floresta", roc_auc_media: 0.7891, roc_auc_dp: 0.0047, pr_auc_media: 0.55, pr_auc_dp: 0.01,
      recall_media: 0.6234, recall_dp: 0.01, precisao_media: 0.4876, precisao_dp: 0.01, f1_media: 0.5472, f1_dp: 0.01,
      vencedor: true, melhores_parametros: { max_depth: null, n_estimators: 300 },
      grade: { max_depth: [8, null], n_estimators: [100, 300] } },
    { modelo: "Árvore", roc_auc_media: 0.7666, roc_auc_dp: 0.0061, pr_auc_media: 0.52, pr_auc_dp: 0.01,
      recall_media: 0.6111, recall_dp: 0.02, precisao_media: 0.4444, precisao_dp: 0.02, f1_media: 0.5147, f1_dp: 0.01,
      vencedor: false, melhores_parametros: { max_depth: 6 }, grade: { max_depth: [4, 6] } },
  ],
};

function metricas(VP, VN, FP, FN, acuracia) {
  return { acuracia, matriz_confusao: { VP, VN, FP, FN }, total_amostras_teste: VP + VN + FP + FN };
}

// ---------------------------------------------------------------------------
// estadoDaApi
// ---------------------------------------------------------------------------

test("estadoDaApi: conectado quando modelo_carregado é true", () => {
  const e = app.estadoDaApi({ status: "online", modelo_carregado: true, modelo_ativo: "Artefato x" });
  assert.equal(e.estado, "conectado");
  assert.equal(e.titulo, "Conectado");
  assert.equal(e.exibeResultados, true);
  assert.deepEqual(e.comandos, []);
});

test("estadoDaApi: API sem modelo mostra a mensagem do 503 e como gerar o modelo", () => {
  const msg = "Modelo preditivo indisponível: arquivo do modelo não encontrado.";
  const e = app.estadoDaApi({ status: "modelo_indisponivel", modelo_carregado: false }, msg);
  assert.equal(e.estado, "sem_modelo");
  assert.equal(e.titulo, "API sem modelo");
  assert.equal(e.exibeResultados, false);
  assert.ok(e.mensagem.includes(msg));
  assert.ok(e.comandos.some((c) => c.includes("analise_e_modelagem_c318.py")));
});

test("estadoDaApi: sem o campo modelo_carregado não exibe resultados", () => {
  assert.equal(app.estadoDaApi({ status: "online" }).estado, "sem_modelo");
});

test("estadoDaApi: API desligada quando não há resposta, com comandos para subir a API", () => {
  for (const raiz of [null, undefined]) {
    const e = app.estadoDaApi(raiz, null, "http://127.0.0.1:8000");
    assert.equal(e.estado, "desligada");
    assert.equal(e.titulo, "API desligada");
    assert.equal(e.exibeResultados, false);
    assert.ok(e.mensagem.includes("http://127.0.0.1:8000"));
    assert.ok(e.comandos.some((c) => c.includes("uvicorn main:app")));
  }
});

// ---------------------------------------------------------------------------
// lerErro
// ---------------------------------------------------------------------------

test("lerErro: lista do Pydantic (422)", () => {
  const corpo = { detail: [
    { loc: ["body", "AGE"], msg: "Input should be greater than or equal to 18", type: "greater_than_equal" },
    { loc: ["body", "LIMIT_BAL"], msg: "Field required", type: "missing" },
  ] };
  assert.equal(app.lerErro(422, corpo),
    "AGE: Input should be greater than or equal to 18; LIMIT_BAL: Field required");
});

test("lerErro: objeto {mensagem, campos_faltantes} do modo estrito (422)", () => {
  const corpo = { detail: { mensagem: "Modo estrito: informe os meses 2 a 6.", campos_faltantes: ["PAY_4", "PAY_5"] } };
  assert.equal(app.lerErro(422, corpo), "Modo estrito: informe os meses 2 a 6. Campos faltantes: PAY_4, PAY_5.");
});

test("lerErro: texto (503)", () => {
  const corpo = { detail: "Modelo preditivo indisponível: modelo não carregado." };
  assert.equal(app.lerErro(503, corpo), corpo.detail);
});

test("lerErro: corpo vazio ou desconhecido cai no código HTTP", () => {
  assert.equal(app.lerErro(500, null), "HTTP 500");
  assert.equal(app.lerErro(502, {}), "HTTP 502");
});

// ---------------------------------------------------------------------------
// Formatação
// ---------------------------------------------------------------------------

test("formatação de números em padrão brasileiro e '—' para ausentes", () => {
  assert.equal(app.formatarNumero(0.78912, 3), "0,789");
  assert.equal(app.formatarPercentual(0.6234), "62,3%");
  assert.equal(app.formatarMediaDp(0.7891, 0.0047), "0,789 ± 0,005");
  assert.equal(app.formatarCorte(0.5), "0,50");
  assert.equal(app.formatarCorte(0.355), "0,355");
  assert.equal(app.formatarPercentualApi("49.9%"), "49,9%");
  assert.equal(app.formatarParametros({ max_depth: null, n_estimators: 300 }), "max_depth = None · n_estimators = 300");
  assert.equal(app.formatarData("2026-11-18T19:35:19+00:00"), "18/11/2026 19:35 (UTC+00:00)");
  for (const v of [null, undefined, NaN, "0.5"]) {
    assert.equal(app.formatarNumero(v), "—");
    assert.equal(app.formatarPercentual(v), "—");
  }
  assert.equal(app.formatarCorte(null), "—");
  assert.equal(app.formatarParametros(null), "—");
});

// ---------------------------------------------------------------------------
// montarLinhasComparacao
// ---------------------------------------------------------------------------

test("montarLinhasComparacao: todos os modelos, vencedor destacado, AUC ± desvio formatada", () => {
  const linhas = app.montarLinhasComparacao(COMPARACAO);
  assert.equal(linhas.length, 4);
  assert.deepEqual(linhas.map((l) => l.modelo), ["Floresta", "Árvore", "KNN", "Chute (referência)"]);
  const vencedores = linhas.filter((l) => l.vencedor);
  assert.equal(vencedores.length, 1);
  assert.equal(vencedores[0].modelo, "Floresta");
  assert.equal(vencedores[0].auc, "0,789 ± 0,005");
  assert.equal(vencedores[0].recall, "62,3%");
  assert.equal(vencedores[0].precisao, "48,8%");
  assert.equal(vencedores[0].f1, "0,547");
  assert.equal(vencedores[0].parametros, "max_depth = None · n_estimators = 300");
  assert.equal(vencedores[0].grade, "max_depth: 8, None · n_estimators: 100, 300");
  const chute = linhas.find((l) => l.modelo.startsWith("Chute"));
  assert.equal(chute.auc, "0,500 ± 0,000");
  assert.equal(chute.parametros, "sem hiperparâmetros");
  assert.equal(chute.grade, "—");
});

test("montarLinhasComparacao: resposta vazia não quebra", () => {
  assert.deepEqual(app.montarLinhasComparacao(null), []);
  assert.deepEqual(app.montarLinhasComparacao({ modelos: [] }), []);
});

// ---------------------------------------------------------------------------
// textoJustificativa: nenhum número fora dos dados de entrada
// ---------------------------------------------------------------------------

function numerosNoTexto(texto) {
  // Números soltos (não colados a letras, como em "F1"), com vírgula ou ponto decimal
  return texto.match(/(?<![\p{L}\d.,])\d+(?:[.,]\d+)?(?![\p{L}\d])/gu) || [];
}

function numerosPermitidos(comparacao) {
  const permitidos = new Set();
  const adicionar = (v) => {
    if (typeof v !== "number") return;
    permitidos.add(String(v));
    for (const casas of [0, 1, 2, 3, 4]) {
      permitidos.add(app.formatarNumero(v, casas));
      permitidos.add(app.formatarPercentual(v, casas).replace("%", ""));
    }
  };
  const percorrer = (x) => {
    if (Array.isArray(x)) x.forEach(percorrer);
    else if (x && typeof x === "object") Object.values(x).forEach(percorrer);
    else adicionar(x);
  };
  percorrer(comparacao);
  return permitidos;
}

function verificarSoNumerosDaEntrada(comparacao) {
  const texto = app.textoJustificativa(comparacao);
  const permitidos = numerosPermitidos(comparacao);
  for (const n of numerosNoTexto(texto)) {
    assert.ok(permitidos.has(n), `número "${n}" não está nos dados de entrada: ${texto}`);
  }
  return texto;
}

test("textoJustificativa: só números dos dados de entrada e os argumentos pedidos", () => {
  const texto = verificarSoNumerosDaEntrada(COMPARACAO);
  assert.ok(texto.includes("Floresta"));
  assert.ok(texto.includes("ROC-AUC"));
  assert.ok(texto.includes("sem depender do corte"));
  assert.ok(texto.includes("KNN"));
  assert.ok(/acurácia/i.test(texto) && texto.includes("desbalanceada"));
  assert.ok(texto.includes("0,789 ± 0,005"));
  assert.ok(texto.includes("maior que um desvio-padrão"));
  assert.ok(numerosNoTexto(texto).length > 0, "o texto deveria citar os números recebidos");
});

test("textoJustificativa: empate técnico e vencedor por desempate também só usam a entrada", () => {
  const empate = structuredClone(COMPARACAO);
  empate.modelos[3].roc_auc_media = 0.7870; // Árvore a menos de 1 dp da Floresta
  const t1 = verificarSoNumerosDaEntrada(empate);
  assert.ok(t1.includes("empate técnico"));

  const desempate = structuredClone(empate);
  desempate.modelos[2].vencedor = false;
  desempate.modelos[3].vencedor = true;
  desempate.modelos[3].roc_auc_media = 0.7860;
  const t2 = verificarSoNumerosDaEntrada(desempate);
  assert.ok(t2.includes("vencedor registrado é Árvore"));
  assert.ok(t2.includes("regra de desempate"));

  const semKnnSemChute = structuredClone(COMPARACAO);
  semKnnSemChute.modelos = semKnnSemChute.modelos.filter((m) => m.modelo === "Floresta" || m.modelo === "Árvore");
  semKnnSemChute.n_folds = 10;
  const t3 = verificarSoNumerosDaEntrada(semKnnSemChute);
  assert.ok(!t3.includes("KNN") && !t3.includes("Chute"));
  assert.ok(t3.includes("10 folds"));
});

test("textoJustificativa: sem vencedor", () => {
  const sem = structuredClone(COMPARACAO);
  sem.modelos.forEach((m) => { m.vencedor = false; });
  assert.ok(app.textoJustificativa(sem).includes("não indica"));
});

// ---------------------------------------------------------------------------
// textoAcuracia: modelo abaixo e acima de "sempre vai pagar" = (VN + FP) / total
// ---------------------------------------------------------------------------

test("textoAcuracia: modelo abaixo do 'sempre vai pagar'", () => {
  // sempre vai pagar = (630 + 150) / 1000 = 78,0%; modelo = (120 + 630) / 1000 = 75,0%
  const m = metricas(120, 630, 150, 100, 0.75);
  const c = app.compararAcuracia(m);
  assert.equal(c.caso, "abaixo");
  assert.equal(c.sempre, 0.78);
  const texto = app.textoAcuracia(m);
  assert.ok(texto.includes("75,0%") && texto.includes("78,0%"));
  assert.ok(texto.includes("abaixo"));
  assert.ok(texto.includes("abre mão de acertos fáceis"));
  assert.ok(texto.includes("desbalanceada"));
  assert.ok(!texto.includes("acima"));
});

test("textoAcuracia: modelo acima do 'sempre vai pagar'", () => {
  // sempre vai pagar = (740 + 20) / 1000 = 76,0%; modelo = (80 + 740) / 1000 = 82,0%
  const m = metricas(80, 740, 20, 160, 0.82);
  const c = app.compararAcuracia(m);
  assert.equal(c.caso, "acima");
  const texto = app.textoAcuracia(m);
  assert.ok(texto.includes("82,0%") && texto.includes("76,0%"));
  assert.ok(texto.includes("acima"));
  assert.ok(!texto.includes("abaixo"));
  assert.ok(!texto.includes("abre mão"));
});

test("textoAcuracia: empate na precisão exibida e matriz ausente", () => {
  const m = metricas(100, 680, 100, 120, 0.78);
  assert.equal(app.compararAcuracia(m).caso, "igual");
  assert.ok(app.textoAcuracia(m).includes("praticamente igual"));
  assert.ok(app.textoAcuracia({ acuracia: 0.8 }).includes("indisponível"));
});

test("textoAvaliacaoTeste: compara a ROC-AUC com a do chute nos dois sentidos", () => {
  const melhor = app.textoAvaliacaoTeste({ roc_auc: 0.71, roc_auc_chute: 0.5, roc_auc_cv_treino: 0.73 });
  assert.ok(melhor.includes("0,710") && melhor.includes("0,500") && melhor.includes("melhor que o acaso"));
  assert.ok(melhor.includes("0,730") && melhor.includes("abaixo"));
  const pior = app.textoAvaliacaoTeste({ roc_auc: 0.49, roc_auc_chute: 0.5 });
  assert.ok(pior.includes("sem superar o chute"));
  assert.ok(app.textoAvaliacaoTeste({ roc_auc: null }).includes("não está no artefato"));
});
