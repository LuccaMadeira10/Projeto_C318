/*
 * Crédito Consciente — Projeto C318 (Inatel)
 * Lógica do front que não mexe na página: estado da API, leitura de erros, formatação e
 * textos montados a partir dos números recebidos da API. Nenhum número do modelo é escrito
 * aqui: tudo vem de GET /, /metrics, /comparacao-modelos e /predict.
 *
 * No navegador, as funções ficam globais (<script src="app.js">). No Node (testes em
 * frontend/tests), são exportadas por module.exports.
 */

// ---------------------------------------------------------------------------
// Formatação (padrão brasileiro: vírgula decimal). Valor ausente vira "—".
// ---------------------------------------------------------------------------

const VAZIO = "—";

function numeroValido(valor) {
  return typeof valor === "number" && Number.isFinite(valor);
}

function formatarNumero(valor, casas = 3) {
  if (!numeroValido(valor)) return VAZIO;
  return valor.toFixed(casas).replace(".", ",");
}

function formatarPercentual(fracao, casas = 1) {
  if (!numeroValido(fracao)) return VAZIO;
  return (fracao * 100).toFixed(casas).replace(".", ",") + "%";
}

function formatarInteiro(valor) {
  if (!numeroValido(valor)) return VAZIO;
  return Math.round(valor).toLocaleString("pt-BR");
}

function formatarMediaDp(media, dp, casas = 3) {
  if (!numeroValido(media)) return VAZIO;
  if (!numeroValido(dp)) return formatarNumero(media, casas);
  return `${formatarNumero(media, casas)} ± ${formatarNumero(dp, casas)}`;
}

/** Corte de decisão com pelo menos 2 casas e sem perder casas (0.5 -> "0,50"; 0.355 -> "0,355"). */
function formatarCorte(limiar) {
  if (!numeroValido(limiar)) return VAZIO;
  const texto = String(limiar);
  const casas = texto.includes(".") && !texto.includes("e") ? texto.split(".")[1].length : 0;
  return formatarNumero(limiar, Math.max(2, casas));
}

/** Percentual que a API já devolve formatado (ex.: "49.9%"): só troca o separador decimal. */
function formatarPercentualApi(texto) {
  if (typeof texto !== "string" || !texto) return VAZIO;
  return texto.replace(".", ",");
}

/** Valor de hiperparâmetro na notação do Python/scikit-learn (null -> None). */
function formatarValorParametro(valor) {
  if (valor === null || valor === undefined) return "None";
  if (valor === true) return "True";
  if (valor === false) return "False";
  return String(valor);
}

function formatarParametros(parametros) {
  if (parametros === null || parametros === undefined || typeof parametros !== "object") return VAZIO;
  const chaves = Object.keys(parametros);
  if (chaves.length === 0) return "sem hiperparâmetros";
  return chaves.map((c) => `${c} = ${formatarValorParametro(parametros[c])}`).join(" · ");
}

function formatarGrade(grade) {
  if (grade === null || grade === undefined || typeof grade !== "object") return VAZIO;
  const chaves = Object.keys(grade);
  if (chaves.length === 0) return VAZIO;
  return chaves
    .map((c) => `${c}: ${(Array.isArray(grade[c]) ? grade[c] : [grade[c]]).map(formatarValorParametro).join(", ")}`)
    .join(" · ");
}

/** "2026-11-18T19:35:19+00:00" -> "18/11/2026 19:35 (UTC+00:00)", sem conversão de fuso. */
function formatarData(iso) {
  if (typeof iso !== "string" || !iso) return VAZIO;
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:\d{2})?$/);
  if (!m) return iso;
  const fuso = m[6] ? ` (UTC${m[6] === "Z" ? "" : m[6]})` : "";
  return `${m[3]}/${m[2]}/${m[1]} ${m[4]}:${m[5]}${fuso}`;
}

function formatarSimNao(valor, textoSim, textoNao) {
  if (valor === true) return textoSim;
  if (valor === false) return textoNao;
  return "não informado no artefato";
}

// ---------------------------------------------------------------------------
// Estado da API (a partir do GET /)
// ---------------------------------------------------------------------------

const COMANDOS_SUBIR_API = [
  "python -m venv .venv",
  ".venv\\Scripts\\activate        (Linux/macOS: source .venv/bin/activate)",
  "pip install -r backend/requirements.txt",
  "uvicorn main:app --app-dir backend",
];

const COMANDOS_GERAR_MODELO = [
  "pip install -r backend/requirements-dev.txt",
  "python backend/analise_e_modelagem_c318.py",
  "# depois reinicie a API (uvicorn main:app --app-dir backend)",
];

/**
 * Estado do site a partir da resposta de GET / (objeto JSON) ou null quando não houve
 * resposta. Só há resultados de modelo no estado "conectado" (modelo_carregado === true).
 * mensagemErro: detalhe do 503 de /metrics, mostrado no estado "sem_modelo".
 */
function estadoDaApi(raiz, mensagemErro, url) {
  const endereco = url || "http://127.0.0.1:8000";
  if (raiz === null || raiz === undefined || typeof raiz !== "object") {
    return {
      estado: "desligada",
      titulo: "API desligada",
      mensagem:
        `Nenhuma resposta da API em ${endereco}. Nenhum resultado de modelo é exibido. ` +
        "Para subir a API, rode na raiz do projeto:",
      comandos: COMANDOS_SUBIR_API,
      exibeResultados: false,
    };
  }
  if (raiz.modelo_carregado === true) {
    return {
      estado: "conectado",
      titulo: "Conectado",
      mensagem: `Modelo carregado: ${raiz.modelo_ativo || "artefato do script de treino"}.`,
      comandos: [],
      exibeResultados: true,
    };
  }
  return {
    estado: "sem_modelo",
    titulo: "API sem modelo",
    mensagem:
      (mensagemErro ||
        "A API está no ar, mas não carregou o artefato do modelo (modelo_carregado: false).") +
      " Nenhum resultado de modelo é exibido. Para gerar o modelo, rode na raiz do projeto:",
    comandos: COMANDOS_GERAR_MODELO,
    exibeResultados: false,
  };
}

// ---------------------------------------------------------------------------
// Leitura de erros da API: lista do Pydantic (422), objeto do modo estrito (422) e texto (503)
// ---------------------------------------------------------------------------

function lerErro(status, corpo) {
  const detalhe = corpo && typeof corpo === "object" ? corpo.detail : undefined;
  if (Array.isArray(detalhe)) {
    const itens = detalhe.map((d) => {
      const loc = Array.isArray(d && d.loc) ? d.loc.filter((p) => p !== "body") : [];
      const campo = loc.length ? loc.join(".") : "requisição";
      return `${campo}: ${(d && d.msg) || "valor inválido"}`;
    });
    return itens.length ? itens.join("; ") : `HTTP ${status}`;
  }
  if (detalhe && typeof detalhe === "object") {
    let texto = detalhe.mensagem || `HTTP ${status}`;
    if (Array.isArray(detalhe.campos_faltantes) && detalhe.campos_faltantes.length) {
      texto += ` Campos faltantes: ${detalhe.campos_faltantes.join(", ")}.`;
    }
    return texto;
  }
  if (typeof detalhe === "string" && detalhe) return detalhe;
  if (typeof corpo === "string" && corpo) return corpo;
  return `HTTP ${status}`;
}

// ---------------------------------------------------------------------------
// Comparação dos modelos (GET /comparacao-modelos)
// ---------------------------------------------------------------------------

function _ordenarPorAuc(modelos) {
  return [...modelos].sort((a, b) => (b.roc_auc_media ?? -Infinity) - (a.roc_auc_media ?? -Infinity));
}

/** Linhas prontas para a tabela, da maior para a menor ROC-AUC média, com o vencedor marcado. */
function montarLinhasComparacao(comparacao) {
  const modelos = (comparacao && Array.isArray(comparacao.modelos)) ? comparacao.modelos : [];
  return _ordenarPorAuc(modelos).map((m) => ({
    modelo: m.modelo,
    vencedor: m.vencedor === true,
    auc: formatarMediaDp(m.roc_auc_media, m.roc_auc_dp, 3),
    pr_auc: formatarMediaDp(m.pr_auc_media, m.pr_auc_dp, 3),
    recall: formatarPercentual(m.recall_media, 1),
    precisao: formatarPercentual(m.precisao_media, 1),
    f1: formatarNumero(m.f1_media, 3),
    parametros: m.melhores_parametros ? formatarParametros(m.melhores_parametros) : "sem hiperparâmetros",
    grade: m.grade ? formatarGrade(m.grade) : VAZIO,
  }));
}

function _ehChute(m) {
  return /chute/i.test(m.modelo || "");
}

/**
 * Justificativa da escolha montada só com os números recebidos (nenhum número fixo).
 * Cobre: vencedor com a maior ROC-AUC (com folga ou empate técnico), vencedor escolhido por
 * desempate, presença ou não do KNN e do Chute.
 */
function textoJustificativa(comparacao) {
  const modelos = (comparacao && Array.isArray(comparacao.modelos)) ? comparacao.modelos : [];
  const vencedor = modelos.find((m) => m.vencedor === true);
  if (!vencedor) return "A comparação recebida não indica um modelo vencedor.";

  const partes = [];
  const folds = numeroValido(comparacao.n_folds) ? ` em ${comparacao.n_folds} folds` : "";
  partes.push(`Os modelos foram comparados por validação cruzada${folds}, usando só os dados de treino.`);

  const criterio = comparacao.criterio;
  if (criterio && criterio !== "roc_auc") {
    partes.push(`O critério de escolha registrado nos resultados foi "${criterio}".`);
  }

  const ordenados = _ordenarPorAuc(modelos);
  const lider = ordenados[0];
  const auc = (m) => formatarMediaDp(m.roc_auc_media, m.roc_auc_dp, 3);
  if (lider === vencedor || lider.roc_auc_media === vencedor.roc_auc_media) {
    const segundo = ordenados.find((m) => m !== vencedor);
    let frase = `${vencedor.modelo} teve a maior ROC-AUC média (${auc(vencedor)})`;
    if (segundo) {
      frase += `, à frente de ${segundo.modelo} (${auc(segundo)})`;
      const folga = vencedor.roc_auc_media - segundo.roc_auc_media;
      frase += numeroValido(vencedor.roc_auc_dp) && folga > vencedor.roc_auc_dp
        ? ": a vantagem é maior que um desvio-padrão do vencedor."
        : ": a vantagem não passa de um desvio-padrão do vencedor, um empate técnico decidido pela regra de desempate do treino.";
    } else {
      frase += ".";
    }
    partes.push(frase);
  } else {
    const dentro = numeroValido(lider.roc_auc_dp) && lider.roc_auc_media - vencedor.roc_auc_media < lider.roc_auc_dp;
    partes.push(
      `${lider.modelo} teve a maior ROC-AUC média (${auc(lider)}), mas o vencedor registrado é ` +
      `${vencedor.modelo} (${auc(vencedor)})` +
      (dentro
        ? ": a diferença é menor que um desvio-padrão do líder, e o empate foi decidido pela regra de desempate do treino."
        : ": a diferença passa de um desvio-padrão do líder, então confira os resultados do treino.")
    );
  }

  partes.push(
    "A ROC-AUC foi usada porque mede se o modelo põe os inadimplentes acima dos bons pagadores " +
    "sem depender do corte de decisão."
  );
  if (modelos.some((m) => /knn|vizinho/i.test(m.modelo || ""))) {
    partes.push(
      "Isso é justo com o KNN, cujo escore (fração de vizinhos inadimplentes) tem outra escala: " +
      "um corte único escolhido para outro modelo o prejudicaria ou o favoreceria."
    );
  }

  partes.push(
    "A acurácia não foi usada porque engana em base desbalanceada: como a maioria dos clientes " +
    "paga em dia, um modelo que nunca aponta inadimplentes já acerta a maior parte dos casos."
  );
  const chute = modelos.find(_ehChute);
  if (chute) {
    const recall = formatarPercentual(chute.recall_media, 1);
    partes.push(
      `O ${chute.modelo} mostra isso: ROC-AUC de ${formatarNumero(chute.roc_auc_media, 3)} e recall de ${recall}` +
      (chute.recall_media === 0 ? ", sem encontrar nenhum inadimplente." : ".")
    );
  }
  return partes.join(" ");
}

// ---------------------------------------------------------------------------
// Avaliação final no teste (GET /metrics)
// ---------------------------------------------------------------------------

/** Acurácia do modelo e da estratégia "sempre vai pagar", (VN + FP) / total, pela própria matriz. */
function compararAcuracia(metricas) {
  const mc = metricas && metricas.matriz_confusao;
  if (!mc || ![mc.VP, mc.VN, mc.FP, mc.FN].every(numeroValido) || !numeroValido(metricas.acuracia)) {
    return null;
  }
  const total = mc.VP + mc.VN + mc.FP + mc.FN;
  if (total <= 0) return null;
  const modelo = metricas.acuracia;
  const sempre = (mc.VN + mc.FP) / total;
  let caso;
  if (formatarPercentual(modelo) === formatarPercentual(sempre)) caso = "igual";
  else caso = modelo < sempre ? "abaixo" : "acima";
  return { caso, modelo, sempre, total, inadimplentes: mc.VP + mc.FN, encontrados: mc.VP };
}

function textoAcuracia(metricas) {
  const c = compararAcuracia(metricas);
  if (!c) return "Matriz de confusão indisponível: não é possível comparar com a estratégia “sempre vai pagar”.";
  const acc = formatarPercentual(c.modelo);
  const sempre = formatarPercentual(c.sempre);
  const achados = `${formatarInteiro(c.encontrados)} dos ${formatarInteiro(c.inadimplentes)} inadimplentes do teste`;
  if (c.caso === "abaixo") {
    return (
      `A acurácia do modelo (${acc}) fica abaixo da estratégia “sempre vai pagar” (${sempre}), ` +
      "que acerta todos os bons pagadores e nenhum inadimplente. Não é defeito: a base é " +
      "desbalanceada e deixar passar um inadimplente custa mais que incomodar um bom pagador, " +
      `então o modelo abre mão de acertos fáceis para encontrar ${achados}.`
    );
  }
  if (c.caso === "acima") {
    return (
      `A acurácia do modelo (${acc}) fica acima da estratégia “sempre vai pagar” (${sempre}): ` +
      `ele erra menos no total do que apostar que todos pagam e ainda encontra ${achados}. ` +
      "Mesmo assim, em base desbalanceada a acurácia sozinha engana; recall e precisão mostram o custo de cada tipo de erro."
    );
  }
  return (
    `A acurácia do modelo (${acc}) é praticamente igual à da estratégia “sempre vai pagar” (${sempre}), ` +
    `mas o modelo encontra ${achados}, enquanto a estratégia não encontra nenhum. ` +
    "Em base desbalanceada, a acurácia sozinha engana."
  );
}

function textoAvaliacaoTeste(metricas) {
  if (!metricas || !numeroValido(metricas.roc_auc)) {
    return "A ROC-AUC no teste não está no artefato carregado.";
  }
  const roc = formatarNumero(metricas.roc_auc, 3);
  let texto = `No conjunto de teste, usado uma única vez, a ROC-AUC do modelo foi ${roc}`;
  if (numeroValido(metricas.roc_auc_chute)) {
    const chute = formatarNumero(metricas.roc_auc_chute, 3);
    texto += metricas.roc_auc > metricas.roc_auc_chute
      ? `, contra ${chute} do chute: o modelo separa inadimplentes de bons pagadores melhor que o acaso.`
      : `, sem superar o chute (${chute}): o modelo não separa os grupos melhor que o acaso.`;
  } else {
    texto += ".";
  }
  if (numeroValido(metricas.roc_auc_cv_treino)) {
    const cv = formatarNumero(metricas.roc_auc_cv_treino, 3);
    texto += metricas.roc_auc < metricas.roc_auc_cv_treino
      ? ` Na validação cruzada do treino ela foi ${cv}; ficar um pouco abaixo no teste é esperado (variação da amostra e otimismo da seleção feita no treino).`
      : ` Na validação cruzada do treino ela foi ${cv}; o teste não ficou abaixo da estimativa do treino.`;
  }
  return texto;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    VAZIO,
    COMANDOS_SUBIR_API,
    COMANDOS_GERAR_MODELO,
    formatarNumero,
    formatarPercentual,
    formatarInteiro,
    formatarMediaDp,
    formatarCorte,
    formatarPercentualApi,
    formatarValorParametro,
    formatarParametros,
    formatarGrade,
    formatarData,
    formatarSimNao,
    estadoDaApi,
    lerErro,
    montarLinhasComparacao,
    textoJustificativa,
    compararAcuracia,
    textoAcuracia,
    textoAvaliacaoTeste,
  };
}
