# Crédito Consciente — Predição de Risco de Inadimplência (C318)

Projeto computacional da disciplina **C318 — Fundamentos de Machine Learning (Tópicos Especiais II)** do **Instituto Nacional de Telecomunicações**.

* **Docente:** Prof. Ricardo Augusto
* **Equipe:** Gustavo, Isaque, Lucca e Warley
* **Base de Dados:** *Default of Credit Card Clients* (UCI Machine Learning Repository — $30.000$ clientes)

---

## Aplicação Publicada

| Parte | Endereço |
|---|---|
| **Site (Vercel)** | https://credito-consciente.vercel.app |
| API (Render) | https://credito-consciente-api.onrender.com |
| Documentação da API (Swagger UI) | https://credito-consciente-api.onrender.com/docs |

> A API está no plano gratuito do Render, que a desliga após 15 minutos sem uso. A primeira visita depois disso leva cerca de 50 segundos enquanto a API "acorda"; o site avisa e espera. Antes de uma apresentação, abra o site com alguns minutos de antecedência.

### Atualizar o deploy

O deploy **não é automático** a cada push:

* **API:** no painel do Render, serviço `credito-consciente-api` → **Manual Deploy → Deploy latest commit**. A configuração do serviço está em [`render.yaml`](render.yaml).
* **Site:** na pasta `frontend/`, rode `npx vercel deploy --prod` (pede login na Vercel na primeira vez). Aberto localmente, o site continua usando a API em `http://127.0.0.1:8000`.

---

## Documentação do Projeto (`docs/`)

A documentação está organizada na pasta `docs/` seguindo cada etapa do escopo da disciplina:

| Documento | Conteúdo Abordado |
| :--- | :--- |
| **[01. Contexto e Perguntas de Negócio](docs/01_contexto_e_perguntas_de_negocio.md)** | Problema de negócio, objetivos (negócio e ciência de dados), enquadramento em ML e respostas escritas às perguntas da proposta com os números reais. |
| **[02. Coleta, Dados e Modelagem](docs/02_coleta_dados_e_modelagem.md)** | Avaliação da base da UCI (disponibilidade e fonte única), prevenção de *data leakage*, achados da EDA com gráficos, validação cruzada em 5 *folds* (antes e depois do `GridSearchCV`), ablação de variáveis sensíveis, importância de atributos e corte por custo. |
| **[03. Resultados (Técnico e Negócio)](docs/03_resultados_tecnicos_e_negocio.md)** | Métricas oficiais e Matriz de Confusão no teste ($N = 6.000$), discussão técnica das decisões e tradução prática para gestão de risco. |
| **[04. Limitações e Próximos Passos](docs/04_limitacoes_e_proximos_passos.md)** | Limitações de dados, hipótese de custo ($3:1$), calibração de escores, hiperparâmetros na borda da grade, evoluções futuras e referências. |
| **[Artefato do Modelo (`backend/ARTEFATO_MODELO.md`)](backend/ARTEFATO_MODELO.md)** | Especificação técnica do artefato serializado `backend/modelo_credito.joblib`. |

---

## Resumo dos Resultados Oficiais (Conjunto de Teste, $N = 6.000$)

Todos os números abaixo vêm de uma **única fonte de verdade** (`backend/modelo_credito.joblib`, API FastAPI e `resultados/avaliacao_teste.csv`), medidos sobre os $6.000$ clientes de teste ($1.327$ inadimplentes e $4.673$ adimplentes), utilizados **uma única vez**:

| Modelo | Limiar | Acurácia | Precisão | Recall | $F_1$-Score | ROC-AUC | PR-AUC | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chute (referência)** | $0,50$ | $77,88\%$ | $0,00\%$ | $0,00\%$ | $0,000$ | $0,500$ | $0,221$ | $0$ | $4.673$ | $0$ | $1.327$ |
| **Random Forest (final)** | $0,50$ | **$76,80\%$** | **$48,05\%$** | **$60,29\%$** | **$0,535$** | **$0,773$** | **$0,549$** | **$800$** | **$3.808$** | **$865$** | **$527$** |

> A análise completa para as equipes técnica e de negócio está em [`docs/03_resultados_tecnicos_e_negocio.md`](docs/03_resultados_tecnicos_e_negocio.md).

---

## Arquitetura da Solução

Além dos entregáveis centrais de Ciência de Dados (scripts Python em `backend/`, notebook executado com saídas em `notebooks/` e documentação em `docs/`), o repositório inclui a camada de aplicação (bônus de engenharia):

* **Modelagem (`Scikit-Learn` — `backend/analise_e_modelagem_c318.py` e `notebooks/analise_e_modelagem_c318.ipynb`):** Comparação de 5 modelos (incluindo *baseline* e Árvore de Decisão) por validação cruzada em 5 *folds* exclusivamente no treino, otimização via `GridSearchCV`, exclusão de atributos demográficos sensíveis (`SEX`, `AGE`, `MARRIAGE`) com perda de apenas $0,0012 \pm 0,0009$ na ROC-AUC e definição do ponto de corte ($0,50$) pela função de custo ($3:1$).
* **Back-end (`FastAPI` — `backend/main.py`):** Carrega estritamente o artefato `backend/modelo_credito.joblib` (retornando HTTP `503` claro se ausente), garante coerência entre classe prevista e faixa de risco e funciona a partir de qualquer diretório de execução.
* **Front-end Web (`frontend/index.html` e `frontend/app.js`):** Interface web para simulação e visualização dos indicadores consumindo os dados reais da API, sinalizando que sexo, idade e estado civil não são utilizados pelo modelo. Sem a API, não exibe nenhum resultado e mostra como iniciá-la.
* **Deploy:** API no Render (configuração em `render.yaml`) e site na Vercel — ver [Aplicação Publicada](#aplicação-publicada).
* **Testes Automatizados (`backend/tests/` e `frontend/tests/`):** **207 testes** em `pytest` (API, consistência entre métricas recalculadas e artefato salvo, ausência de *data leakage* com teste de embaralhamento do conjunto de teste e verificações estáticas do front-end) e **18 testes** em `node --test` para a lógica do front-end.

---

## Como Executar e Testar Localmente

> Ambiente testado em **Python 3.14** e **3.13** (o deploy no Render usa 3.13.2). As dependências estão separadas por finalidade com versões fixadas em `backend/requirements*.txt`. A versão do `scikit-learn` precisa ser a do `requirements.txt`: o artefato `.joblib` não carrega em outras versões.

### 1. Criar ambiente virtual e instalar dependências

```bash
# Criar e ativar o ambiente virtual na raiz do repositório
python -m venv .venv

# Windows (PowerShell / CMD)
.venv\Scripts\activate
# Se o PowerShell bloquear o script de ativação, libere para o usuário atual e tente de novo:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Linux / macOS
source .venv/bin/activate

# Dependências mínimas para rodar a API
pip install -r backend/requirements.txt

# Dependências adicionais para rodar o pipeline de treino e gerar gráficos
pip install -r backend/requirements-treino.txt

# Dependências adicionais para rodar os testes automatizados
pip install -r backend/requirements-dev.txt
```

### 2. Executar o pipeline de modelagem (gera o artefato e a pasta `resultados/`)

```bash
python backend/analise_e_modelagem_c318.py
```

### 3. Executar os testes automatizados

```bash
# Back-end, pipeline de ML e verificações estáticas do front-end (207 testes)
pytest -c backend/pytest.ini backend/tests

# Lógica do front-end (18 testes; requer Node.js)
node --test frontend/tests/app.test.js
```

### 4. Iniciar a API FastAPI

A API funciona a partir de qualquer pasta (raiz ou `backend/`):

```bash
# Executando a partir da raiz do repositório
python -m uvicorn main:app --app-dir backend --reload
```

Documentação interativa da API (Swagger UI): `http://127.0.0.1:8000/docs`.

> Use `python -m uvicorn` (e não só `uvicorn`) para garantir que a API rode com o Python do ambiente virtual. Se a porta 8000 estiver ocupada, adicione `--port 8010` e informe `http://127.0.0.1:8010` no campo de endereço no topo do site; o site lembra o último endereço que funcionou.

### 5. Abrir o Front-end Web localmente

Com a API em execução, sirva a pasta `frontend/`:

```bash
python -m http.server 5500 --directory frontend
```

Acesse no navegador: `http://127.0.0.1:5500`. Também funciona abrindo o arquivo `frontend/index.html` direto no navegador.

---

## Estrutura do Repositório

```text
.
├── backend/
│   ├── tests/                              # Testes automatizados da API e do pipeline de ML
│   ├── ARTEFATO_MODELO.md                  # Especificação do artefato serializado
│   ├── analise_e_modelagem_c318.py         # Script de EDA, CV (5 folds), GridSearchCV e teste
│   ├── main.py                             # API FastAPI (inferência e métricas reais)
│   ├── modelo_credito.joblib               # Artefato final do modelo Random Forest
│   ├── pytest.ini                          # Configuração do pytest
│   ├── requirements.txt                    # Dependências de produção (API)
│   ├── requirements-treino.txt             # Dependências de treino, EDA e gráficos
│   └── requirements-dev.txt                # Dependências de desenvolvimento e testes
├── data/
│   └── UCI_Credit_Card.csv                 # Base de dados UCI (30.000 clientes)
├── docs/
│   ├── img/                                # Capturas de tela do site (plano B da apresentação)
│   ├── 01_contexto_e_perguntas_de_negocio.md
│   ├── 02_coleta_dados_e_modelagem.md
│   ├── 03_resultados_tecnicos_e_negocio.md
│   └── 04_limitacoes_e_proximos_passos.md
├── frontend/
│   ├── tests/                              # Testes do front-end (node --test)
│   ├── .vercelignore                       # Arquivos que não vão para o site publicado
│   ├── app.js                              # Lógica de integração com a API FastAPI
│   └── index.html                          # Interface web do simulador e métricas
├── notebooks/
│   └── analise_e_modelagem_c318.ipynb      # Notebook executado com saídas e análises
├── resultados/
│   ├── avaliacao_teste.csv / .json         # Métricas oficiais no conjunto de teste
│   ├── comparacao_modelos_cv_inicial.csv   # CV (5 folds) antes do GridSearchCV
│   ├── comparacao_modelos_cv.csv / .png    # CV (5 folds) após o GridSearchCV
│   ├── grid_search_resultados.csv          # Grade completa de hiperparâmetros avaliada
│   ├── importancia_atributos.csv / .png    # Importância por permutação e nativa
│   ├── limiar_custo.csv                    # Avaliação de limiares (0,10 a 0,90) no treino OOF
│   ├── limiar_sensibilidade_custo.csv      # Sensibilidade do limiar às razões 1:1 até 10:1
│   ├── curva_cortes.png                    # Trade-off Recall x Precisão e custo relativo
│   ├── sensiveis_com_vs_sem.csv            # Ablação pareada de SEX, AGE e MARRIAGE
│   ├── melhores_hiperparametros.json       # Hiperparâmetros ótimos por modelo
│   ├── matriz_confusao_final.png           # Matriz de confusão no teste (N = 6.000)
│   └── eda_*.png                           # Gráficos da análise exploratória (treino)
├── .gitignore
├── render.yaml                             # Configuração da API no Render (deploy)
└── README.md                               # Guia técnico e índice do projeto
```