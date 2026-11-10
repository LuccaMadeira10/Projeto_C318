# Projeto_C318

```
projeto-c318-credito-consciente/
│
├── data/
│   └── UCI_Credit_Card.csv          # Base de dados bruta da UCI
│
├── notebooks/                       # ENTREGÁVEL 1: Scripts Computacionais
│   ├── 01_EDA_e_Limpeza.ipynb       # Exploração, gráficos e separação treino/teste
│   └── 02_Pipeline_Modelagem.ipynb  # Pipeline Scikit-Learn, métricas e exportação do modelo
│
├── backend/                         # API para servir o modelo
│   ├── modelo_credito.joblib        # Pipeline exportado (Scaler + Modelo)
│   ├── main.py                      # Código FastAPI com rota POST /predict
│   └── requirements.txt             # fastapi, uvicorn, scikit-learn, pandas, joblib
│
├── frontend/                        # Aplicação Web (Deploy na Vercel)
│   ├── src/                         # Telas React/Next.js + Tailwind CSS
│   └── package.json
│
└── README.md                        # ENTREGÁVEL 2: Documentação 
```

# Avaliação Técnica (C318) e Guia Prático de Execução e Testes

## 1. Avaliação do Código e do Modelo de IA para o Projeto C318

Sim, **a sua API e o seu modelo de Machine Learning já estão excelentes e prontos** para atender aos critérios exigidos pelo Prof. Ricardo Augusto na disciplina C318. Abaixo estão os pontos que garantem uma nota alta na avaliação técnica:

1. **Zero Vazamento de Dados (*Data Leakage*):**
   * O código realiza o `train_test_split(..., test_size=0.20, stratify=y)` **antes** de ajustar o `StandardScaler` e o `OneHotEncoder`.
   * O uso de `stratify=y` garante que tanto o conjunto de treino ($24.000$ clientes) quanto o conjunto de teste ($6.000$ clientes) mantenham exatamente a mesma proporção original da classe alvo ($\approx 22,12\%$ de inadimplentes).
2. **Pipeline Oficial do Scikit-Learn (`Pipeline` + `ColumnTransformer`):**
   * Em vez de aplicar transformações soltas no código, o modelo encapsula o escalonamento de variáveis numéricas (`StandardScaler`) e a codificação de variáveis categóricas (`OneHotEncoder`) dentro de um único objeto `Pipeline`.
3. **Tratamento do Desbalanceamento de Classes:**
   * O `RandomForestClassifier` está configurado com `class_weight="balanced"`. Na apresentação para o professor, destaque que essa escolha foi feita porque a Acurácia isolada é enganosa em bases desbalanceadas, e o objetivo de negócio é maximizar o **Recall** ($\text{Recall} = \frac{VP}{VP + FN}$) para reduzir os Falsos Negativos (calotes não previstos).
4. **Duas Melhorias que Acabei de Incluir no seu `main.py`:**
   * **Persistência das Métricas Reais:** Na versão anterior, quando o servidor reiniciava e carregava o arquivo `modelo_credito.joblib`, as métricas voltavam para os valores estáticos iniciais. Agora, o `joblib.dump()` salva um dicionário contendo o `pipeline` e as `metricas` calculadas sobre os $6.000$ clientes de teste.
   * **Preenchimento Inteligente do Histórico (`PAY_2` a `PAY_6`):** Como o formulário simplificado do Front-end envia apenas 8 campos (incluindo o `PAY_0`), ajustei a rota `/predict` para propagar proporcionalmente o status de atraso caso os meses anteriores não sejam informados (`None`), tornando o modelo muito mais sensível aos testes que você fizer na tela.

---

## 2. Passo a Passo: Como Rodar o Projeto no seu Computador

### Passo 1: Organizar os Arquivos na Mesma Pasta
Crie uma pasta para o projeto (por exemplo, `credito-consciente`) e coloque dentro dela os 3 arquivos:
* `main.py` (o código da API FastAPI atualizado)
* `UCI_Credit_Card.csv` (o dataset que você enviou)
* `index.html` (o Front-end do simulador que geramos)

*Observação:* Se já existir um arquivo `modelo_credito.joblib` antigo na pasta, apague-o antes de rodar pela primeira vez para que o `main.py` gere um novo com as métricas reais do seu CSV.

### Passo 2: Criar um Ambiente Virtual e Instalar as Bibliotecas
Abra o terminal (PowerShell, CMD ou terminal do VS Code) dentro da pasta do projeto e execute:

    # 1. Cria o ambiente virtual (opcional, mas recomendado)
    python -m venv venv

    # 2. Ativa o ambiente virtual
    # No Windows (PowerShell/CMD):
    venv\Scripts\activate
    # No Linux ou Mac:
    source venv/bin/activate

    # 3. Instala as dependências necessárias
    pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic

### Passo 3: Iniciar o Servidor FastAPI
Ainda no terminal, execute o comando abaixo:

    uvicorn main:app --reload

**O que vai acontecer no terminal?**
Na primeira vez que você rodar, verá a mensagem:
`[C318] Lendo base de dados 'UCI_Credit_Card.csv' e treinando Pipeline Scikit-Learn...`
Em cerca de $3$ a $5$ segundos, ele treinará a Random Forest com $24.000$ clientes, avaliará nos $6.000$ clientes de teste, salvará o arquivo `modelo_credito.joblib` e exibirá:
`Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

---

## 3. Como Testar a API e o Modelo de IA (3 Formas)

### Forma 1: Testar Visualmente pelo Front-end (`index.html`)
1. Com o terminal rodando o `uvicorn`, basta dar **duplo clique no arquivo `index.html`** para abri-lo no seu navegador (Chrome, Edge, etc.).
2. No topo da página, você verá o indicador ficar **Verde (`Online`)** conectado em `http://127.0.0.1:8000`.
3. Clique nas 3 abas no canto superior direito:
   * **Simulador de Risco:** Altere o campo *Status Pagamento Mês Atual (`PAY_0`)* de `-1 = Pago em dia` para `2 = Atraso de 2 meses` e clique em **Executar Predição de Risco**. Você verá a barra saltar de **Risco Baixo (Verde)** para **Risco Alto (Vermelho)** na hora!
   * **Métricas do Modelo (UCI):** Exibe os cards com Acurácia, Recall, Precisão, $F_1\text{-Score}$ e a Matriz de Confusão real calculada sobre o seu CSV.
   * **Validação do Dataset:** Confirma que o `UCI_Credit_Card.csv` possui $30.000$ linhas, $25$ colunas e $\approx 22,12\%$ de taxa de inadimplência.

### Forma 2: Testar pela Documentação Automática do FastAPI (Swagger UI)
O FastAPI cria uma tela de testes interativa automática (excelente para mostrar ao professor durante as mentorias!):
1. Abra o navegador e acesse: `http://127.0.0.1:8000/docs`
2. Clique na rota **`GET /dataset-check`** $\rightarrow$ **Try it out** $\rightarrow$ **Execute** para ver o diagnóstico do CSV.
3. Clique na rota **`POST /predict`** $\rightarrow$ **Try it out** e teste colando os dois cenários abaixo no corpo da requisição (*Request body*):

**Cenário A — Cliente Bom Pagador (Deve retornar Risco Baixo / Verde):**

    {
      "LIMIT_BAL": 150000,
      "SEX": 2,
      "EDUCATION": 1,
      "MARRIAGE": 1,
      "AGE": 38,
      "PAY_0": -1,
      "BILL_AMT1": 12000,
      "PAY_AMT1": 12000
    }

**Cenário B — Cliente com Atraso Recente e Pagamento Insuficiente (Deve retornar Risco Alto / Vermelho):**

    {
      "LIMIT_BAL": 30000,
      "SEX": 1,
      "EDUCATION": 2,
      "MARRIAGE": 2,
      "AGE": 25,
      "PAY_0": 2,
      "BILL_AMT1": 29500,
      "PAY_AMT1": 0
    }

### Forma 3: E para o Entregável do Jupyter Notebook?
Lembre-se de que o PDF da disciplina pede os **Scripts Computacionais** (além da apresentação/sistema). Embora o `main.py` já faça todo o treinamento do Pipeline automaticamente, para garantir $100\%$ da nota na rubrica acadêmica vale a pena ter um arquivo `.ipynb` (Jupyter Notebook) com os gráficos da Análise Exploratória (EDA) e a comparação entre 2 ou 3 algoritmos (por exemplo: Regressão Logística vs. KNN vs. Random Forest) justificando por que a Random Forest foi a escolhida para ir para a API.