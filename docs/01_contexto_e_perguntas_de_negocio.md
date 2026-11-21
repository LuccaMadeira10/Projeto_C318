# 01. Contexto, Enquadramento e Perguntas de Negócio

[Voltar ao README](../README.md) | [Próximo: 02. Coleta, Dados e Modelagem](02_coleta_e_modelagem.md)

---

## 1. Contexto e Problema de Negócio

Atrasos em pagamentos de faturas de cartão de crédito geram perdas financeiras diretas, elevam o custo de provisão de capital e dificultam o gerenciamento de risco nas instituições financeiras. Quando a instituição reage apenas após o vencimento da fatura, a recuperação do valor tende a ser mais cara e menos eficaz.

O projeto **Crédito Consciente** utiliza o histórico de faturas e pagamentos dos últimos 6 meses para estimar o risco de inadimplência no mês seguinte, permitindo ações preventivas de baixo atrito (como lembretes de vencimento, oferta antecipada de parcelamento ou revisão de limite) antes que o atraso se consolide.

---

## 2. Objetivos do Projeto

### Objetivo de Negócio
Apoiar a gestão de risco de cartão de crédito identificando antecipadamente clientes com maior propensão a atrasar a fatura no mês seguinte, reduzindo o prejuízo de inadimplências não previstas (Falsos Negativos) sem gerar atrito excessivo junto a clientes adimplentes (Falsos Positivos).

### Objetivo de Ciência de Dados
Desenvolver, validar e comparar modelos supervisionados de classificação binária sobre a base pública *Default of Credit Card Clients* (UCI), aplicando integralmente o processo de Ciência de Dados estudado na disciplina C318: separação imediata entre treino e teste, prevenção comprovada de *data leakage*, seleção via validação cruzada e `GridSearchCV` apenas no treino, avaliação adequada a dados desbalanceados e definição do ponto de corte baseada em custo.

---

## 3. Enquadramento do Problema em Machine Learning

| Critério Teórico (C318) | Definição no Projeto Crédito Consciente |
| :--- | :--- |
| **Grau de Supervisionamento** | **Aprendizagem Supervisionada — Classificação Binária** ($y \in \{0, 1\}$, onde $1$ indica inadimplência no mês seguinte e $0$ indica pagamento em dia). |
| **Forma de Incremento** | **Offline (*Batch Learning*)**, com treinamento realizado sobre a base histórica consolidada da UCI. |
| **Modo de Aprendizagem** | Comparação experimental entre abordagens **baseadas em modelos** (Regressão Logística, Árvore de Decisão e *Random Forest*) e **baseadas em instâncias** (K-NN), contrastadas contra um classificador de referência (*Chute* da classe majoritária). |
| **Métricas de Desempenho** | Em função do desbalanceamento ($\approx 22,12\%$ de inadimplentes e $\approx 77,88\%$ de adimplentes), a acurácia isolada é enganosa. Priorizam-se **ROC-AUC** (ordenação de risco), **PR-AUC**, **Recall (Sensibilidade)**, **Precisão** e **$F_1$-Score**. |

---

## 4. Perguntas de Negócio e Respostas Obtidas

Abaixo estão as perguntas definidas na etapa inicial do projeto, respondidas com os resultados reais obtidos na validação cruzada e no conjunto de teste ($N = 6.000$ clientes):

### Pergunta 1
> *"Com base no histórico de faturas e pagamentos dos últimos 6 meses, é possível estimar o risco de um cliente atrasar o pagamento da fatura no mês seguinte de forma útil para o negócio?"*

**Resposta:** **Sim.** No conjunto de teste isolado ($6.000$ clientes), o modelo final (*Random Forest*) alcançou **ROC-AUC de $0,7734$** e **PR-AUC de $0,5486$** (contra $0,5000$ e $0,2212$ do *baseline* "sempre paga"). Com o limiar de decisão em $0,50$, o modelo detectou **$800$ dos $1.327$ inadimplentes reais (Recall de $60,29\%$)** com **Precisão de $48,05\%$** ($800$ acertos em $1.665$ alertas) e **$F_1$-Score de $0,5348$**.

### Pergunta 2
> *"Quais fatores pesam mais na previsão de inadimplência? Atributos demográficos (sexo, idade e estado civil) são indispensáveis para o modelo?"*

**Resposta:** **O histórico comportamental recente concentra o sinal preditivo, e sexo, idade e estado civil são dispensáveis.** Na análise de importância por permutação, o status de pagamento do mês mais recente (`PAY_0`) liderou com ampla folga ($0,0691 \pm 0,0049$), seguido pelo limite de crédito (`LIMIT_BAL`, $0,0141 \pm 0,0025$), status de 2 meses atrás (`PAY_2`, $0,0092 \pm 0,0022$) e valor da fatura mais recente (`BILL_AMT1`, $0,0087 \pm 0,0009$). Ao remover `SEX`, `AGE` e `MARRIAGE`, a ROC-AUC na validação cruzada caiu apenas **$0,0012 \pm 0,0009$** (de $0,7857$ para $0,7845$), enquanto o Recall subiu levemente de $61,80\%$ para $62,06\%$.

### Pergunta 3
> *"Qual ponto de corte equilibra o prejuízo de deixar passar um inadimplente (FN) contra o custo de abordar preventivamente um bom pagador (FP)?"*

**Resposta:** Sob a hipótese de custo adotada pelo grupo de **$3$ para $1$** (um Falso Negativo custa $3$ vezes mais que um Falso Positivo), a análise *out-of-fold* no conjunto de treino identificou o **limiar ótimo em $0,50$** (custo mínimo de $9.558$ unidades). Aplicando essa mesma regra ao conjunto de teste, o custo ponderado cai de **$3.981$** no *baseline* para **$2.446$** com o *Random Forest* — uma **redução de $38,56\%$ no custo total de erros**.

---

[Voltar ao README](../README.md) | [Próximo: 02. Coleta, Dados e Modelagem](02_coleta_e_modelagem.md)