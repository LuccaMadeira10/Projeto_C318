# 04. Limitações do Projeto, Evoluções Futuras e Referências

[Anterior: 03. Resultados](03_resultados_negocio_e_tecnico.md) | [Voltar ao README](../README.md)

---

## 1. Limitações Conhecidas

1. **Fonte única e defasagem temporal/geográfica:**  
   Todos os $30.000$ registros de `data/UCI_Credit_Card.csv` vêm de uma única instituição financeira em Taiwan no ano de 2005. Comportamentos de consumo, meios de pagamento atuais, juros e cenário econômico no Brasil diferem significativamente dessa base. O pipeline demonstra rigor metodológico, mas exige retreinamento com dados locais recentes antes de qualquer operação real.
2. **Hipótese de custo dos erros ($3$ para $1$):**  
   A relação $3:1$ entre o custo de um Falso Negativo e um Falso Positivo — que posicionou o limiar em $0,50$ — é uma premissa de projeto, não um cálculo derivado de métricas financeiras reais (como valor médio da fatura inadimplente, taxa de recuperação e custo real da régua de contato). Conforme `resultados/limiar_sensibilidade_custo.csv`, caso a razão real fosse $5:1$, o corte cairia para $0,40$; se fosse $10:1$, cairia para $0,25$, exigindo recalibrar as faixas de risco em `backend/main.py`.
3. **Escores não calibrados:**  
   Os valores contínuos produzidos pelo *Random Forest* não passaram por calibração de probabilidade (*Platt Scaling* ou *Isotonic Regression*). Servem para ordenar clientes do menor para o maior risco, e não como probabilidades literais para precificação atuarial de juros ou provisão contábil.
4. **Hiperparâmetros selecionados na borda da grade (`GridSearchCV`):**  
   Em todos os modelos ajustados (`resultados/grid_search_resultados.csv`), pelo menos um hiperparâmetro ótimo atingiu o limite superior da grade avaliada:
   * **Random Forest (vencedor):** `n_estimators = 500` ficou no teto da grade (`[200, 500]`), embora a diferença frente a `n_estimators = 200` com os mesmos parâmetros seja de apenas $0,00018$ na ROC-AUC ($0,78551$ vs. $0,78570$), indicando um **platô de desempenho**. Já `min_samples_leaf = 20` ficou no interior da grade (`[10, 20, 40]`).
   * **Modelos concorrentes:** `C = 10.0` na Regressão Logística (platô desde `C = 0.1`), `n_neighbors = 301` no KNN e `min_samples_leaf = 500` na Árvore de Decisão também ficaram no limite superior de suas respectivas grades. Valores além dessas bordas não foram testados neste ciclo.
5. **Exclusão de atributos demográficos (`SEX`, `AGE`, `MARRIAGE`):**  
   A retirada dessas variáveis reduziu a ROC-AUC na validação cruzada em **$0,0012 \pm 0,0009$** — uma queda pequena, mas consistente entre os 5 *folds*. A equipe assumiu conscientemente essa troca em favor de governança e menor risco de discriminação, mantendo a escolha configurável no topo de `backend/analise_e_modelagem_c318.py`.
6. **Dependência de comportamento recente (`PAY_0`) e variáveis pouco informativas:**  
   A importância por permutação (`resultados/importancia_atributos.csv`) mostra forte concentração em `PAY_0` ($0,0691$), enquanto atributos antigos (`BILL_AMT4`, `BILL_AMT6`, `PAY_AMT5`) e `EDUCATION` têm contribuição próxima de zero. Para clientes sem histórico prévio (*cold start*), o desempenho do modelo seria naturalmente reduzido.

---

## 2. Próximos Passos (Evoluções Futuras)

* **Calibração de probabilidades:** Aplicar `CalibratedClassifierCV` (isotônica ou sigmoide) em validação cruzada para transformar os escores em probabilidades calibradas reais.
* **Função de custo baseada em valor financeiro (`BILL_AMT`):** Substituir o custo unitário fixo ($3:1$) por um custo ponderado pelo saldo exposto de cada cliente.
* **Novas famílias de modelos e expansão de grade:** Avaliar algoritmos de *Gradient Boosting* (`HistGradientBoostingClassifier`, LightGBM, XGBoost) e testar combinações além da borda atual do `GridSearchCV`.
* **Monitoramento em produção (*Data Drift*):** Acompanhar a distribuição de `PAY_0`, utilização de limite e taxa de inadimplência observada ao longo do tempo para disparar retreinamentos periódicos.

---

## 3. Referências

* **Yeh, I. C., & Lien, C. H. (2009).** *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
* **UCI Machine Learning Repository.** *Default of Credit Card Clients Dataset.* Disponível em: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
* **Scikit-Learn Developers.** *User Guide — Pipelines, Cross-Validation, GridSearchCV, Permutation Importance and Metrics.*
* **Inatel — Disciplina C318.** *Fundamentos de Machine Learning (Tópicos Especiais II)*, notas de aula e escopo de projeto computacional, Prof. Ricardo Augusto.

---

[Anterior: 03. Resultados](03_resultados_negocio_e_tecnico.md) | [Voltar ao README](../README.md)