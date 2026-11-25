# Contrato do artefato `modelo_credito.joblib`

Gerado **somente** por `analise_e_modelagem_c318.py`. A API (`main.py`) apenas carrega e
valida o arquivo no startup; ela nunca treina nem grava o artefato.

```python
joblib.dump({
    "pipeline": Pipeline,     # obrigatório: predict_proba, classes_ == [0, 1] e
                              # feature_names_in_ == FEATURE_COLUMNS (23 colunas, com PAY_0)
    "metricas": {             # obrigatório: avaliação no conjunto de teste
        "algoritmo": str, "total_amostras_teste": int,
        "acuracia": float, "precisao": float,
        "recall_sensibilidade": float, "f1_score": float,
        "matriz_confusao": {"VP": int, "VN": int, "FP": int, "FN": int},
        # informativas (ignoradas pela API): "roc_auc", "pr_auc", "roc_auc_chute",
        # "roc_auc_cv_treino"
    },
    "metadados": {            # opcional (padrões abaixo)
        "sklearn_version": str,      # versão usada no treino
        "treinado_em": str,          # data/hora ISO 8601 (UTC)
        "csv_origem": str,           # CSV lido no treino (relativo à raiz do projeto, se dentro dele)
        "limiar_decisao": float,     # 0 < limiar < 1; padrão 0.5
        # informativos (gravados pelo script, ignorados pela API):
        "atributos_sensiveis_usados": bool,  # False: SEX, AGE e MARRIAGE descartados no ColumnTransformer
        "modelo": str, "hiperparametros": dict, "custo_fn": float, "custo_fp": float,
        "n_folds_cv": int, "modo_rapido": bool,
    },
}, caminho, compress=3)
```

- As métricas são calculadas **uma única vez** no conjunto de teste (20%, `random_state=42`,
  `stratify=y`), com o mesmo `limiar_decisao` que a API usa. O limiar é escolhido pelo script
  (custo dos erros em validação cruzada no treino), nunca olhando o teste.
- Mesmo sem os atributos sensíveis, o pipeline recebe as 23 colunas: eles são descartados dentro
  do `ColumnTransformer` (transformador `"descartados"`).

- Chaves adicionais são ignoradas. Um `Pipeline` puro (sem dicionário) é rejeitado.
- `limiar_decisao` define a classe e a faixa de risco em `POST /predict`:
  classe 1 (Risco Alto) se escore >= limiar; Moderado se min(0.35, limiar) <= escore < limiar;
  Baixo abaixo disso. Mudar o limiar não exige alteração na API, desde que ele seja maior
  que 0,35: com limiar <= 0,35 a faixa Moderado desaparece e `classificar_risco` precisaria de
  faixas proporcionais ao limiar (o script de treino avisa quando isso acontece).
- **Casas decimais do limiar (restrição de exibição):** `POST /predict` devolve
  `probabilidade_atraso` (4 casas) e `probabilidade_percentual` (1 casa) **truncados para baixo**,
  enquanto a classe usa o escore bruto. Isso garante que o valor exibido fica do mesmo lado do
  corte que a classe **somente se `limiar_decisao` tiver no máximo 3 casas decimais** (1 casa no
  percentual = 3 casas na probabilidade). Os cortes avaliados pelo script (0,10 a 0,90, passo
  0,05) atendem. Com um corte como 0,5037, o escore 0,50375 é classe 1 mas aparece como 50,3%.
  A API registra um aviso no log ao carregar um artefato assim, sem bloquear o carregamento.
- Artefato ausente, vazio, corrompido ou fora do contrato: `/predict` e `/metrics`
  retornam HTTP 503 e `GET /` informa `modelo_carregado: false`.

**Gerar:** `python backend/analise_e_modelagem_c318.py` (de qualquer pasta). Destino padrão:
`backend/modelo_credito.joblib`; use `MODEL_OUTPUT_PATH` para gravar em outro caminho.
**Carregar outro artefato na API:** defina `MODEL_PATH` antes de iniciar o uvicorn.
**Compatibilidade:** gere e carregue com as versões de `requirements.txt`.
