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
    },
    "metadados": {            # opcional (padrões abaixo)
        "sklearn_version": str,      # versão usada no treino
        "treinado_em": str,          # data/hora ISO 8601 (UTC)
        "csv_origem": str,           # CSV usado no treino
        "limiar_decisao": float,     # 0 < limiar < 1; padrão 0.5
    },
}, caminho)
```

- Chaves adicionais são ignoradas. Um `Pipeline` puro (sem dicionário) é rejeitado.
- `limiar_decisao` define a classe e a faixa de risco em `POST /predict`:
  classe 1 (Risco Alto) se escore >= limiar; Moderado se min(0.35, limiar) <= escore < limiar;
  Baixo abaixo disso. Mudar o limiar não exige alteração na API.
- Artefato ausente, vazio, corrompido ou fora do contrato: `/predict` e `/metrics`
  retornam HTTP 503 e `GET /` informa `modelo_carregado: false`.

**Gerar:** `python backend/analise_e_modelagem_c318.py` (de qualquer pasta). Destino padrão:
`backend/modelo_credito.joblib`; use `MODEL_OUTPUT_PATH` para gravar em outro caminho.
**Carregar outro artefato na API:** defina `MODEL_PATH` antes de iniciar o uvicorn.
**Compatibilidade:** gere e carregue com as versões de `requirements.txt`.
