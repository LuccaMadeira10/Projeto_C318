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
