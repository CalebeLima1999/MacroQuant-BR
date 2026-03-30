# 📊 MacroQuant-BR: Institutional Dashboard & Quant Pipeline

Uma arquitetura quantitativa *end-to-end* que cruza dados macroeconômicos brasileiros (Banco Central) com ativos de risco e criptomoedas. Desenhado para análise de portfólio, detecção de regimes e simulação de choques de liquidez.

## 🧠 Arquitetura e Modelagem

O projeto é dividido em um *Back-end* de engenharia de dados/modelagem e um *Front-end* analítico:

1. **Ingestão e Limpeza (ETL):** Extração assíncrona da API SGS do Banco Central do Brasil (Selic, IPCA, M2, IBC-Br, Focus) e do Yahoo Finance (BTC, ETH, IBOV, Câmbio).
2. **Econometria e Raiz Unitária:** Testes ADF e KPSS para validação de estacionariedade (I(0) e I(1)). Teste de Cointegração de Johansen.
3. **Análise de Choques (VAR):** Modelo de Vetores Autorregressivos (VAR) para mapear o impacto ortogonalizado (Cholesky) de choques na taxa Selic sobre a ação de preço do Bitcoin via Funções de Impulso-Resposta (IRF) e Decomposição da Variância de Erros (FEVD).
4. **Machine Learning (HMM):** Algoritmo não-supervisionado *Hidden Markov Model* para detecção empírica de regimes de mercado (Alta Volatilidade vs. Baixa Volatilidade) baseado na ação do preço, mitigando o viés analítico.
5. **Portfolio Optimizer:** Alocação de capital Macro-Aware cruzando a teoria clássica de **Markowitz (Max Sharpe + Ledoit-Wolf Shrinkage)** com a abordagem robusta de Machine Learning do **Hierarchical Risk Parity (HRP)**.

## 🛠️ Stack Tecnológica

* **Dados & Engenharia:** `pandas`, `numpy`, `pyarrow` (Parquet).
* **Econometria & Estatística:** `statsmodels`, `scikit-learn`, `PyPortfolioOpt`.
* **Machine Learning:** `hmmlearn`.
* **Visualização & UI:** `streamlit`, `plotly`.
* **Gestão de Ambiente:** `uv`.

## 🚀 Como Executar Localmente

```bash
# 1. Clone o repositório
git clone [https://github.com/CalebeLima1999/MacroQuant-BR.git](https://github.com/CalebeLima1999/MacroQuant-BR.git)
cd MacroQuant-BR

# 2. Crie e ative o ambiente virtual usando o 'uv'
uv venv
source .venv/Scripts/activate # Windows (Git Bash)

# 3. Instale as dependências
uv pip install -r requirements.txt

# 4. Inicie o Dashboard Institucional
streamlit run dashboard/app.py