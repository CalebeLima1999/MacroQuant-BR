import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("docs/results")

def load_data_for_hmm() -> pd.DataFrame:
    """Carrega os dados processados para a modelagem de regimes."""
    dataset_path = PROCESSED_DIR / "dataset_final.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {dataset_path}. Rode o M2 primeiro.")
    
    df = pd.read_parquet(dataset_path)
    # Selecionamos o Preço (para plotagem) e o Retorno (para o treino do HMM)
    df_hmm = df[['BTC-USD', 'ret_btc-usd']].dropna()
    return df_hmm

def train_and_predict_hmm(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Treina o Hidden Markov Model usando os log-retornos.
    n_components=2 geralmente converge para: Regime 0 (Baixa Volatilidade) e Regime 1 (Alta Volatilidade).
    """
    logger.info(f"Treinando Hidden Markov Model ({n_components} regimes) para o BTC-USD...")
    
    # O hmmlearn exige um array 2D
    X = df['ret_btc-usd'].values.reshape(-1, 1)
    
    # Instancia e treina o modelo Gaussiano
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Previsão dos estados ocultos (Regimes)
    states = model.predict(X)
    df['regime_hmm'] = states
    
    # --- CORREÇÃO DE LABEL SWITCHING ---
    # Os algoritmos de clusterização dão nomes aleatórios aos regimes (0 ou 1).
    # Nós forçamos o Regime 0 a ser sempre o de MENOR volatilidade para consistência.
    vol_regime_0 = df[df['regime_hmm'] == 0]['ret_btc-usd'].std()
    vol_regime_1 = df[df['regime_hmm'] == 1]['ret_btc-usd'].std()
    
    if vol_regime_0 > vol_regime_1:
        logger.info("Ajustando labels: Regime 0 será o de Baixa Vol, Regime 1 será o de Alta Vol.")
        df['regime_hmm'] = df['regime_hmm'].map({0: 1, 1: 0})
        # Recalcula as vols para o log ficar correto
        vol_regime_0, vol_regime_1 = vol_regime_1, vol_regime_0
        
    logger.info(f"Regime 0 (Baixa Volatilidade / Estável) -> Vol Semanal: {vol_regime_0:.2%}")
    logger.info(f"Regime 1 (Alta Volatilidade / Crise)   -> Vol Semanal: {vol_regime_1:.2%}")
    
    return df

def plot_regimes(df: pd.DataFrame) -> None:
    """Gera o gráfico pintando o preço do BTC de acordo com o regime identificado."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Mapeamento de cores: Cinza para Regime 0 (Tranquilo), Vermelho/Laranja para Regime 1 (Volátil)
    colors = {0: 'lightgray', 1: '#FF4B4B'}
    
    # Para pintar as linhas contínuas, iteramos sobre os blocos de regimes
    for i in range(1, len(df)):
        ax.plot(df.index[i-1:i+1], df['BTC-USD'].iloc[i-1:i+1], 
                color=colors[df['regime_hmm'].iloc[i]], linewidth=2)
                
    # Elementos de UI do gráfico
    ax.set_title('Regimes Ocultos de Mercado do Bitcoin (Hidden Markov Model)', fontsize=16, pad=15)
    ax.set_yscale('log') # Eixo Log é mandatório para BTC a longo prazo
    ax.set_ylabel('Preço (USD) - Escala Logarítmica', fontsize=12)
    ax.grid(True, alpha=0.1)
    
    # Legenda customizada
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lightgray', lw=3),
                    Line2D([0], [0], color='#FF4B4B', lw=3)]
    ax.legend(custom_lines, ['Regime 0: Baixa Vol (Estável)', 'Regime 1: Alta Vol (Turbulência)'], loc='upper left')
    
    plot_path = RESULTS_DIR / "btc_hmm_regimes.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico de Regimes salvo em: {plot_path}")

def run_ml_pipeline() -> None:
    try:
        df = load_data_for_hmm()
        df_regimes = train_and_predict_hmm(df, n_components=2)
        plot_regimes(df_regimes)
        
        # Salva o output para ser consumido instantaneamente pelo Front-end (Streamlit)
        output_path = PROCESSED_DIR / "btc_regimes.parquet"
        df_regimes.to_parquet(output_path)
        logger.info(f"Tabela de Regimes salva em: {output_path}")
        logger.info("M7 CONCLUÍDO! Back-end quantitativo 100% finalizado.")
        
    except Exception as e:
        logger.error(f"Falha na execução do pipeline de ML: {e}")
        raise e

if __name__ == "__main__":
    run_ml_pipeline()