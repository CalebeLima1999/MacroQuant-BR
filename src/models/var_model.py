import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Configuração rigorosa de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("docs/results")

def prepare_var_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma variáveis I(1) em I(0) e seleciona o bloco económico estrutural.
    A ordenação das colunas define a identificação de Cholesky.
    """
    df_var = pd.DataFrame(index=df.index)
    
    # A Selic em nível é I(1). O modelo exige a primeira diferença (d_selic)
    df_var['d_selic'] = df['selic_diaria'].diff()
    
    # Retornos já foram validados como I(0) pelo teste ADF
    df_var['ret_brl'] = df['ret_brl=x']
    df_var['ret_bvsp'] = df['ret_bvsp']
    df_var['ret_btc'] = df['ret_btc-usd']
    
    # O dropna é seguro aqui porque removemos o M2 (que causava o corte para 98 linhas)
    return df_var.dropna()

def run_var_model() -> None:
    """Orquestra a estimação do modelo VAR, IRF e FEVD."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dataset_path = PROCESSED_DIR / "dataset_final.parquet"
        df = pd.read_parquet(dataset_path)
        
        df_var = prepare_var_data(df)
        logger.info(f"Dataset do VAR estabilizado. Shape de treino: {df_var.shape}")
        
        # 1. Ajuste do Modelo VAR
        model = VAR(df_var)
        
        # Seleção automática de Lags pelo critério de Akaike (AIC)
        lag_selection = model.select_order(maxlags=8)
        optimal_lags = lag_selection.aic
        logger.info(f"Lags ótimos selecionados (AIC): {optimal_lags} semanas")
        
        results = model.fit(optimal_lags)
        logger.info("Modelo VAR convergido e ajustado com sucesso.")
        
        # 2. Funções de Impulso-Resposta Ortogonalizadas (IRF) - Horizonte de 12 semanas
        irf = results.irf(12)
        
        # Plot da resposta acumulada/simples do BTC a um choque na política monetária (Selic)
        fig = irf.plot(impulse='d_selic', response='ret_btc', figsize=(10, 6), orth=True)
        plt.title("Resposta do Bitcoin a um Choque Ortogonal na Taxa Selic", fontsize=14, pad=15)
        
        plot_path = RESULTS_DIR / "irf_selic_to_btc.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico da Função de Impulso-Resposta (IRF) guardado em: {plot_path}")
        
        # 3. Decomposição da Variância do Erro de Previsão (FEVD)
        fevd = results.fevd(12)
        logger.info("==========================================================")
        logger.info("DECOMPOSIÇÃO DE VARIÂNCIA DO BITCOIN (12 SEMANAS À FRENTE)")
        logger.info("==========================================================")
        
        # Extrai a última linha (semana 12) da variável de índice 3 (ret_btc)
        btc_variance = fevd.decomp[-1, 3, :] 
        logger.info(f"Choque na Selic (Política Monetária): {btc_variance[0]*100:.2f}% da variância")
        logger.info(f"Choque no Câmbio (Risco Cambial)    : {btc_variance[1]*100:.2f}% da variância")
        logger.info(f"Choque no IBOV (Risco Local B3)     : {btc_variance[2]*100:.2f}% da variância")
        logger.info(f"Choque Idiossincrático (Próprio BTC): {btc_variance[3]*100:.2f}% da variância")

    except Exception as e:
        logger.error(f"Falha crítica na execução do modelo VAR: {e}")
        raise e

if __name__ == "__main__":
    run_var_model()