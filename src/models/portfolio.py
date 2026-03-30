import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# PyPortfolioOpt para otimização institucional
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("docs/results")

def load_portfolio_data() -> tuple[pd.DataFrame, float]:
    """
    Carrega os preços históricos e define a taxa livre de risco (Risk-Free Rate).
    O universo foi ajustado para remover ativos com falha de API (IFIX).
    """
    dataset_path = PROCESSED_DIR / "dataset_final.parquet"
    df = pd.read_parquet(dataset_path)
    
    # 1. Defesa: Usamos apenas ativos com liquidez e histórico validados no M4
    assets = ['^BVSP', 'BTC-USD', 'ETH-USD', 'LTC-USD']
    
    # Garante que só tentamos puxar colunas que realmente existem no parquet
    valid_assets = [col for col in assets if col in df.columns]
    df_prices = df[valid_assets].dropna()
    
    # 2. Matemática Financeira: A Selic (SGS 11) é diária (ex: 0.042%). 
    # Convertemos para decimal e anualizamos (252 dias úteis).
    selic_diaria_decimal = df['selic_diaria'].dropna().iloc[-1] / 100.0
    rf_rate = ((1 + selic_diaria_decimal) ** 252) - 1
    
    return df_prices, rf_rate

def optimize_markowitz(df_prices: pd.DataFrame, rf_rate: float) -> dict:
    """Otimização Clássica: Fronteira Eficiente de Markowitz (Max Sharpe)."""
    logger.info("A iniciar otimização via Markowitz (Max Sharpe)...")
    
    # Média histórica para retornos e Shrinkage de Ledoit-Wolf para a Covariância (estabilidade)
    mu = expected_returns.mean_historical_return(df_prices)
    S = risk_models.CovarianceShrinkage(df_prices).ledoit_wolf()
    
    # Restrição correta: weight_bounds=(min, max) fixa o limite de 30% por ativo
    # Isso impede que o Markowitz coloque 100% no ativo de maior Sharpe
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.30))
    
    # Otimiza o portfólio para o maior prémio de risco possível
    raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
    cleaned_weights = ef.clean_weights()
    
    logger.info(f"Pesos Markowitz: {cleaned_weights}")
    print("\n--- PERFORMANCE MARKOWITZ (IN-SAMPLE) ---")
    ef.portfolio_performance(verbose=True, risk_free_rate=rf_rate)
    
    return cleaned_weights

def optimize_hrp(df_prices: pd.DataFrame) -> dict:
    """Otimização Machine Learning: Hierarchical Risk Parity (HRP)."""
    logger.info("A iniciar otimização via Hierarchical Risk Parity (HRP)...")
    
    # O HRP exige retornos simples
    returns = df_prices.pct_change().dropna()
    
    hrp = HRPOpt(returns)
    raw_weights = hrp.optimize()
    cleaned_weights = hrp.clean_weights()
    
    logger.info(f"Pesos HRP: {cleaned_weights}")
    print("\n--- PERFORMANCE HRP (IN-SAMPLE) ---")
    hrp.portfolio_performance(verbose=True)
    
    return cleaned_weights

def plot_allocations(w_markowitz: dict, w_hrp: dict) -> None:
    """Gera uma visualização comparativa clara para relatórios e dashboards."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Estilização visual (mantendo a sobriedade institucional)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filtra ativos residuais (peso < 1%)
    w_m_filt = {k: v for k, v in w_markowitz.items() if v > 0.01}
    ax1.pie(w_m_filt.values(), labels=w_m_filt.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax1.set_title('Markowitz (Max Sharpe + Ledoit-Wolf)', fontsize=14, pad=15)
    
    w_h_filt = {k: v for k, v in w_hrp.items() if v > 0.01}
    ax2.pie(w_h_filt.values(), labels=w_h_filt.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax2.set_title('HRP (Hierarchical Risk Parity)', fontsize=14, pad=15)
    
    plt.suptitle("Comparação de Otimização: Portfólio Macro-Aware (B3 + Cripto)", fontsize=18, y=1.05)
    
    plot_path = RESULTS_DIR / "portfolio_allocation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info(f"Gráfico de alocação guardado em: {plot_path}")

def run_portfolio_optimization() -> None:
    """Função orquestradora do Módulo 6."""
    try:
        df_prices, rf_rate = load_portfolio_data()
        logger.info(f"Dados carregados. Taxa Livre de Risco (Selic base): {rf_rate:.2%}")
        
        w_markowitz = optimize_markowitz(df_prices, rf_rate)
        w_hrp = optimize_hrp(df_prices)
        
        plot_allocations(w_markowitz, w_hrp)
        
    except Exception as e:
        logger.error(f"Falha na otimização do portfólio: {e}")
        raise e

if __name__ == "__main__":
    run_portfolio_optimization()