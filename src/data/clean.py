import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def _strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Força a conversão para UTC e remove o timezone para garantir merges perfeitos."""
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_macro = pd.read_parquet(RAW_DIR / "bcb_macro_data.parquet")
    df_focus = pd.read_parquet(RAW_DIR / "bcb_focus_ipca12m.parquet")
    df_market = pd.read_parquet(RAW_DIR / "market_data_tidy.parquet")
    return _strip_timezone(df_macro), _strip_timezone(df_focus), _strip_timezone(df_market)

def align_to_weekly(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    return df.resample('W-FRI').agg(method)

def process_market_data(df_market: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processando dados de mercado...")
    df_wide = df_market.pivot_table(index='data', columns='ticker', values='fechamento')
    df_weekly = align_to_weekly(df_wide, method='last')
    
    # DEFESA 1: Remove colunas (ativos) que o YFinance trouxe 100% vazias ANTES do cálculo
    df_weekly = df_weekly.dropna(axis=1, how='all')
    df_weekly = df_weekly.ffill(limit=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_returns = np.log(df_weekly / df_weekly.shift(1))
    
    log_returns.columns = [f"ret_{col.lower().replace('^', '').replace('.sa', '')}" for col in log_returns.columns]
    return pd.concat([df_weekly, log_returns], axis=1)

def process_macro_data(df_macro: pd.DataFrame, df_focus: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processando dados macroeconômicos e Focus...")
    df_macro_w = align_to_weekly(df_macro, method='last')
    df_macro_w['ibc_br'] = df_macro_w['ibc_br'].ffill()
    df_macro_w['ipca_mensal'] = df_macro_w['ipca_mensal'].ffill()
    
    df_focus_w = align_to_weekly(df_focus[['Mediana']], method='last')
    df_focus_w.rename(columns={'Mediana': 'focus_ipca_12m'}, inplace=True)
    df_focus_w = df_focus_w.ffill(limit=2)
    
    df_econ = pd.merge(df_macro_w, df_focus_w, left_index=True, right_index=True, how='left')
    df_econ['selic_real_ex_ante'] = df_econ['selic_diaria'] - df_econ['focus_ipca_12m']
    return df_econ

def build_final_dataset() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df_macro_raw, df_focus_raw, df_market_raw = load_raw_data()
    df_market_clean = process_market_data(df_market_raw)
    df_macro_clean = process_macro_data(df_macro_raw, df_focus_raw)
    
    dataset_final = pd.merge(df_market_clean, df_macro_clean, left_index=True, right_index=True, how='inner')
    logger.info(f"Shape inicial do Merge: {dataset_final.shape}")
    
    # DEFESA 2: Removemos variáveis mortas para salvar o dropna das linhas
    dataset_final = dataset_final.dropna(axis=1, how='all')
    
    ret_cols = [col for col in dataset_final.columns if col.startswith('ret_')]
    dataset_final = dataset_final.dropna(subset=ret_cols)
    
    output_path = PROCESSED_DIR / "dataset_final.parquet"
    dataset_final.to_parquet(output_path)
    
    logger.info(f"M2 CONCLUÍDO! Shape Final pronto para Econometria: {dataset_final.shape}")

if __name__ == "__main__":
    build_final_dataset()