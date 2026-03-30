import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from src.data.schemas import MarketDataSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
CACHE_FILE = DATA_DIR / "market_data_tidy.parquet"

# Unificamos TUDO no YFinance
TICKERS = [
    "^BVSP",      # IBOVESPA
    "IFIX.SA",    # FIIs
    "IMAT.SA",    # Materiais Básicos
    "BRL=X",      # Câmbio USD/BRL
    "BTC-USD",    # Bitcoin
    "ETH-USD",    # Ethereum
    "LTC-USD"     # Litecoin
]

def fetch_yfinance_data(start_date: str) -> pd.DataFrame:
    logger.info(f"Coletando mercado e cripto via Yahoo Finance: {TICKERS}")
    try:
        df_yf = yf.download(TICKERS, start=start_date, progress=False, threads=True)
        
        if df_yf.empty:
            raise ValueError("Yahoo Finance retornou vazio.")

        price_col = 'Adj Close' if 'Adj Close' in df_yf.columns else 'Close'
        
        # Stacking transforma as colunas em linhas (Tidy format)
        df_close = df_yf[price_col].stack().reset_index()
        df_close.columns = ['data', 'ticker', 'fechamento']
        
        df_vol = df_yf['Volume'].stack().reset_index()
        df_vol.columns = ['data', 'ticker', 'volume']
        
        df_tidy = pd.merge(df_close, df_vol, on=['data', 'ticker'], how='left')
        df_tidy = df_tidy.set_index('data')
        
        # Removemos os Timezones para evitar conflitos na Modelagem
        if df_tidy.index.tz is not None:
            df_tidy.index = df_tidy.index.tz_localize(None)
            
        return df_tidy

    except Exception as e:
        logger.error(f"Falha na extração do YFinance: {e}")
        raise RuntimeError(f"YFinance indisponível: {e}")

def build_market_dataset(start_date: str = "2015-01-01", force_update: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force_update:
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_mod_time < timedelta(days=7):
            logger.info("Carregando mercado do cache local...")
            return pd.read_parquet(CACHE_FILE)

    df_final = fetch_yfinance_data(start_date)
    df_final = df_final.sort_index()

    logger.info("Validando schema via Pandera...")
    df_validated = MarketDataSchema.validate(df_final)
    
    df_validated.to_parquet(CACHE_FILE)
    logger.info(f"Dados salvos em {CACHE_FILE}. Total: {len(df_validated)} linhas.")
    
    return df_validated

if __name__ == "__main__":
    df_market = build_market_dataset(force_update=True)