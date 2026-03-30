import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from bcb import sgs
from src.data.schemas import BCBMacroSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SGS_CODES = {
    'selic_diaria': 11,
    'cambio_brl_usd': 1,
    'ibc_br': 24364,
    'ipca_mensal': 433,
    'm1': 27788,
    'm2': 27789,
    'm3': 27790
}

DATA_DIR = Path("data/raw")
CACHE_FILE = DATA_DIR / "bcb_macro_data.parquet"

def fetch_sgs_data(start_date: str = "2015-01-01", force_update: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force_update:
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_mod_time < timedelta(days=7):
            logger.info("Carregando dados do BCB do cache local...")
            return pd.read_parquet(CACHE_FILE)

    logger.info("Buscando dados BCB (Micro-Chunking Anual para evitar Timeouts)...")
    
    try:
        start_year = int(start_date.split('-')[0])
        current_year = datetime.now().year
        
        chunks = []
        for year in range(start_year, current_year + 1):
            logger.info(f"Extraindo dados do SGS para o ano: {year}...")
            # Pede exatamente de 01/01 a 31/12 daquele ano
            df_year = sgs.get(SGS_CODES, start=f"{year}-01-01", end=f"{year}-12-31")
            chunks.append(df_year)
            
            # Pequeno delay (throttle) para não ser banido pela proteção de DDoS do governo
            time.sleep(0.5) 
            
        # Merge de todos os anos
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep='first')]

        df.index.name = "date"
        df.index = pd.to_datetime(df.index)
        
        logger.info("Validando schema dos dados recebidos...")
        df_validated = BCBMacroSchema.validate(df)
        
        df_validated.to_parquet(CACHE_FILE)
        logger.info(f"Dados salvos com sucesso em {CACHE_FILE}. Total de linhas: {len(df_validated)}")
        return df_validated

    except Exception as e:
        logger.error(f"Falha crítica ao obter dados do BCB: {e}")
        raise RuntimeError(f"Pipeline BCB falhou: {e}")

if __name__ == "__main__":
    df_macro = fetch_sgs_data(force_update=True)