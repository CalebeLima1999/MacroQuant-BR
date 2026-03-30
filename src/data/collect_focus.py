import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Nova lib do BCB para acessar o OData do Focus
from bcb import Expectativas
from src.data.schemas import FocusSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
CACHE_FILE = DATA_DIR / "bcb_focus_ipca12m.parquet"

def fetch_focus_ipca_12m(start_date: str = "2015-01-01", force_update: bool = False) -> pd.DataFrame:
    """
    Coleta expectativas de mercado (Relatório Focus) via API OData do BCB.
    Filtra especificamente para IPCA Suavizado 12 Meses à frente.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force_update:
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_mod_time < timedelta(days=7):
            logger.info("Carregando dados do Focus do cache local (Parquet)...")
            return pd.read_parquet(CACHE_FILE)

    logger.info(f"Buscando expectativas Focus (OData) a partir de {start_date}...")
    
    try:
        # Instancia o cliente de Expectativas
        em = Expectativas()
        
        # Seleciona o endpoint específico para inflação 12 meses suavizada
        ep = em.get_endpoint('ExpectativasMercadoInflacao12Meses')
        
        # Constrói a query OData (executa o filtro no servidor do BCB, economizando RAM)
        df = (ep.query()
              .filter(ep.Indicador == 'IPCA')
              .filter(ep.Suavizada == 'S')
              .filter(ep.Data >= start_date)
              .select(ep.Data, ep.Indicador, ep.Mediana, ep.Media, ep.DesvioPadrao)
              .collect())
        
        if df.empty:
            raise ValueError("A query OData retornou um DataFrame vazio.")

        # Tratamento de índice e tipos para o Pandera
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.set_index('Data')
        df = df.sort_index()

        # Validação do contrato de dados
        logger.info("Validando schema do Focus via Pandera...")
        df_validated = FocusSchema.validate(df)
        
        # Remove duplicatas caso o BCB publique correções no mesmo dia
        df_validated = df_validated[~df_validated.index.duplicated(keep='last')]
        
        df_validated.to_parquet(CACHE_FILE)
        logger.info(f"Dados do Focus salvos com sucesso em {CACHE_FILE}. Total de linhas: {len(df_validated)}")
        
        return df_validated

    except Exception as e:
        logger.error(f"Falha crítica ao extrair dados OData do Focus: {e}")
        # Nunca usamos 'pass'. O pipeline deve quebrar alto e claro se a fonte secar.
        raise RuntimeError(f"Pipeline de ingestão Focus falhou: {e}")

if __name__ == "__main__":
    df_focus = fetch_focus_ipca_12m(force_update=True)
    print("\nAmostra das expectativas de IPCA 12m:")
    print(df_focus[['Indicador', 'Mediana']].tail())