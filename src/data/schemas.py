import pandera.pandas as pa  # <-- Import atualizado para remover o FutureWarning
from pandera.typing import Series, Index
import pandas as pd

class BCBMacroSchema(pa.DataFrameModel):
    """Schema de validação para os dados extraídos do SGS/BCB."""
    date: Index[pd.Timestamp] = pa.Field(unique=True, check_name=True)
    selic_diaria: Series[float] = pa.Field(ge=0.0, le=50.0, nullable=True)
    cambio_brl_usd: Series[float] = pa.Field(gt=0.0, nullable=True)
    ibc_br: Series[float] = pa.Field(gt=0.0, nullable=True)
    ipca_mensal: Series[float] = pa.Field(ge=-5.0, le=10.0, nullable=True)

    class Config:
        strict = False 
        coerce = True  

class FocusSchema(pa.DataFrameModel):
    """Schema de validação para a API OData de Expectativas (Relatório Focus)."""
    Data: Index[pd.Timestamp] = pa.Field(check_name=True)
    Mediana: Series[float] = pa.Field(ge=-2.0, le=25.0, nullable=False)
    Indicador: Series[str] = pa.Field(isin=["IPCA"])

    class Config:
        strict = False
        coerce = True

class MarketDataSchema(pa.DataFrameModel):
    """Schema Tidy (Longo) para dados de mercado (B3, Global e Cripto)."""
    data: Index[pd.Timestamp] = pa.Field(check_name=True)
    ticker: Series[str] = pa.Field(nullable=False)
    fechamento: Series[float] = pa.Field(gt=0.0, nullable=False)
    volume: Series[float] = pa.Field(ge=0.0, nullable=True)

    class Config:
        strict = False
        coerce = True