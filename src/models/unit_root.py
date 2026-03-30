"""
M4 — Testes de Raiz Unitária (ADF, KPSS, Zivot-Andrews).

Determina a ordem de integração de cada série temporal do dataset_final,
condição necessária para a especificação correta de modelos VAR / VECM.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schwert_max_lags(n_obs: int) -> int:
    """Regra de Schwert (1989) para o número máximo de lags: int(12 * (T/100)^(1/4))."""
    return int(12.0 * (n_obs / 100.0) ** 0.25)


# ---------------------------------------------------------------------------
# Testes individuais
# ---------------------------------------------------------------------------

def run_adf_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Teste Augmented Dickey-Fuller com seleção automática de lags via AIC
    e limite máximo definido pela regra de Schwert.

    Returns
    -------
    dict com chaves: statistic, p_value, lags_used, reject_null
    """
    series_clean = series.dropna()
    max_lags = _schwert_max_lags(len(series_clean))
    try:
        adf_stat, p_value, lags_used, n_obs, crit_values, _ = adfuller(
            series_clean,
            maxlag=max_lags,
            autolag="AIC",
            regression="ct",  # constante + tendência
        )
        return {
            "statistic": round(adf_stat, 4),
            "p_value": round(p_value, 4),
            "lags_used": lags_used,
            "reject_null": p_value < significance,
        }
    except Exception as exc:
        logger.warning("ADF falhou para '%s': %s", series.name, exc)
        return {"statistic": np.nan, "p_value": np.nan, "lags_used": np.nan, "reject_null": False}


def run_kpss_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Teste KPSS (H0: série é estacionária).

    Returns
    -------
    dict com chaves: statistic, crit_5pct, reject_null
    """
    series_clean = series.dropna()
    try:
        stat, p_value, lags_used, crit_values = kpss(
            series_clean, regression="ct", nlags="auto"
        )
        return {
            "statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "crit_5pct": round(crit_values["5%"], 4),
            "reject_null": stat > crit_values[f"{int(significance * 100)}%"],
        }
    except Exception as exc:
        logger.warning("KPSS falhou para '%s': %s", series.name, exc)
        return {"statistic": np.nan, "p_value": np.nan, "crit_5pct": np.nan, "reject_null": False}


def run_zivot_andrews_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Teste Zivot-Andrews para raiz unitária com quebra estrutural endógena.

    Returns
    -------
    dict com chaves: statistic, crit_5pct, break_date, reject_null
    """
    series_clean = series.dropna()
    try:
        result = zivot_andrews(series_clean, model="c", autolag="AIC")
        za_stat = result[0]
        p_value = result[1]
        crit_values = result[3]
        break_idx = result[4]

        # Resolve a data da quebra a partir do índice
        if isinstance(series_clean.index, pd.DatetimeIndex):
            break_date = series_clean.index[break_idx].strftime("%Y-%m-%d")
        else:
            break_date = str(break_idx)

        return {
            "statistic": round(za_stat, 4),
            "p_value": round(p_value, 4),
            "crit_5pct": round(crit_values["5%"], 4),
            "break_date": break_date,
            "reject_null": p_value < significance,
        }
    except Exception as exc:
        logger.warning("Zivot-Andrews falhou para '%s': %s", series.name, exc)
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "crit_5pct": np.nan,
            "break_date": None,
            "reject_null": False,
        }


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def run_unit_root_report(
    df: pd.DataFrame,
    significance: float = 0.05,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Executa ADF, KPSS e Zivot-Andrews em cada coluna numérica de *df* e devolve
    um DataFrame-relatório consolidado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset com index datetime e colunas numéricas (séries temporais).
    significance : float
        Nível de significância (padrão 5 %).
    columns : list[str] | None
        Subconjunto de colunas a testar. Se None, testa todas as numéricas.

    Returns
    -------
    pd.DataFrame  com colunas:
        Serie, ADF_Stat, ADF_pvalue, KPSS_Stat, KPSS_Result,
        ZA_Stat, ZA_BreakDate, Conclusao
    """
    target_cols = columns or [c for c in df.select_dtypes(include=[np.number]).columns]
    rows: list[dict] = []

    for col in target_cols:
        logger.info("Testando raiz unitária: %s", col)
        series = df[col]

        adf = run_adf_test(series, significance)
        kpss_res = run_kpss_test(series, significance)
        za = run_zivot_andrews_test(series, significance)

        # Conclusão combinada:
        # - ADF rejeita H0 (tem raiz unitária) => estacionária
        # - KPSS não rejeita H0 (é estacionária) => estacionária
        # Ambos concordam => I(0); caso contrário => I(1) ou inconclusivo
        adf_says_stationary = adf["reject_null"]
        kpss_says_stationary = not kpss_res["reject_null"]

        if adf_says_stationary and kpss_says_stationary:
            conclusion = "I(0) — Estacionária"
        elif not adf_says_stationary and not kpss_says_stationary:
            conclusion = "I(1) — Raiz Unitária"
        else:
            conclusion = "Inconclusivo (ADF/KPSS divergem)"

        rows.append(
            {
                "Serie": col,
                "ADF_Stat": adf["statistic"],
                "ADF_pvalue": adf["p_value"],
                "ADF_Lags": adf["lags_used"],
                "KPSS_Stat": kpss_res["statistic"],
                "KPSS_Result": "Rejeita H0 (não-est.)" if kpss_res["reject_null"] else "Não rejeita H0 (est.)",
                "ZA_Stat": za["statistic"],
                "ZA_BreakDate": za["break_date"],
                "Conclusao": conclusion,
            }
        )

    report = pd.DataFrame(rows)
    logger.info("Relatório de raiz unitária concluído para %d séries.", len(report))
    return report


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_path = PROCESSED_DIR / "dataset_final.parquet"
    logger.info("Carregando dataset: %s", dataset_path)

    try:
        df = pd.read_parquet(dataset_path)
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado. Execute a Fase 2 (clean.py) primeiro.", dataset_path)
        raise

    report = run_unit_root_report(df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("RELATÓRIO DE RAIZ UNITÁRIA (ADF / KPSS / Zivot-Andrews)")
    print("=" * 80)
    print(report.to_string(index=False))
    print("=" * 80 + "\n")
