"""
M4 — Teste de Cointegração de Johansen.

Identifica relações de equilíbrio de longo prazo entre séries I(1),
determinando se um modelo VECM é apropriado em vez de um VAR em diferenças.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# Rótulos padrão dos níveis de significância retornados por coint_johansen
_SIG_LABELS: list[str] = ["90%", "95%", "99%"]


_MAX_JOHANSEN_VARS: int = 12


def select_core_block(
    df: pd.DataFrame,
    core_vars: list[str],
) -> pd.DataFrame:
    """
    Filtra o DataFrame para um subconjunto de variáveis centrais (core block),
    validando existência e removendo NaN residuais.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo (pode ter mais de 12 colunas).
    core_vars : list[str]
        Lista de nomes de colunas que compõem o bloco temático a testar.

    Returns
    -------
    pd.DataFrame  filtrado, sem NaN, apenas com as colunas solicitadas.
    """
    missing = [v for v in core_vars if v not in df.columns]
    if missing:
        raise KeyError(
            f"Variáveis ausentes no DataFrame: {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df_core = df[core_vars].dropna()

    logger.info(
        "Core block selecionado — colunas: %s | shape após dropna: %s",
        core_vars,
        df_core.shape,
    )
    return df_core


def run_johansen_test(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Executa o teste de cointegração de Johansen no DataFrame multivariado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com séries I(1). Apenas colunas numéricas; sem NaN.
        Máximo de 12 variáveis (limite do statsmodels).
    det_order : int
        Ordem da tendência determinística:
         -1 = sem constante/tendência,
          0 = constante restrita ao espaço de cointegração (padrão),
          1 = tendência linear.
    k_ar_diff : int
        Número de lags em diferenças (default=1).

    Returns
    -------
    dict com chaves:
        trace_results  : pd.DataFrame  (Trace Statistic)
        max_eig_results: pd.DataFrame  (Max-Eigenvalue Statistic)
        n_coint_trace  : int (nº de vetores de cointegração pelo Trace a 5 %)
        n_coint_maxeig : int (nº de vetores de cointegração pelo Max-Eigen a 5 %)
    """
    df_clean = df.select_dtypes(include=[np.number]).dropna()

    if df_clean.shape[1] < 2:
        raise ValueError("O teste de Johansen exige pelo menos 2 séries numéricas.")

    if df_clean.shape[1] > _MAX_JOHANSEN_VARS:
        raise ValueError(
            f"statsmodels.coint_johansen suporta no máximo {_MAX_JOHANSEN_VARS} variáveis, "
            f"mas recebeu {df_clean.shape[1]}. Use select_core_block() para reduzir."
        )

    logger.info(
        "Rodando Johansen — %d séries, %d obs, det_order=%d, k_ar_diff=%d",
        df_clean.shape[1],
        df_clean.shape[0],
        det_order,
        k_ar_diff,
    )

    try:
        result = coint_johansen(df_clean, det_order=det_order, k_ar_diff=k_ar_diff)
    except Exception as exc:
        logger.error("Johansen falhou: %s", exc)
        raise

    n_vars = df_clean.shape[1]
    var_names = list(df_clean.columns)

    # --- Trace Statistic ---
    trace_rows: list[dict] = []
    n_coint_trace = 0
    for i in range(n_vars):
        stat = result.lr1[i]
        cvs = result.cvt[i]  # [90%, 95%, 99%]
        reject_95 = stat > cvs[1]
        if reject_95:
            n_coint_trace += 1
        trace_rows.append(
            {
                "H0": f"r <= {i}",
                "Trace_Stat": round(float(stat), 4),
                "CV_90%": round(float(cvs[0]), 4),
                "CV_95%": round(float(cvs[1]), 4),
                "CV_99%": round(float(cvs[2]), 4),
                "Rejeita_H0_5%": "Sim" if reject_95 else "Não",
            }
        )

    # --- Max-Eigenvalue Statistic ---
    max_eig_rows: list[dict] = []
    n_coint_maxeig = 0
    for i in range(n_vars):
        stat = result.lr2[i]
        cvs = result.cvm[i]
        reject_95 = stat > cvs[1]
        if reject_95:
            n_coint_maxeig += 1
        max_eig_rows.append(
            {
                "H0": f"r <= {i}",
                "MaxEig_Stat": round(float(stat), 4),
                "CV_90%": round(float(cvs[0]), 4),
                "CV_95%": round(float(cvs[1]), 4),
                "CV_99%": round(float(cvs[2]), 4),
                "Rejeita_H0_5%": "Sim" if reject_95 else "Não",
            }
        )

    trace_df = pd.DataFrame(trace_rows)
    maxeig_df = pd.DataFrame(max_eig_rows)

    logger.info(
        "Johansen concluído — Vetores de cointegração: Trace=%d, MaxEig=%d",
        n_coint_trace,
        n_coint_maxeig,
    )

    return {
        "trace_results": trace_df,
        "max_eig_results": maxeig_df,
        "n_coint_trace": n_coint_trace,
        "n_coint_maxeig": n_coint_maxeig,
        "variables": var_names,
    }


def format_johansen_report(johansen_output: dict) -> str:
    """Formata a saída de `run_johansen_test` como texto legível para console."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("TESTE DE COINTEGRAÇÃO DE JOHANSEN")
    lines.append(f"Variáveis: {johansen_output['variables']}")
    lines.append("=" * 90)

    lines.append("\n>> Estatística do Traço (Trace Statistic)")
    lines.append(johansen_output["trace_results"].to_string(index=False))
    lines.append(f"   => Vetores cointegrados (5%): {johansen_output['n_coint_trace']}")

    lines.append("\n>> Estatística de Máximo Autovalor (Max-Eigenvalue)")
    lines.append(johansen_output["max_eig_results"].to_string(index=False))
    lines.append(f"   => Vetores cointegrados (5%): {johansen_output['n_coint_maxeig']}")

    lines.append("=" * 90)
    return "\n".join(lines)


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

    logger.info("Dataset carregado — shape bruto: %s | colunas: %s", df.shape, list(df.columns))

    # Bloco temático central: tese Selic Real + M2 → Cripto
    core_vars: list[str] = ["BTC-USD", "ETH-USD", "selic_real_ex_ante", "m2"]

    df_core = select_core_block(df, core_vars)

    logger.info(
        "Iniciando Johansen no core block — shape: %s | colunas: %s",
        df_core.shape,
        list(df_core.columns),
    )

    result = run_johansen_test(df_core, det_order=0, k_ar_diff=1)
    print("\n" + format_johansen_report(result) + "\n")
