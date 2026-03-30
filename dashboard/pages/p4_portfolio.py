import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.exceptions import OptimizationError
from pages.p1_macro_monitor import load_data, filter_by_lookback
from src.viz.theme import COLORS, PLOTLY_BASE_LAYOUT

@st.cache_data(show_spinner=False)
def run_optimization(df: pd.DataFrame, selected_assets: list):
    df_prices = df[selected_assets].dropna()
    selic_diaria_decimal = df['selic_diaria'].dropna().iloc[-1] / 100.0
    rf_rate = ((1 + selic_diaria_decimal) ** 252) - 1
    
    mu = expected_returns.mean_historical_return(df_prices)
    S = risk_models.CovarianceShrinkage(df_prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.30))
    ef.max_sharpe(risk_free_rate=rf_rate)
    w_markowitz = ef.clean_weights()
    perf_markowitz = ef.portfolio_performance(risk_free_rate=rf_rate)
    
    returns = df_prices.pct_change().dropna()
    hrp = HRPOpt(returns)
    hrp.optimize()
    w_hrp = hrp.clean_weights()
    perf_hrp = hrp.portfolio_performance()
    
    return w_markowitz, perf_markowitz, w_hrp, perf_hrp, rf_rate

def render_page():
    st.header("Portfolio Optimizer")
    st.markdown("Alocação de capital Macro-Aware: Teoria Clássica de Markowitz vs. Machine Learning (HRP).")
    
    st.subheader("Configuração do Portfólio")
    ALL_ASSETS = {
        "IBOVESPA (^BVSP)": "^BVSP",
        "Bitcoin (BTC)":     "BTC-USD",
        "Ethereum (ETH)":    "ETH-USD",
        "Litecoin (LTC)":    "LTC-USD",
    }
    selected_labels = st.multiselect("Ativos a incluir", options=list(ALL_ASSETS.keys()), default=list(ALL_ASSETS.keys()))
    selected_assets = [ALL_ASSETS[l] for l in selected_labels]
    
    if len(selected_assets) < 2:
        st.warning("Selecione pelo menos 2 ativos para a otimização.")
        st.stop()
    st.write("")
    
    try:
        df_full = load_data()
        df = filter_by_lookback(df_full, st.session_state.global_lookback)
        
        if len(df) < 50:
            st.warning("⚠️ Janela muito curta. Selecione pelo menos 1Y na barra lateral.")
            return
            
        with st.spinner("Calculando fronteiras..."):
            w_m, p_m, w_h, p_h, rf = run_optimization(df, selected_assets)
            
        st.caption(f"Taxa Livre de Risco (Selic Anualizada) base: **{rf:.2%}**")
        
        c1, c2 = st.columns(2)
        pie_colors = [COLORS.get(k.replace('-USD','').lower().replace('^',''), COLORS["neutral"]) for k in selected_assets]
        
        with c1:
            st.subheader("Markowitz (Max Sharpe)")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Retorno Esperado", f"{p_m[0]:.1%}")
            mc2.metric("Volatilidade", f"{p_m[1]:.1%}")
            mc3.metric("Sharpe Ratio", f"{p_m[2]:.2f}", help="Retorno por unidade de risco (base Selic). >1 bom, >2 excelente.")
            
            fig_m = go.Figure(data=[go.Pie(labels=list(w_m.keys()), values=list(w_m.values()), hole=.4)])
            fig_m.update_traces(hoverinfo='label+percent', textinfo='percent+label', marker=dict(colors=pie_colors))
            fig_m.update_layout(**PLOTLY_BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)
            
        with c2:
            st.subheader("Hierarchical Risk Parity")
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Retorno Esperado", f"{p_h[0]:.1%}")
            hc2.metric("Volatilidade", f"{p_h[1]:.1%}")
            hc3.metric("Sharpe Ratio", f"{p_h[2]:.2f}", help="Retorno por unidade de risco (base Selic). >1 bom, >2 excelente.")
            
            fig_h = go.Figure(data=[go.Pie(labels=list(w_h.keys()), values=list(w_h.values()), hole=.4)])
            fig_h.update_traces(hoverinfo='label+percent', textinfo='percent+label', marker=dict(colors=pie_colors))
            fig_h.update_layout(**PLOTLY_BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

    except OptimizationError:
        st.error("Otimização inviável na janela selecionada.")
        st.info("Tente ampliar a janela histórica ou remover ativos com retorno negativo no período.")
    except Exception as e:
        st.error(f"Erro inesperado: {e}")