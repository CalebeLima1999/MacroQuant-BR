import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from pages.p1_macro_monitor import load_data, filter_by_lookback
from src.viz.theme import COLORS, PLOTLY_BASE_LAYOUT, PLOTLY_AXES_STYLE

@st.cache_data(show_spinner=False)
def fit_var_model(df: pd.DataFrame):
    df_var = pd.DataFrame(index=df.index)
    df_var['d_selic'] = df['selic_diaria'].diff()
    df_var['ret_brl'] = df['ret_brl=x']
    df_var['ret_bvsp'] = df['ret_bvsp']
    df_var['ret_btc'] = df['ret_btc-usd']
    df_var = df_var.dropna()
    
    model = VAR(df_var)
    opt_lags = model.select_order(maxlags=8).aic
    results = model.fit(opt_lags)
    return results.irf(12), opt_lags, df_var.columns.tolist()

def render_page():
    st.header("Análise de Choques — Modelo VAR")
    st.markdown("Funções de Impulso-Resposta (IRF) Ortogonalizadas via decomposição de Cholesky.")
    
    try:
        df_full = load_data()
        df = filter_by_lookback(df_full, st.session_state.global_lookback)
        
        if len(df) < 50:
            st.warning("⚠️ Janela demasiado curta para estimação econométrica. Selecione pelo menos 1Y.")
            return

        with st.spinner("A estimar sistema de equações..."):
            irf, lags, col_names = fit_var_model(df)
            
        st.caption(f"Modelo estabilizado. Memória do mercado (AIC): **{lags} semanas**.")
        
        c1, c2 = st.columns(2)
        shock_var = c1.selectbox("Variável de Choque (Impulso)", col_names, index=0)
        resp_var = c2.selectbox("Variável Impactada (Resposta)", col_names, index=3)
        
        idx_shock = col_names.index(shock_var)
        idx_resp = col_names.index(resp_var)
        
        st.write("")
        st.subheader("Gráfico de Impulso-Resposta")
        
        mean_irf = irf.orth_irfs[:, idx_resp, idx_shock]
        stderr = irf.stderr(orth=True)[:, idx_resp, idx_shock]
        upper_bound = mean_irf + (1.96 * stderr)
        lower_bound = mean_irf - (1.96 * stderr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(13)), y=upper_bound, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=list(range(13)), y=lower_bound, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=COLORS['band'], name='IC 95%'))
        fig.add_trace(go.Scatter(x=list(range(13)), y=mean_irf, mode='lines+markers', line=dict(color=COLORS['highlight'], width=2), name='Resposta'))
        fig.add_hline(y=0, line_width=1, line_color=COLORS['border'])
        
        fig.update_layout(**PLOTLY_BASE_LAYOUT, xaxis_title="Semanas após o choque")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("")
        st.subheader("Interpretação Econômica")
        
        peak_idx   = int(np.argmax(np.abs(mean_irf)))
        peak_val   = float(mean_irf[peak_idx])
        direction  = "queda" if peak_val < 0 else "alta"
        sig_weeks  = [i for i in range(len(lower_bound)) if lower_bound[i] > 0 or upper_bound[i] < 0]
        
        col_interp, col_stats = st.columns([2, 1])
        with col_interp:
            st.info(
                f"Um choque de +1 desvio-padrão em **{shock_var}** está associado a uma "
                f"**{direction} máxima de {abs(peak_val):.4f}** em **{resp_var}** "
                f"na semana **{peak_idx}** após o choque. "
                + (f"O efeito é estatisticamente significante nas semanas: {sig_weeks}." 
                   if sig_weeks else "O efeito **não é estatisticamente diferente de zero** ao nível de 95%.")
            )
        with col_stats:
            st.metric("Semanas até o pico", f"{peak_idx}w")
            st.metric("Resposta cumulativa", f"{sum(mean_irf):.4f}")
            st.metric("Semanas significantes", str(len(sig_weeks)))

    except Exception as e:
        st.error(f"**Falha na Análise VAR:** {e}")