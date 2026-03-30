import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from src.viz.theme import COLORS, PLOTLY_BASE_LAYOUT, PLOTLY_AXES_STYLE

@st.cache_data(show_spinner=False)
def load_regimes_data():
    try:
        return pd.read_parquet("data/processed/btc_regimes.parquet")
    except FileNotFoundError:
        return pd.DataFrame()

def filter_by_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if lookback == 'MAX': return df
    end_date = df.index[-1]
    val = int(lookback[:-1])
    unit = lookback[-1]
    if unit == 'M': start_date = end_date - relativedelta(months=val)
    elif unit == 'Y': start_date = end_date - relativedelta(years=val)
    return df[df.index >= start_date]

def render_page():
    st.header("Detector de Regimes — HMM")
    st.markdown("Identificação algorítmica de ciclos de mercado (*Hidden Markov Model*).")
    
    with st.spinner("A carregar inferências..."):
        df_full = load_regimes_data()
        
    if df_full.empty:
        st.error("**Pipeline de ML Pendente:** O ficheiro `btc_regimes.parquet` não foi encontrado.")
        return
        
    df = filter_by_lookback(df_full, st.session_state.global_lookback)
    if df.empty: return

    current_regime = df['regime_hmm'].iloc[-1]
    df['regime_changed'] = df['regime_hmm'].diff().ne(0)
    df['regime_block'] = df['regime_changed'].cumsum()
    current_block = df['regime_block'].iloc[-1]
    vol_atual = df[df['regime_block'] == current_block]['ret_btc-usd'].std() * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Regime Atual", "Crise" if current_regime == 1 else "Estável", delta="Ativo", delta_color="normal" if current_regime == 0 else "inverse")
    c2.metric("Volatilidade do Bloco", f"{vol_atual:.1f}%")
    c3.metric("Duração do Ciclo", f"{(df['regime_block'] == current_block).sum()} semanas")
    
    st.write("")
    st.subheader("Ação do Preço sob Fundo de Risco")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['BTC-USD'], mode='lines', line=dict(color=COLORS['btc'], width=2), name='Bitcoin (USD)'))
    
    # Legenda para os Vrects
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, color=COLORS['negative'], symbol='square', opacity=0.4), name="Regime de Crise (HMM)", showlegend=True))
    
    blocks = df[df['regime_hmm'] == 1].groupby('regime_block')
    for _, block in blocks:
        fig.add_vrect(x0=block.index[0], x1=block.index[-1], fillcolor=COLORS['negative'], opacity=0.15, layer="below", line_width=0)
    
    fig.update_layout(**PLOTLY_BASE_LAYOUT, yaxis_type="log")
    fig.update_yaxes(title_text="Preço (Escala Log)", **PLOTLY_AXES_STYLE)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("")
    st.subheader("Matriz de Transição de Markov")
    st.markdown("Probabilidade de transição entre regimes de uma semana para a próxima.")
    
    regimes = df['regime_hmm'].values
    trans_matrix = np.zeros((2, 2))
    for i in range(len(regimes) - 1):
        trans_matrix[int(regimes[i]), int(regimes[i+1])] += 1
        
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(trans_matrix, row_sums, out=np.zeros_like(trans_matrix), where=row_sums!=0)
    
    col_matrix, col_persist = st.columns([1.5, 1])
    with col_matrix:
        state_names = ["Estável", "Crise"]
        fig_trans = px.imshow(trans_matrix, x=state_names, y=state_names, text_auto=".1%", color_continuous_scale=[[0, "#161B22"], [1, COLORS["highlight"]]], zmin=0, zmax=1)
        fig_trans.update_layout(**PLOTLY_BASE_LAYOUT)
        st.plotly_chart(fig_trans, use_container_width=True)
        
    with col_persist:
        persist_prob = trans_matrix[current_regime, current_regime]
        st.metric("P(permanecer 1 semana)",  f"{persist_prob:.1%}")
        st.metric("P(permanecer 4 semanas)", f"{persist_prob ** 4:.1%}")
        st.metric("P(permanecer 8 semanas)", f"{persist_prob ** 8:.1%}")
        st.caption("Calculado como Pᴺ onde P é a probabilidade de auto-transição e N é o horizonte em semanas.")