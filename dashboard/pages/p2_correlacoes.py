import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pages.p1_macro_monitor import load_data, filter_by_lookback
from src.viz.theme import COLORS, PLOTLY_BASE_LAYOUT, PLOTLY_AXES_STYLE

def render_page():
    st.header("Correlações Dinâmicas")
    st.markdown("Análise de co-movimento entre ativos de risco e a macroeconomia brasileira.")
    
    try:
        with st.spinner("Processando matriz de covariância..."):
            df_full = load_data()
        
        df = filter_by_lookback(df_full, st.session_state.global_lookback)
        if df.empty:
            st.warning("⚠️ Não há dados disponíveis para a janela selecionada.")
            return

        cols_to_corr = ['ret_btc-usd', 'ret_eth-usd', 'ret_bvsp', 'ret_brl=x', 'selic_real_ex_ante', 'm2']
        available_cols = [c for c in cols_to_corr if c in df.columns]
        df_corr_source = df[available_cols].dropna()
        
        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.subheader("Matriz Linear (Pearson)")
            corr_matrix = df_corr_source.corr()
            fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_heatmap.update_layout(**PLOTLY_BASE_LAYOUT)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        with c2:
            st.subheader("Correlação Rolante")
            col_a, col_b, col_w = st.columns([2, 2, 1])
            asset_a = col_a.selectbox("Ativo A", available_cols, index=0)
            asset_b = col_b.selectbox("Ativo B", available_cols, index=2)
            window  = col_w.number_input("Janela", min_value=4, max_value=52, value=12, step=4)
            
            min_periods_needed = window + 1
            if len(df_corr_source) < min_periods_needed:
                st.info(f"Dados insuficientes para janela de {window} semanas.")
            else:
                rolling_corr = df_corr_source[asset_a].rolling(window=window).corr(df_corr_source[asset_b])
                
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=rolling_corr.index, y=rolling_corr,
                    fill='tozeroy', fillcolor="rgba(0,164,166,0.15)",
                    line=dict(color=COLORS["selic"], width=2),
                    name=f"Corr({asset_a}, {asset_b})"
                ))
                fig_roll.update_layout(**PLOTLY_BASE_LAYOUT)
                fig_roll.update_yaxes(title="Correlação", range=[-1, 1], **PLOTLY_AXES_STYLE)
                st.plotly_chart(fig_roll, use_container_width=True)

    except Exception as e:
        st.error(f"**Falha ao renderizar análises:** {e}")