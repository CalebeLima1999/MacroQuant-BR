import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
from src.viz.theme import COLORS, ASSET_COLOR_MAP, PLOTLY_BASE_LAYOUT, PLOTLY_AXES_STYLE

REQUIRED_COLUMNS = ['selic_diaria', 'BTC-USD', 'BRL=X', 'ipca_mensal']

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_parquet("data/processed/dataset_final.parquet")

def validate_columns(df: pd.DataFrame, required: list) -> list:
    return [col for col in required if col not in df.columns]

def filter_by_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if lookback == 'MAX': return df
    end_date = df.index[-1]
    val = int(lookback[:-1])
    unit = lookback[-1]
    if unit == 'M': start_date = end_date - relativedelta(months=val)
    elif unit == 'Y': start_date = end_date - relativedelta(years=val)
    return df[df.index >= start_date]

def safe_delta(series: pd.Series, periods: int = 4):
    if len(series) < periods + 1: return None
    return series.iloc[-1] - series.iloc[-(periods + 1)]

def render_page():
    st.header("Macro Monitor")
    st.markdown("Visão geral do ambiente macroeconômico brasileiro e sua relação com ativos de risco.")
    
    try:
        with st.spinner("Carregando dados do pipeline..."):
            df_full = load_data()
            
        missing = validate_columns(df_full, REQUIRED_COLUMNS)
        if missing:
            st.error(f"Colunas ausentes no dataset: `{missing}`. Verifique o pipeline de dados.")
            st.stop()

        df = filter_by_lookback(df_full, st.session_state.global_lookback)
        if df.empty:
            st.warning("⚠️ Não há dados disponíveis para a janela selecionada.")
            return

        last_date = df.index[-1].strftime('%d/%m/%Y')
        st.caption(f"Última atualização dos dados: **{last_date}**")
        
        c1, c2, c3, c4 = st.columns(4)
        
        current_selic = df['selic_diaria'].iloc[-1]
        delta_selic = safe_delta(df['selic_diaria'])
        delta_selic_bps = (delta_selic * 100) if delta_selic is not None else None
        c1.metric("Taxa Selic", f"{current_selic:.2f}%", f"{delta_selic_bps:+.0f} bps (1M)" if delta_selic_bps is not None else "N/D")
        
        current_ipca = df['ipca_mensal'].iloc[-1]
        delta_ipca = safe_delta(df['ipca_mensal'])
        c2.metric("IPCA Mensal", f"{current_ipca:.2f}%", f"{delta_ipca:+.2f} p.p. (1M)" if delta_ipca is not None else "N/D", delta_color="inverse")
        
        current_brl = df['BRL=X'].iloc[-1]
        delta_brl = (current_brl / df['BRL=X'].iloc[-5]) - 1 if len(df) >= 5 else None
        c3.metric("Câmbio (USD/BRL)", f"R$ {current_brl:.2f}", f"{delta_brl:+.2%}" if delta_brl is not None else "N/D", delta_color="inverse")
        
        current_btc = df['BTC-USD'].iloc[-1]
        delta_btc = (current_btc / df['BTC-USD'].iloc[-5]) - 1 if len(df) >= 5 else None
        c4.metric("Bitcoin (USD)", f"$ {current_btc:,.0f}", f"{delta_btc:+.2%}" if delta_btc is not None else "N/D")
        
        st.write("")
        st.subheader("Dinâmica de Preços vs. Taxa de Juros")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['BTC-USD'], name="Bitcoin", line=dict(color=COLORS['btc'], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['selic_diaria'], name="Selic", line=dict(color=COLORS['selic'], width=1.5, shape='hv')), secondary_y=True)
        
        fig.update_layout(**PLOTLY_BASE_LAYOUT)
        fig.update_yaxes(title_text="Preço BTC ($)", secondary_y=False, **PLOTLY_AXES_STYLE)
        fig.update_yaxes(title_text="Selic (%)", secondary_y=True, showgrid=False, zeroline=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver Dados Brutos e Exportar"):
            st.dataframe(df[['BTC-USD', 'selic_diaria', 'BRL=X', 'ipca_mensal']].sort_index(ascending=False))
            col_csv, col_xlsx, _ = st.columns([2, 2, 6])
            
            csv_data = df.to_csv().encode('utf-8')
            col_csv.download_button("📥 Exportar CSV", data=csv_data, file_name=f"macroquant_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')
            
            xlsx_buffer = io.BytesIO()
            df.to_excel(xlsx_buffer, index=True, sheet_name="MacroQuant-BR")
            col_xlsx.download_button("📊 Exportar Excel", data=xlsx_buffer.getvalue(), file_name=f"macroquant_{datetime.now().strftime('%Y%m%d')}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        st.error(f"**Falha Crítica do Sistema:** {e}")