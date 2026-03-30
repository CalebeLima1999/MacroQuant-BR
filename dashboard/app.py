import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# --- RESOLUÇÃO DE PATH (ROOT DIR) ---
# Adiciona a raiz do projeto (MacroQuant-BR) ao PYTHONPATH
# Isso garante que a pasta 'src' seja visível globalmente
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Configuração MUST BE a primeira chamada do Streamlit
st.set_page_config(
    page_title="MacroQuant-BR | Institutional Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IMPORTS DE ROTEAMENTO ---
# Agora o Python consegue aceder ao src.viz.theme dentro destas páginas
from pages.p1_macro_monitor import render_page as render_macro
from pages.p2_correlacoes import render_page as render_correlacoes
from pages.p3_var_analysis import render_page as render_var
from pages.p4_portfolio import render_page as render_portfolio
from pages.p5_regimes import render_page as render_regimes

PAGES = {
    "1. Macro Monitor": render_macro,
    "2. Correlações Dinâmicas": render_correlacoes,
    "3. Análise VAR": render_var,
    "4. Portfolio Optimizer": render_portfolio,
    "5. Regime Detector": render_regimes,
}

st.markdown("""
<style>
    /* Remove padding excessivo do container principal */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
    
    /* Cards de KPI */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1rem 1rem 1rem 1.25rem;
        transition: border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(255,255,255,0.18);
    }
    
    /* Números dos KPIs em monospace */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 1.4rem;
    }
    
    /* Delta dos KPIs menor e discreto */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-size: 0.75rem;
    }
    
    /* Sidebar mais limpa */
    [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    
    /* Títulos de seção sem emoji e com peso correto */
    h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.02em; }
    
    /* Expander com borda sutil */
    [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; }
    
    /* Remove borda inferior do hr padrão do streamlit */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }
    
    /* Tabela de dados com fonte monospace */
    [data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

if 'global_lookback' not in st.session_state:
    st.session_state.global_lookback = '1Y'

with st.sidebar:
    st.markdown("### MacroQuant-BR")
    st.caption("v1.0.0 · Production")
    st.write("")
    
    page = st.radio("Navegação", list(PAGES.keys()), label_visibility="collapsed")
    
    st.write("")
    st.session_state.global_lookback = st.selectbox(
        "⏳ Janela Histórica",
        options=['3M', '6M', '1Y', '3Y', '5Y', 'MAX'],
        index=2
    )
    
    st.write("")
    try:
        mtime = os.path.getmtime("data/processed/dataset_final.parquet")
        last_update = datetime.fromtimestamp(mtime)
        hours_ago = (datetime.now() - last_update).total_seconds() / 3600
        
        if hours_ago < 24:
            status = "🟢 Online"
        elif hours_ago < 72:
            status = "🟡 Desatualizado"
        else:
            status = "🔴 Offline"
            
        st.caption(f"Pipeline: {status} · {last_update.strftime('%d/%m %H:%M')}")
    except Exception:
        st.caption("Pipeline: ⚪ Status desconhecido")

if page in PAGES:
    PAGES[page]()