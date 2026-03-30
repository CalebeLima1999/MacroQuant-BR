"""
Sistema de design centralizado do MacroQuant-BR.
Todas as constantes visuais devem ser importadas daqui.
Nunca hardcodar cores, fontes ou espaçamentos nas páginas.
"""

# Paleta principal — cores com função semântica definida
COLORS = {
    "btc":       "#F7931A",   # Bitcoin — laranja canônico
    "eth":       "#627EEA",   # Ethereum — roxo canônico  
    "ibov":      "#00C805",   # B3/IBOV — verde mercado
    "selic":     "#00A4A6",   # Política monetária — teal
    "usd":       "#85B7EB",   # Câmbio/dólar — azul claro
    "positive":  "#3DBA6F",   # Retorno positivo
    "negative":  "#F0505A",   # Retorno negativo / risco
    "neutral":   "#8B949E",   # Informação secundária
    "highlight": "#3B82F6",   # Destaque interativo (azul)
    "warning":   "#D29922",   # Alerta amarelo
    "surface":   "#161B22",   # Fundo de card
    "border":    "rgba(255,255,255,0.08)",
    "grid":      "rgba(255,255,255,0.06)",
    "band":      "rgba(59,130,246,0.12)",  # IC de modelos
}

# Mapa de ativos para uso em gráficos com múltiplos ativos
ASSET_COLOR_MAP = {
    "BTC-USD":  COLORS["btc"],
    "ETH-USD":  COLORS["eth"],
    "^BVSP":    COLORS["ibov"],
    "selic_diaria": COLORS["selic"],
    "BRL=X":    COLORS["usd"],
}

# Layout Plotly padrão — aplicar em todos os gráficos
PLOTLY_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(family="JetBrains Mono, Roboto Mono, monospace", size=12),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
)

PLOTLY_AXES_STYLE = dict(
    showgrid=True,
    gridcolor=COLORS["grid"],
    zeroline=False,
    linecolor=COLORS["border"],
)