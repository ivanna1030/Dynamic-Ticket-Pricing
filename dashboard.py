import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from config import cargar_parametros
from simulador import generar_escenarios, simular
from politicas import (
    politica_base,
    politica_pfa,
    politica_dla,
    afinar_base,
    afinar_pfa,
    afinar_dla,
    proyectar_a_malla,
)

# ==============================================================================
# 1. PAGE CONFIG AND VISUAL THEME (CLEAN BLUE TICKETING STYLE)
# ==============================================================================
st.set_page_config(
    page_title="BoxOffice Pricing | Panel del Promotor",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens shared by CSS and Plotly figures
NAVY = "#0B1B3A"
BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
SKY = "#38BDF8"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"
SOLD = "#2563EB"
AVAILABLE = "#E2E8F0"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #F5F8FC; color: #0B1B3A; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: #F5F8FC; }
[data-testid="stAppDeployButton"] { display: none; }
.block-container { padding-top: 4.6rem; max-width: 1350px; }

/* ---------- Tabs as a website navigation menu ---------- */
.stTabs [role="tablist"] {
    gap: 6px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 6px; margin: 0 0 8px 0; width: 100%; display: flex; box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    border-bottom: 1px solid #E2E8F0 !important;
    position: sticky; top: 64px; z-index: 100;
}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before { display: none !important; }
.stTabs [role="tablist"] > :not([role="tab"]) { display: none !important; }
.stTabs [role="tab"] {
    flex: 1 1 0; justify-content: center; text-align: center;
    border-radius: 10px !important; padding: 10px 22px !important; height: auto;
    color: #475569; background: transparent; border: none !important; box-shadow: none !important;
}
.stTabs [role="tab"] p { font-size: 0.95rem; font-weight: 600; color: #475569; }
.stTabs [role="tab"]:hover { background: #EFF6FF; }
.stTabs [role="tab"]:hover p { color: #2563EB; }
.stTabs [role="tab"][aria-selected="true"] { background: #2563EB; }
.stTabs [role="tab"][aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator { display: none !important; }

/* ---------- Sidebar: the promoter's control panel ---------- */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
.side-header { font-weight: 900; font-size: 1.35rem; color: #0B1B3A; margin-bottom: 2px; }
.side-sub { color: #64748B; font-size: 0.82rem; margin-bottom: 12px; line-height: 1.4; }
section[data-testid="stSidebar"] details {
    border: 1px solid #E2E8F0 !important; border-radius: 12px !important; background: #F8FAFC;
}
section[data-testid="stSidebar"] summary p { font-weight: 700; color: #0B1B3A; }

/* ---------- Pills that separate "you control" vs "we recommend" ---------- */
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
}
.pill-control { background: #2563EB; color: #FFFFFF; }
.pill-result  { background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }

.section-title { display: flex; align-items: center; gap: 12px; margin: 18px 0 10px 0; }
.section-title h3 { font-weight: 800; font-size: 1.35rem; margin: 0; padding: 0; color: #0B1B3A; }
.section-desc { color: #64748B; font-size: 0.92rem; margin: -4px 0 14px 0; }

/* ---------- Hero banner ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 20px; padding: 32px 36px;
    background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #38BDF8 100%);
    box-shadow: 0 14px 40px rgba(37,99,235,0.25); margin-top: 6px;
}
.hero::after {
    content: ""; position: absolute; right: -90px; top: -90px; width: 340px; height: 340px;
    border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.25), transparent 65%);
}
.hero-tag {
    display: inline-block; background: rgba(255,255,255,0.18); color: #FFFFFF;
    padding: 5px 12px; border-radius: 999px; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em;
}
.hero h1 { font-weight: 900; font-size: 2.9rem; line-height: 1.05; color: #FFFFFF; margin: 12px 0 6px 0; padding: 0; }
.hero-sub { color: rgba(255,255,255,0.92); font-size: 1rem; max-width: 720px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.chip {
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.3);
    color: #FFFFFF; padding: 8px 14px; border-radius: 12px; font-size: 0.85rem;
}
.chip b { font-weight: 800; }

/* ---------- Guide boxes ---------- */
.guide { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 16px; }
.guide-box {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 14px 18px; font-size: 0.9rem; color: #334155;
}
.guide-box b { color: #0B1B3A; }
.guide-num { font-weight: 900; color: #2563EB; font-size: 1.1rem; margin-right: 6px; }

/* ---------- Bordered containers ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border-radius: 16px !important; border: 1px solid #BFDBFE !important;
}

/* ---------- Ticket card ---------- */
.ticket {
    display: flex; margin-top: 14px; border-radius: 18px; overflow: hidden;
    background: #FFFFFF; border: 1px solid #BFDBFE; box-shadow: 0 10px 30px rgba(37,99,235,0.12);
}
.ticket-main { flex: 2.2; padding: 22px 24px; }
.ticket-stub {
    flex: 1; padding: 22px 18px; position: relative;
    border-left: 2px dashed #93C5FD; background: #EFF6FF;
    display: flex; flex-direction: column; justify-content: center; gap: 12px;
}
.ticket-stub::before, .ticket-stub::after {
    content: ""; position: absolute; left: -13px; width: 24px; height: 24px;
    border-radius: 50%; background: #F5F8FC; border: 1px solid #BFDBFE;
}
.ticket-stub::before { top: -13px; }
.ticket-stub::after { bottom: -13px; }
.t-price { font-weight: 900; font-size: 3.6rem; line-height: 1; color: #1E3A8A; margin: 12px 0 4px 0; }
.t-price span { font-size: 1rem; color: #64748B; font-weight: 500; }
.t-policy { font-size: 0.88rem; color: #475569; }
.stub-val { font-weight: 900; font-size: 1.5rem; line-height: 1; color: #0B1B3A; }
.stub-lbl { font-size: 0.68rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

/* ---------- KPI cards ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 16px; }
.kpi {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04); border-top: 4px solid #2563EB;
}
.kpi-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #64748B; }
.kpi-value { font-weight: 900; font-size: 1.75rem; color: #0B1B3A; line-height: 1.15; margin-top: 6px; }
.kpi-delta { font-size: 0.82rem; margin-top: 6px; color: #64748B; }
.kpi-delta.up { color: #059669; font-weight: 600; }
.kpi-delta.down { color: #DC2626; font-weight: 600; }

/* ---------- Callout ---------- */
.callout {
    background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB;
    border-radius: 12px; padding: 16px 20px; color: #1E293B; font-size: 0.92rem;
}
.callout b { color: #1E3A8A; }

@media (max-width: 900px) {
    .kpi-grid, .guide { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str, kind: str, desc: str = "") -> None:
    """Render a section header tagged as a promoter control or a system result."""
    pill = (
        '<span class="pill pill-control">Tú ajustas</span>'
        if kind == "control"
        else '<span class="pill pill-result">Resultado</span>'
    )
    html = f'<div class="section-title"><h3>{title}</h3>{pill}</div>'
    if desc:
        html += f'<div class="section-desc">{desc}</div>'
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the clean blue theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=NAVY, size=12),
        title=dict(font=dict(size=15, color=NAVY)),
        height=height,
        margin=dict(l=16, r=16, t=55, b=16),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BLUE, font=dict(color=NAVY)),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor="#F1F5F9", zerolinecolor=BORDER, linecolor=BORDER)
    return fig


# ==============================================================================
# 2. SIDEBAR: PROMOTER INPUTS (CONNECTED TO config.py DEFAULTS)
# ==============================================================================
par_default = cargar_parametros()

st.sidebar.markdown(
    '<div class="side-header">Configura tu evento</div>'
    '<div class="side-sub">Todo lo de esta barra lo ajustas tú. '
    "Las pestañas de la derecha se actualizan solas.</div>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("1 · Tu evento", expanded=True):
    nombre_evento = st.text_input("Nombre del evento", value="Concierto en vivo")
    capacidad = st.number_input(
        "Capacidad del recinto (asientos)",
        min_value=100,
        max_value=50000,
        value=int(par_default.C),
        step=100,
    )
    ventana_venta = st.slider(
        "Días de venta antes del evento",
        min_value=15,
        max_value=120,
        value=int(par_default.T),
        step=5,
    )

with st.sidebar.expander("2 · Precios permitidos", expanded=True):
    precio_min, precio_max = st.slider(
        "Rango de precios ($)",
        min_value=10,
        max_value=500,
        value=(int(par_default.precio_min), int(par_default.precio_max)),
        step=10,
    )
    opciones_paso = [5, 10, 20, 25]
    idx_paso = opciones_paso.index(int(par_default.paso_precio)) if int(par_default.paso_precio) in opciones_paso else 1
    paso_precio = st.selectbox("Saltos de precio ($)", options=opciones_paso, index=idx_paso)

with st.sidebar.expander("3 · Tu público", expanded=True):
    wtp_mediana_default = int(round(np.exp(par_default.mu_V)))
    wtp_media = st.slider(
        "Disposición a pagar mediana ($)",
        min_value=40,
        max_value=250,
        value=wtp_mediana_default,
        step=5,
        help="Mediana de la distribución Lognormal de disposición a pagar (WTP).",
    )
    compradores_totales = st.slider(
        "Compradores esperados en la ventana",
        min_value=1000,
        max_value=20000,
        value=int(par_default.compradores_totales),
        step=500,
        help="Total de compradores potenciales esperados durante toda la ventana de venta.",
    )
    factor_escasez = st.slider(
        "Impulso por escasez (<10% aforo)",
        min_value=1.0,
        max_value=3.0,
        value=float(par_default.kappa),
        step=0.1,
        help="Cuánto se acelera la llegada de compradores cuando queda menos del 10% del aforo.",
    )

with st.sidebar.expander("4 · Memoria de precios", expanded=True):
    theta = st.slider(
        "Memoria del comprador (θ)",
        0.0,
        0.99,
        float(par_default.theta),
        0.05,
        help="Qué tanto recuerdan los compradores los precios anteriores.",
    )
    eta = st.slider(
        "Penalización por subida (η)",
        0.0,
        3.0,
        float(par_default.eta),
        0.1,
        help="Qué tanto castigan los compradores que el precio suba.",
    )
    gamma = st.slider(
        "Premio por descuento (γ)",
        0.0,
        2.0,
        float(par_default.gamma),
        0.1,
        help="Qué tanto premian los compradores que el precio baje.",
    )

with st.sidebar.expander("5 · Precisión de la simulación", expanded=True):
    n_escenarios = st.selectbox(
        "Escenarios simulados",
        [500, 1000, 2000],
        index=0,
        help="Más escenarios = resultados más precisos (mínimo 500 requeridos).",
    )
    reafinar_auto = st.checkbox(
        "Re-afinar políticas al cambiar parámetros",
        value=False,
        help="Si está desactivado, usa los parámetros afinados guardados en resultados/parametros_afinados.json (mucho más rápido).",
    )

# ==============================================================================
# 3. SIMULATION ENGINE (CONNECTED TO config.py, simulador.py & politicas.py)
# ==============================================================================
RUTA_AFINADOS = Path(__file__).parent / "resultados" / "parametros_afinados.json"


def _obtener_parametros_politicas(par, reafinar: bool):
    """Carga los parámetros afinados desde resultados/parametros_afinados.json o los afina en vivo."""
    if not reafinar and RUTA_AFINADOS.exists():
        with open(RUTA_AFINADOS, encoding="utf-8") as f:
            af = json.load(f)
        p_fijo = float(proyectar_a_malla(af["base"]["p_fijo"], par))
        p_base = float(proyectar_a_malla(af["pfa"]["p_base"], par))
        a_pfa = float(af["pfa"]["a"])
        beta_dla = float(af["dla"]["beta"])
        return p_fijo, (p_base, a_pfa), beta_dla

    # Si no existe el JSON o el usuario pidió re-afinar con los nuevos inputs:
    esc_afinar = generar_escenarios(200, par, semilla=1)
    p_fijo, _ = afinar_base(esc_afinar, par)
    theta_pfa, _ = afinar_pfa(esc_afinar, par)
    beta_dla, _ = afinar_dla(esc_afinar, par, betas=(0.9, 0.95, 1.0, 1.05, 1.1))
    return float(p_fijo), (float(theta_pfa[0]), float(theta_pfa[1])), float(beta_dla)


@st.cache_data(show_spinner="Ejecutando simulación en el modelo de Powell...")
def obtener_resultados_simulacion(
    cap, dias, p_min, p_max, paso, wtp_med, comp_tot, kappa_val, theta_val, eta_val, gamma_val, n_sim, reafinar
):
    par = cargar_parametros(
        C=int(cap),
        T=int(dias),
        precio_min=float(p_min),
        precio_max=float(p_max),
        paso_precio=float(paso),
        mu_V=float(np.log(wtp_med)),
        compradores_totales=float(comp_tot),
        kappa=float(kappa_val),
        theta=float(theta_val),
        eta=float(eta_val),
        gamma=float(gamma_val),
    )

    p_fijo, (p_base, a_pfa), beta_dla = _obtener_parametros_politicas(par, reafinar)

    politicas_dict = {
        "Precio Fijo (Baseline)": politica_base(p_fijo),
        "Regla Dinámica (PFA)": politica_pfa(p_base, a_pfa),
        "Lookahead (DLA)": politica_dla(beta_dla),
    }

    # Números aleatorios comunes (semilla 2 distinta a la semilla 1 de afinación)
    esc = generar_escenarios(int(n_sim), par, semilla=2)

    registros_escenarios = []
    registros_trayectoria = []
    dias_restantes = np.arange(par.T, 0, -1)  # d = T, T-1, ..., 1

    for nombre_pol, fn_pol in politicas_dict.items():
        res = simular(fn_pol, esc, par, guardar_trayectorias=True)

        # 1) Resultados por escenario (para boxplots, histogramas, media e IC 95%)
        ingresos = res["ingreso"]
        agotados = res["agotado"].astype(int)
        for i in range(int(n_sim)):
            registros_escenarios.append({
                "Escenario": i + 1,
                "Politica": nombre_pol,
                "Ingresos": float(ingresos[i]),
                "Sold_Out": int(agotados[i]),
            })

        # 2) Trayectoria promedio día a día (para el gráfico de precios y mapa de asientos)
        precio_prom = res["precio"].mean(axis=0)
        ref_prom = res["referencia"].mean(axis=0)
        inv_prom = res["inventario"].mean(axis=0)

        for d, p, r, inv in zip(dias_restantes, precio_prom, ref_prom, inv_prom):
            registros_trayectoria.append({
                "Dias_Restantes": int(d),
                "Politica": nombre_pol,
                "Precio": round(float(p), 2),
                "Precio_Referencia": round(float(r), 2),
                "Asientos_Restantes": int(round(max(0.0, float(inv)))),
            })

        # Punto final d = 0 (día del evento) con el inventario restante tras el último día
        inv_final = float((par.C - res["vendidos"]).mean())
        registros_trayectoria.append({
            "Dias_Restantes": 0,
            "Politica": nombre_pol,
            "Precio": round(float(precio_prom[-1]), 2),
            "Precio_Referencia": round(float(ref_prom[-1]), 2),
            "Asientos_Restantes": int(round(max(0.0, inv_final))),
        })

    info_params = {
        "p_fijo": p_fijo,
        "p_base": p_base,
        "a_pfa": a_pfa,
        "beta_dla": beta_dla,
        "forma_llegadas": par.forma_llegadas,
        "b_llegadas": par.b_llegadas,
    }
    return pd.DataFrame(registros_escenarios), pd.DataFrame(registros_trayectoria), info_params


@st.cache_data(show_spinner="Analizando sensibilidad al precio de referencia...")
def calcular_sensibilidad_eta(
    cap, dias, p_min, p_max, paso, wtp_med, comp_tot, kappa_val, theta_val, gamma_val, p_fijo, p_base, a_pfa, beta_dla
):
    """Evalúa las tres políticas reales variando eta en 6 niveles sobre los mismos escenarios."""
    etas = np.linspace(0.0, 2.5, 6)
    sens_data = []

    for e_val in etas:
        par_e = cargar_parametros(
            C=int(cap),
            T=int(dias),
            precio_min=float(p_min),
            precio_max=float(p_max),
            paso_precio=float(paso),
            mu_V=float(np.log(wtp_med)),
            compradores_totales=float(comp_tot),
            kappa=float(kappa_val),
            theta=float(theta_val),
            eta=float(e_val),
            gamma=float(gamma_val),
        )
        esc_e = generar_escenarios(300, par_e, semilla=99)
        pols = {
            "Precio Fijo (Baseline)": politica_base(p_fijo),
            "Regla Dinámica (PFA)": politica_pfa(p_base, a_pfa),
            "Lookahead (DLA)": politica_dla(beta_dla),
        }
        for nombre_pol, fn_pol in pols.items():
            res_e = simular(fn_pol, esc_e, par_e, guardar_trayectorias=False)
            sens_data.append({
                "Eta": float(e_val),
                "Politica": nombre_pol,
                "Ingreso_Medio": float(res_e["ingreso"].mean()),
            })

    return pd.DataFrame(sens_data)


df_escenarios, df_trayectorias, info_params = obtener_resultados_simulacion(
    capacidad,
    ventana_venta,
    precio_min,
    precio_max,
    paso_precio,
    wtp_media,
    compradores_totales,
    factor_escasez,
    theta,
    eta,
    gamma,
    n_escenarios,
    reafinar_auto,
)

# ==============================================================================
# 4. SUMMARY STATISTICS
# ==============================================================================
resumen_stats = []
for pol, grupo in df_escenarios.groupby("Politica"):
    media = grupo["Ingresos"].mean()
    std = grupo["Ingresos"].std(ddof=1)
    p5 = np.percentile(grupo["Ingresos"], 5)
    p95 = np.percentile(grupo["Ingresos"], 95)
    ci_low, ci_high = stats.t.interval(0.95, len(grupo) - 1, loc=media, scale=stats.sem(grupo["Ingresos"]))
    prob_sold_out = grupo["Sold_Out"].mean() * 100

    resumen_stats.append({
        "Política": pol,
        "Ingreso Esperado ($)": media,
        "Desv. Estándar ($)": std,
        "Percentil 5% ($)": p5,
        "Percentil 95% ($)": p95,
        "IC 95% Inferior ($)": ci_low,
        "IC 95% Superior ($)": ci_high,
        "Prob. Sold-Out (%)": prob_sold_out,
    })

df_stats = pd.DataFrame(resumen_stats).sort_values(by="Ingreso Esperado ($)", ascending=False)
mejor_politica = df_stats.iloc[0]
baseline_row = df_stats[df_stats["Política"].str.contains("Baseline")].iloc[0]
mejora_vs_base = (
    (mejor_politica["Ingreso Esperado ($)"] - baseline_row["Ingreso Esperado ($)"])
    / baseline_row["Ingreso Esperado ($)"]
) * 100

COLOR_MAP = {
    "Precio Fijo (Baseline)": "#94A3B8",
    "Regla Dinámica (PFA)": "#14B8A6",
    "Lookahead (DLA)": "#2563EB",
}

nombre_corto = mejor_politica["Política"].split(" (")[0]
df_ganadora = df_trayectorias[df_trayectorias["Politica"] == mejor_politica["Política"]].sort_values(
    "Dias_Restantes", ascending=False
)

# ==============================================================================
# 5. TOP NAVIGATION TABS
# ==============================================================================
tab_inicio, tab_taquilla, tab_precios, tab_estrategias, tab_sens = st.tabs([
    "Inicio",
    "Taquilla",
    "Precios",
    "Estrategias",
    "Sensibilidad",
])

# ==============================================================================
# 6. TAB: HOME (HERO + EXECUTIVE SUMMARY)
# ==============================================================================
with tab_inicio:
    st.markdown(
        f"""
<div class="hero">
  <span class="hero-tag">PANEL DEL PROMOTOR</span>
  <h1>{nombre_evento}</h1>
  <div class="hero-sub">Te decimos a qué precio vender cada día para maximizar tu taquilla,
  comparando tres estrategias en {n_escenarios:,} escenarios calibrados de demanda.</div>
  <div class="chips">
    <div class="chip">Aforo <b>{capacidad:,}</b> asientos</div>
    <div class="chip">Venta <b>{ventana_venta}</b> días</div>
    <div class="chip">Precios <b>${precio_min}–${precio_max}</b></div>
    <div class="chip">Estrategia ganadora <b>{nombre_corto}</b></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    ic_half = mejor_politica["IC 95% Superior ($)"] - mejor_politica["Ingreso Esperado ($)"]
    delta_so = mejor_politica["Prob. Sold-Out (%)"] - baseline_row["Prob. Sold-Out (%)"]
    cls_mejora = "up" if mejora_vs_base >= 0 else "down"
    cls_so = "up" if delta_so >= 0 else "down"

    section("Tu resultado esperado", "result", "Lo que obtendrías si aplicas la estrategia recomendada.")
    st.markdown(
        f"""
<div class="kpi-grid" style="margin-top:0">
  <div class="kpi">
    <div class="kpi-label">Estrategia recomendada</div>
    <div class="kpi-value">{nombre_corto}</div>
    <div class="kpi-delta {cls_mejora}">{mejora_vs_base:+.1f}% vs precio fijo</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Ingreso esperado</div>
    <div class="kpi-value">${mejor_politica['Ingreso Esperado ($)']:,.0f}</div>
    <div class="kpi-delta">IC 95%: ±${ic_half:,.0f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Rango probable (P5 a P95)</div>
    <div class="kpi-value">${mejor_politica['Percentil 5% ($)']:,.0f} – ${mejor_politica['Percentil 95% ($)']:,.0f}</div>
    <div class="kpi-delta">9 de cada 10 escenarios caen aquí</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Probabilidad de sold-out</div>
    <div class="kpi-value">{mejor_politica['Prob. Sold-Out (%)']:.1f}%</div>
    <div class="kpi-delta {cls_so}">{delta_so:+.1f} pts vs precio fijo</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    section("Cómo usar este panel", "control")
    st.markdown(
        """
<div class="guide" style="margin-top:0">
  <div class="guide-box"><span class="guide-num">1</span><b>Configura tu evento</b> en la barra izquierda:
    aforo, días de venta, precios y tipo de público.</div>
  <div class="guide-box"><span class="guide-num">2</span><b>Ve a Taquilla</b> y mueve el día para saber
    a qué precio vender hoy.</div>
  <div class="guide-box"><span class="guide-num">3</span><b>Revisa Precios, Estrategias y Sensibilidad</b>
    para entender por qué recomendamos esa estrategia.</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ==============================================================================
# 7. TAB: LIVE BOX OFFICE (DAY SELECTOR + TICKET + SEAT MAP)
# ==============================================================================
with tab_taquilla:
    section(
        "Taquilla en vivo",
        "control",
        "Mueve el día de venta y te decimos qué precio publicar y cómo va la ocupación promedio.",
    )

    col_ctrl, col_mapa = st.columns([1.1, 1.4], gap="large")

    with col_ctrl:
        with st.container(border=True):
            dia_actual = st.slider(
                "Días que faltan para el concierto",
                min_value=0,
                max_value=ventana_venta,
                value=int(ventana_venta // 2),
                help="Mueve este control para ver qué precio debes fijar en cada momento de la venta.",
            )

        fila_dia = df_ganadora[df_ganadora["Dias_Restantes"] == dia_actual].iloc[0]
        precio_hoy = fila_dia["Precio"]
        asientos_disp = fila_dia["Asientos_Restantes"]
        asientos_vendidos = max(0, capacidad - asientos_disp)
        pct_ocupacion = (asientos_vendidos / capacidad) * 100

        st.markdown(
            f"""
<div class="ticket">
  <div class="ticket-main">
    <span class="pill pill-result">Precio recomendado hoy</span>
    <div class="t-price">${precio_hoy:,.0f} <span>/ boleto</span></div>
    <div class="t-policy">Estrategia: <b>{mejor_politica['Política']}</b></div>
  </div>
  <div class="ticket-stub">
    <div><div class="stub-val">{dia_actual}</div><div class="stub-lbl">días para el show</div></div>
    <div><div class="stub-val">{asientos_disp:,}</div><div class="stub-lbl">lugares libres</div></div>
    <div><div class="stub-val">{pct_ocupacion:.0f}%</div><div class="stub-lbl">vendido</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_mapa:
        filas, cols = 15, 40
        total_puntos = filas * cols
        puntos_vendidos = int(round((pct_ocupacion / 100) * total_puntos))

        matriz_x, matriz_y, estado_asiento = [], [], []
        contador = 0
        for r in range(filas):
            for c in range(cols):
                matriz_x.append(c)
                matriz_y.append(filas - r)
                estado_asiento.append("Vendido" if contador < puntos_vendidos else "Disponible")
                contador += 1

        fig_estadio = go.Figure()
        fig_estadio.add_shape(
            type="rect",
            x0=8,
            y0=filas + 1.2,
            x1=31,
            y1=filas + 2.4,
            fillcolor=BLUE_DARK,
            line=dict(width=0),
            layer="above",
        )
        fig_estadio.add_annotation(
            x=19.5,
            y=filas + 1.8,
            text="<b>ESCENARIO</b>",
            showarrow=False,
            font=dict(color="white", size=11, family="Inter, sans-serif"),
        )

        for estado, color in [("Vendido", SOLD), ("Disponible", AVAILABLE)]:
            idx = [i for i, e in enumerate(estado_asiento) if e == estado]
            fig_estadio.add_trace(
                go.Scatter(
                    x=[matriz_x[i] for i in idx],
                    y=[matriz_y[i] for i in idx],
                    mode="markers",
                    name=f"Vendido ({pct_ocupacion:.0f}%)" if estado == "Vendido" else f"Disponible ({100 - pct_ocupacion:.0f}%)",
                    marker=dict(size=7, color=color, symbol="circle"),
                    hoverinfo="skip",
                )
            )

        fig_estadio.update_layout(
            title=dict(text=f"Ocupación del recinto · {asientos_vendidos:,} / {capacidad:,} boletos"),
            xaxis=dict(visible=False, range=[-1, cols]),
            yaxis=dict(visible=False, range=[0, filas + 3]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
        )
        style_fig(fig_estadio, height=340)
        with st.container(border=True):
            st.plotly_chart(fig_estadio, width="stretch", config={"displayModeBar": False})

# ==============================================================================
# 8. TAB: PRICE PATH
# ==============================================================================
with tab_precios:
    section(
        "Trayectoria de precios",
        "result",
        "Cómo evoluciona el precio promedio día a día con cada estrategia y cuántos boletos quedan por vender.",
    )
    c_izq, c_der = st.columns([3, 2])

    with c_izq:
        fig_precios = px.line(
            df_trayectorias,
            x="Dias_Restantes",
            y="Precio",
            color="Politica",
            color_discrete_map=COLOR_MAP,
            title="¿A qué precio vende cada estrategia?",
            labels={"Dias_Restantes": "Días para el evento", "Precio": "Precio promedio ($)", "Politica": "Estrategia"},
        )
        fig_precios.update_traces(line=dict(width=3))
        fig_precios.add_vline(
            x=dia_actual,
            line_dash="dash",
            line_color=BLUE,
            annotation_text="Día seleccionado en Taquilla",
            annotation_font_color=BLUE,
        )
        fig_precios.update_xaxes(autorange="reversed", showgrid=False)
        fig_precios.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, title=None),
        )
        st.plotly_chart(style_fig(fig_precios, 440), width="stretch")

    with c_der:
        fig_asientos = px.line(
            df_trayectorias,
            x="Dias_Restantes",
            y="Asientos_Restantes",
            color="Politica",
            color_discrete_map=COLOR_MAP,
            title="Boletos que quedan por vender",
            labels={"Dias_Restantes": "Días para el evento", "Asientos_Restantes": "Asientos disponibles"},
        )
        fig_asientos.update_traces(line=dict(width=3))
        fig_asientos.update_xaxes(autorange="reversed", showgrid=False)
        fig_asientos.update_layout(showlegend=False, hovermode="x unified")
        st.plotly_chart(style_fig(fig_asientos, 440), width="stretch")

# ==============================================================================
# 9. TAB: STRATEGY COMPARISON
# ==============================================================================
with tab_estrategias:
    section(
        f"Comparación de estrategias en {n_escenarios:,} escenarios",
        "result",
        "Cada caja resume los ingresos de todos los escenarios simulados con números aleatorios comunes.",
    )
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        fig_box = px.box(
            df_escenarios,
            x="Politica",
            y="Ingresos",
            color="Politica",
            color_discrete_map=COLOR_MAP,
            title="Ingresos posibles por estrategia",
            labels={"Ingresos": "Ingresos totales ($)", "Politica": ""},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_box, 420), width="stretch")

    with col_g2:
        fig_hist = px.histogram(
            df_escenarios,
            x="Ingresos",
            color="Politica",
            color_discrete_map=COLOR_MAP,
            barmode="overlay",
            opacity=0.6,
            title="¿Qué tan seguido se obtiene cada ingreso?",
            labels={"Ingresos": "Ingresos totales ($)", "Politica": "Estrategia"},
        )
        fig_hist.update_xaxes(showgrid=False)
        fig_hist.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, title=None))
        st.plotly_chart(style_fig(fig_hist, 420), width="stretch")

    st.markdown("#### Tabla comparativa")
    st.dataframe(
        df_stats.style.format({
            "Ingreso Esperado ($)": "${:,.2f}",
            "Desv. Estándar ($)": "${:,.2f}",
            "Percentil 5% ($)": "${:,.2f}",
            "Percentil 95% ($)": "${:,.2f}",
            "IC 95% Inferior ($)": "${:,.2f}",
            "IC 95% Superior ($)": "${:,.2f}",
            "Prob. Sold-Out (%)": "{:.1f}%",
        }).highlight_max(subset=["Ingreso Esperado ($)", "Prob. Sold-Out (%)"], color="#DBEAFE"),
        width="stretch",
        hide_index=True,
    )

# ==============================================================================
# 10. TAB: REFERENCE-PRICE SENSITIVITY
# ==============================================================================
with tab_sens:
    section(
        "¿Qué pasa si tu público odia que suban los precios?",
        "result",
        "Simulación real variando la penalización por subida de precio (η) desde 0 hasta 2.5 con tu memoria actual (θ).",
    )

    df_sens = calcular_sensibilidad_eta(
        capacidad,
        ventana_venta,
        precio_min,
        precio_max,
        paso_precio,
        wtp_media,
        compradores_totales,
        factor_escasez,
        theta,
        gamma,
        info_params["p_fijo"],
        info_params["p_base"],
        info_params["a_pfa"],
        info_params["beta_dla"],
    )

    fig_sens = px.line(
        df_sens,
        x="Eta",
        y="Ingreso_Medio",
        color="Politica",
        color_discrete_map=COLOR_MAP,
        markers=True,
        title="Ingreso esperado según la aversión a subidas de precio (η)",
        labels={
            "Eta": "Penalización por subida de precio (η)",
            "Ingreso_Medio": "Ingreso esperado ($)",
            "Politica": "Estrategia",
        },
    )
    fig_sens.update_traces(line=dict(width=3), marker=dict(size=9))
    fig_sens.add_vline(
        x=eta,
        line_dash="dot",
        line_color=BLUE,
        annotation_text="Tu valor actual de η",
        annotation_font_color=BLUE,
    )
    fig_sens.update_xaxes(showgrid=False)
    fig_sens.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, title=None))
    st.plotly_chart(style_fig(fig_sens, 460), width="stretch")

    st.markdown(
        '<div class="callout"><b>¿Cuándo cambiaría la recomendación?</b> Si la penalización por subida de precio (η) '
        "y la memoria (θ) son muy altas, los incrementos dinámicos destruyen demanda y conviene optar por una "
        "tarifa fija o promociones tempranas.</div>",
        unsafe_allow_html=True,
    )