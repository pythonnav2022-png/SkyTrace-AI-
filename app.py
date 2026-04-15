"""
app.py — SkyTrace AI Carbon Intelligence
Streamlit Dashboard — Green Industry Hackathon
Lancement : streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os
from datetime import datetime

st.set_page_config(
    page_title="SkyTrace AI — Carbon Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:#0a0f0a;color:#e8f5e8;}
.main{background:#0a0f0a;}
[data-testid="stSidebar"]{background:#080e08;border-right:1px solid #1a3a1a;}
.kpi-box{background:linear-gradient(135deg,#0f2010,#1a3a1f);border:1px solid #2d5a30;
         border-radius:10px;padding:1rem 1.4rem;text-align:center;margin-bottom:.5rem;}
.kpi-label{font-size:.72rem;color:#81c784;text-transform:uppercase;letter-spacing:1px;}
.kpi-value{font-size:1.9rem;font-weight:700;color:#00e676;font-family:'JetBrains Mono',monospace;}
.kpi-unit{font-size:.72rem;color:#a5d6a7;margin-top:.1rem;}
.alerte-rouge{background:#3a0a0a;border-left:4px solid #ef5350;border-radius:6px;padding:.8rem 1rem;margin:.5rem 0;}
.alerte-verte{background:#0a2010;border-left:4px solid #00e676;border-radius:6px;padding:.8rem 1rem;margin:.5rem 0;}
.stButton>button{background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#e8f5e8;
                 border:1px solid #43a047;border-radius:8px;font-weight:600;padding:.6rem 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1f0d,#1a3a1a,#0f2a1a);border:1px solid #2d5a2d;
     border-radius:12px;padding:1.8rem 2rem;margin-bottom:1.5rem;">
  <h1 style="font-size:2.2rem;font-weight:700;margin:0;
    background:linear-gradient(90deg,#00e676,#69f0ae,#00bcd4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🏭 SkyTrace AI — Carbon Intelligence Platform
  </h1>
  <p style="color:#81c784;margin:.4rem 0 0;font-size:.9rem;">
    Surveillance · Prédiction · Optimisation des émissions CO₂ industrielles · Maroc 2026
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Données ──────────────────────────────────────────────────────────────────
@st.cache_data
def charger_donnees():
    np.random.seed(42)
    zones_coords = {
        "Jorf Lasfar (OCP)":         {"lon":-8.638,"lat":33.105,"secteur":"Chimie/Phosphates"},
        "El Jadida":                  {"lon":-8.500,"lat":33.231,"secteur":"Port/Chimie"},
        "Safi Industrial":            {"lon":-9.237,"lat":32.299,"secteur":"Phosphates"},
        "Mohammedia Port":            {"lon":-7.383,"lat":33.686,"secteur":"Raffinerie"},
        "Casablanca Industrial Zone": {"lon":-7.603,"lat":33.589,"secteur":"Industrie mixte"},
        "Nouaceur / Midparc":         {"lon":-7.589,"lat":33.367,"secteur":"Aeronautique"},
        "Kenitra Industrial Zone":    {"lon":-6.578,"lat":34.261,"secteur":"Automobile"},
        "Tangier Med":                {"lon":-5.503,"lat":35.893,"secteur":"Port/Logistique"},
        "Tangier Automotive City":    {"lon":-5.912,"lat":35.726,"secteur":"Automobile"},
        "Khouribga":                  {"lon":-6.906,"lat":32.881,"secteur":"Phosphates"},
        "Settat Industrial Zone":     {"lon":-7.620,"lat":33.000,"secteur":"Cimenterie"},
        "Nador West Med":             {"lon":-2.928,"lat":35.169,"secteur":"Siderurgie"},
        "Fes Industrial Zone":        {"lon":-5.003,"lat":34.033,"secteur":"Tannerie"},
        "Safi Industrial":            {"lon":-9.237,"lat":32.299,"secteur":"Phosphates"},
        "Laayoune Industrial Zone":   {"lon":-13.203,"lat":27.153,"secteur":"Phosphates"},
        "Dakhla Industrial Zone":     {"lon":-15.957,"lat":23.684,"secteur":"Peche"},
    }
    PARAMS = {
        "Chimie/Phosphates": {"conso":2800,"oee":0.74,"scope1_base":18,"scope2_base":55},
        "Raffinerie":        {"conso":3500,"oee":0.78,"scope1_base":24,"scope2_base":68},
        "Phosphates":        {"conso":2200,"oee":0.71,"scope1_base":15,"scope2_base":44},
        "Industrie mixte":   {"conso":1800,"oee":0.68,"scope1_base":10,"scope2_base":36},
        "Automobile":        {"conso":2100,"oee":0.82,"scope1_base":8,"scope2_base":42},
        "Port/Logistique":   {"conso":1200,"oee":0.70,"scope1_base":6,"scope2_base":24},
        "Aeronautique":      {"conso":900, "oee":0.88,"scope1_base":4,"scope2_base":18},
        "Cimenterie":        {"conso":2500,"oee":0.73,"scope1_base":20,"scope2_base":50},
        "Siderurgie":        {"conso":3200,"oee":0.75,"scope1_base":28,"scope2_base":64},
        "Tannerie":          {"conso":500, "oee":0.66,"scope1_base":5,"scope2_base":10},
        "Phosphates":        {"conso":2200,"oee":0.71,"scope1_base":15,"scope2_base":44},
        "Peche":             {"conso":200, "oee":0.60,"scope1_base":2,"scope2_base":4},
    }
    records = []
    for zone, info in zones_coords.items():
        sec = info["secteur"]
        p   = PARAMS.get(sec, PARAMS["Industrie mixte"])
        n   = np.random.uniform(0.88, 1.12)
        s1  = p["scope1_base"] * n
        s2  = p["scope2_base"] * n
        records.append({
            "zone": zone, "lon": info["lon"], "lat": info["lat"], "secteur": sec,
            "conso_kwh_j": p["conso"]*1000*n, "oee": p["oee"]*np.random.uniform(0.95,1.05),
            "scope1_tco2_j": s1, "scope2_tco2_j": s2, "total_tco2_j": s1+s2,
            "intensite_gco2_kwh": (s1+s2)*1e6/(p["conso"]*1000*n),
            "part_renouv": np.random.uniform(0.08, 0.30),
        })
    df = pd.DataFrame(records).drop_duplicates("zone").set_index("zone")

    # Séries temporelles 30 jours pour la zone sélectionnée
    return df

df_pivot = charger_donnees()

def gen_ts(zone, nb_jours=30):
    base = df_pivot.loc[zone]
    dates = pd.date_range(f"2026-03-01", periods=nb_jours*24, freq="h")
    records = []
    for i, dt in enumerate(dates):
        h, dow = dt.hour, dt.dayofweek
        fh = 0.55+0.45*np.sin(np.pi*max(0,h-5)/13) if 5<=h<=18 else 0.55
        fw = 0.62 if dow>=5 else 1.0
        n  = np.random.normal(1.0,0.06)
        anom = 2.5 if (dt.day in [8,22] and h in [10,11,14]) else 1.0
        total = base["total_tco2_j"]/24 * fh * fw * n * anom
        records.append({"datetime":dt,"total_tco2":max(0,total),"conso_kwh":base["conso_kwh_j"]/24*fh*fw*n})
    return pd.DataFrame(records)

def simuler_action(zone, action):
    b = df_pivot.loc[zone]
    ACTIONS = {
        "Decalage hors-pointe": {"pct":0.22,"co2_pct":0.19,"eco_kwh":0,"desc":"Déplacer 22% production vers 22h-6h (FE -140 gCO2/kWh)"},
        "Amelioration OEE":     {"pct":0.08,"co2_pct":0.10,"eco_kwh":0.08,"desc":"OEE+8% → même prod, -10% énergie"},
        "Passage ENR (30%)":    {"pct":0.15,"co2_pct":0.18,"eco_kwh":0.06,"desc":"Remplacer 15% réseau par PV/éolien"},
        "Arret veille IoT":     {"pct":0.08,"co2_pct":0.08,"eco_kwh":0.08,"desc":"Couper 8% stand-by avec capteurs IoT"},
    }
    a = ACTIONS.get(action, ACTIONS["Decalage hors-pointe"])
    co2_evite_an = b["total_tco2_j"]*365*a["co2_pct"]
    eco_mad_an   = b["conso_kwh_j"]*a["eco_kwh"]*1.05*365
    return co2_evite_an, eco_mad_an

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    st.markdown("---")
    zone_sel = st.selectbox("🏭 Zone industrielle", options=sorted(df_pivot.index.tolist()), index=0)
    gaz_sel  = st.selectbox("🔬 Gaz de référence", ["CO₂ total (Scope 1+2)","Scope 1 (direct)","Scope 2 (électricité)"])
    horizon  = st.slider("📅 Horizon prévision (jours)", 1, 14, 7)
    action   = st.selectbox("⚡ Action verte à simuler",
                            ["Decalage hors-pointe","Amelioration OEE","Passage ENR (30%)","Arret veille IoT"])
    simuler  = st.button("🌿 Simuler l'action verte", use_container_width=True)
    st.markdown("---")

    # Export PDF
    if st.button("📄 Exporter rapport PDF", use_container_width=True):
        st.info("Lance le notebook Module 9 pour générer le PDF.")

    st.markdown("---")
    st.markdown("**Stack technique**")
    st.markdown("🛰️ ESA Sentinel-5P · GEE  \n🤖 XGBoost · LightGBM · Prophet  \n🗺️ Plotly · Folium · Streamlit")

# ─── KPIs ─────────────────────────────────────────────────────────────────────
b = df_pivot.loc[zone_sel]
co2_evite_an, eco_mad_an = simuler_action(zone_sel, action)
pct_red = co2_evite_an/(b["total_tco2_j"]*365)*100

col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">CO₂ Scope 1+2</div><div class="kpi-value">{b["total_tco2_j"]:.1f}</div><div class="kpi-unit">tCO₂/jour</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">OEE actuel</div><div class="kpi-value">{b["oee"]*100:.0f}%</div><div class="kpi-unit">{"⚠️ < 75%" if b["oee"]<0.75 else "✅ OK"}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">Intensité carbone</div><div class="kpi-value">{b["intensite_gco2_kwh"]:.0f}</div><div class="kpi-unit">gCO₂/kWh</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">Part ENR</div><div class="kpi-value">{b["part_renouv"]*100:.0f}%</div><div class="kpi-unit">énergie verte</div></div>', unsafe_allow_html=True)

# ─── CARTE + SÉRIE TEMPORELLE ─────────────────────────────────────────────────
col_carte, col_ts = st.columns([1.2, 1])

with col_carte:
    st.markdown("#### 🗺️ Carte des émissions CO₂ — Maroc Industriel")
    fig_map = go.Figure(go.Scattermapbox(
        lat=df_pivot["lat"], lon=df_pivot["lon"], mode="markers",
        marker=dict(
            size=df_pivot["total_tco2_j"]/3+12,
            color=df_pivot["intensite_gco2_kwh"],
            colorscale="RdYlGn_r", cmin=350, cmax=750,
            colorbar=dict(title="gCO₂/kWh"), opacity=0.90
        ),
        text=df_pivot.reset_index().apply(lambda r:
            f"<b>{r['zone']}</b><br>Secteur: {r['secteur']}<br>"
            f"CO₂: {r['total_tco2_j']:.1f} tCO₂/j<br>OEE: {r['oee']*100:.0f}%<br>"
            f"Intensité: {r['intensite_gco2_kwh']:.0f} gCO₂/kWh", axis=1).values,
        hoverinfo="text",
    ))
    # Highlight zone sélectionnée
    fig_map.add_trace(go.Scattermapbox(
        lat=[b["lat"]], lon=[b["lon"]], mode="markers",
        marker=dict(size=25, color="cyan", opacity=0.9, symbol="circle"),
        name=zone_sel, hoverinfo="skip",
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=31.5,lon=-7.0), zoom=4.0),
        height=380, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#0a0f0a", font=dict(color="#e8f5e8"), showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_ts:
    st.markdown(f"#### 📈 Évolution CO₂ — {zone_sel.split()[0]} (30 jours)")
    df_ts = gen_ts(zone_sel)
    df_daily = df_ts.set_index("datetime")["total_tco2"].resample("D").sum()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily.values,
        fill="tozeroy", fillcolor="rgba(0,230,118,0.1)",
        line=dict(color="#00e676", width=2), name="CO₂ réel"
    ))
    # Prévision simple (Prophet simulé)
    last = df_daily.index[-1]; last_v = df_daily.values[-1]
    fut_dates = pd.date_range(last, periods=horizon+1)[1:]
    fut_vals  = [last_v*(1+np.random.normal(0,0.04)) for _ in range(horizon)]
    fig_ts.add_trace(go.Scatter(
        x=fut_dates, y=fut_vals, line=dict(color="#00bcd4", width=2, dash="dash"),
        name=f"Prévision {horizon}j"
    ))
    fig_ts.add_vrect(x0=last, x1=fut_dates[-1], fillcolor="rgba(0,188,212,0.08)", line_width=0)
    fig_ts.update_layout(
        paper_bgcolor="#0a0f0a", plot_bgcolor="#0a0f0a",
        height=380, margin=dict(l=10,r=10,t=20,b=10),
        font=dict(color="#e8f5e8"), legend=dict(font=dict(color="#e8f5e8")),
        xaxis=dict(showgrid=False,color="#81c784"),
        yaxis=dict(showgrid=True,gridcolor="#1a3a1a",color="#81c784",title="tCO₂/jour"),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ─── SIMULATION ACTION VERTE ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### ⚡ Simulation d'action verte")

if simuler:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">CO₂ évité par an</div>
          <div class="kpi-value" style="color:#69f0ae;">{co2_evite_an:,.0f}</div>
          <div class="kpi-unit">tCO₂/an</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">Économies financières</div>
          <div class="kpi-value" style="color:#69f0ae;">{eco_mad_an/1000:,.0f}k</div>
          <div class="kpi-unit">MAD/an</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">Réduction CO₂</div>
          <div class="kpi-value" style="color:#69f0ae;">-{pct_red:.1f}%</div>
          <div class="kpi-unit">des émissions actuelles</div>
        </div>""", unsafe_allow_html=True)

    # Graphique avant/après
    mois = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    base_m = b["total_tco2_j"]*365/12
    avant  = [base_m]*12
    apres  = [base_m*(1-pct_red/100*min(1,i/8)) for i in range(12)]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=mois,y=avant,name="Avant",line=dict(color="red",width=2,dash="dash")))
    fig_comp.add_trace(go.Scatter(x=mois,y=apres,name="Après",line=dict(color="#00e676",width=2.5),
                                  fill="tonexty",fillcolor="rgba(0,230,118,0.12)"))
    fig_comp.add_annotation(x=mois[-1],y=apres[-1],text=f"-{pct_red:.0f}% CO₂",
                            font=dict(size=14,color="#00e676"),showarrow=False,xshift=-30)
    fig_comp.update_layout(
        title=f"Trajectoire de réduction CO₂ — {action}",
        paper_bgcolor="#0a0f0a",plot_bgcolor="#0a0f0a",
        height=280,margin=dict(l=10,r=10,t=40,b=10),
        font=dict(color="#e8f5e8"),legend=dict(font=dict(color="#e8f5e8")),
        xaxis=dict(showgrid=False,color="#81c784"),
        yaxis=dict(showgrid=True,gridcolor="#1a3a1a",color="#81c784",title="tCO₂/mois"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    if pct_red > 15:
        st.markdown(f'<div class="alerte-verte">🟢 <b>Action à fort impact</b> — Réduction de {pct_red:.1f}% des émissions · ROI estimé &lt; 12 mois · Conforme NDC Maroc 2030</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alerte-verte">✅ <b>Action complémentaire</b> — {pct_red:.1f}% de réduction · À combiner avec d\'autres leviers pour atteindre -30%</div>', unsafe_allow_html=True)
else:
    st.info("👆 Sélectionne une action verte dans le panneau gauche et clique sur **Simuler**")

# ─── CLASSEMENT ZONES ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🏆 Classement des zones par émissions CO₂")
df_rank = df_pivot.sort_values("total_tco2_j", ascending=False).head(10).reset_index()
df_rank["Secteur"] = df_rank["secteur"]
df_rank["tCO₂/j"] = df_rank["total_tco2_j"].round(1)
df_rank["OEE"] = (df_rank["oee"]*100).round(0).astype(int).astype(str) + "%"
df_rank["Intensité gCO₂/kWh"] = df_rank["intensite_gco2_kwh"].round(0).astype(int)
df_rank["🚦"] = df_rank["total_tco2_j"].apply(lambda x: "🔴" if x>50 else "🟠" if x>25 else "🟢")
st.dataframe(df_rank[["zone","🚦","Secteur","tCO₂/j","OEE","Intensité gCO₂/kWh"]].rename(columns={"zone":"Zone"}),
             use_container_width=True, height=320)

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#4a7a4a;font-size:.8rem;">SkyTrace AI · Green Industry Hackathon · ESA Copernicus Sentinel-5P · XGBoost · LightGBM · Prophet</div>', unsafe_allow_html=True)
