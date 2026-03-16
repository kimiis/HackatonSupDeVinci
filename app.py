
# ==============================================================================
# HACKATHON #26 - app.py
# Dashboard Streamlit - Visualisation des données climatiques France
#
# Lancer avec : streamlit run app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================

st.set_page_config(
    page_title="Climat France — Hackathon #26",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer le rendu visuel
st.markdown("""
<style>
    /* Fond général */
    .main { background-color: #0e1117; }

    /* Cartes KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #aaa; margin: 0; }
    .kpi-delta { font-size: 0.9rem; margin-top: 4px; }

    /* Titres de section */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 1px solid #333;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Alertes */
    .alert-rouge {
        background: rgba(255,80,80,0.15);
        border-left: 4px solid #ff5050;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-orange {
        background: rgba(255,165,0,0.15);
        border-left: 4px solid orange;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-verte {
        background: rgba(0,200,100,0.15);
        border-left: 4px solid #00c864;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CHARGEMENT DES DONNÉES (mis en cache pour ne pas recharger à chaque interaction)
# ==============================================================================

@st.cache_data
def load_historique():
    df = pd.read_csv("data/clean/dataset_final.csv")
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    return df

@st.cache_data
def load_forecasts():
    noms = ["temperature", "co2", "niveau_mer", "jours_chauds",
            "jours_gel", "empreinte_carbone", "vendanges", "cout_catastrophes"]
    forecasts = {}
    for nom in noms:
        path = f"data/resultats/forecast_{nom}.csv"
        if os.path.exists(path):
            fc = pd.read_csv(path)
            fc["annee"] = pd.to_numeric(fc["annee"], errors="coerce")
            forecasts[nom] = fc
    return forecasts

@st.cache_data
def load_scenarios():
    path = "data/resultats/scenarios_temperature.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ports():
    df_ports = pd.read_excel("data/raw/id_port_fr.xlsx")
    df_mer   = pd.read_csv("data/clean/niveau_mer.csv")

    # On récupère la dernière valeur disponible pour chaque port
    # en associant le nom de la ville à sa colonne dans niveau_mer.csv
    derniere_annee = df_mer["annee"].max()
    derniere_ligne = df_mer[df_mer["annee"] == derniere_annee].iloc[0]

    resultats = []
    for _, row in df_ports.iterrows():
        col = f"niveau_mer_{row['Ville'].lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}"
        # Cherche la colonne correspondante (correspondance partielle si besoin)
        col_match = next((c for c in df_mer.columns if row["Ville"].lower()[:5] in c), None)
        niveau = derniere_ligne[col_match] if col_match and col_match in derniere_ligne.index else None
        resultats.append({
            "ville": row["Ville"],
            "lat":   row["Lat"],
            "lon":   row["Long"],
            "niveau_mm": niveau
        })
    return pd.DataFrame(resultats)

df       = load_historique()
forecasts = load_forecasts()
scenarios = load_scenarios()
df_ports  = load_ports()


# ==============================================================================
# SIDEBAR — Filtres globaux
# ==============================================================================

with st.sidebar:
    st.markdown("## 🌍 Climat France")
    st.markdown("*Hackathon #26 — Données officielles*")
    st.divider()

    st.markdown("### Filtres")

    annee_min = int(df["annee"].min())
    annee_max = int(df["annee"].max())
    periode = st.slider(
        "Période historique",
        min_value=annee_min,
        max_value=annee_max,
        value=(1950, annee_max),
        step=1
    )

    scenario_choisi = st.selectbox(
        "Scénario de projection",
        options=["optimiste", "intermediaire", "pessimiste"],
        index=1,
        format_func=lambda x: {
            "optimiste":     "🟢 Optimiste (+1.4°C)",
            "intermediaire": "🟠 Intermédiaire (+2.7°C)",
            "pessimiste":    "🔴 Pessimiste (+4.4°C)"
        }[x]
    )

    annee_projection = st.slider(
        "Horizon de projection",
        min_value=2026,
        max_value=2100,
        value=2050,
        step=5
    )

    st.divider()
    st.markdown("**Sources**")
    st.caption("Météo France · NOAA · CITEPA · PSMSL · INSEE · EM-DAT")


# ==============================================================================
# CALCUL DES KPIs
# ==============================================================================

df_filtre = df[(df["annee"] >= periode[0]) & (df["annee"] <= periode[1])]

# Température actuelle vs baseline 1900-1920
baseline   = df.loc[(df["annee"] >= 1900) & (df["annee"] <= 1920), "temp_moy_france"].mean()
temp_recente = df.loc[df["temp_moy_france"] > 9.5, "temp_moy_france"].iloc[-1]
annee_recente = int(df.loc[df["temp_moy_france"] > 9.5, "annee"].iloc[-1])
anomalie_temp = temp_recente - baseline

# CO2 actuel
co2_actuel = df["co2_ppm"].dropna().iloc[-1] if "co2_ppm" in df.columns else None

# Niveau de la mer (dernière valeur)
niveau_actuel = df["niveau_mer_mm"].dropna().iloc[-1] if "niveau_mer_mm" in df.columns else None

# Empreinte carbone (dernière valeur)
empreinte_actuelle = df["empreinte_tCO2_hab"].dropna().iloc[-1] if "empreinte_tCO2_hab" in df.columns else None

# Valeur projetée au slider selon le scénario choisi
temp_projetee = None
if scenarios is not None:
    sc_filter = scenarios[
        (scenarios["scenario"] == scenario_choisi) &
        (scenarios["annee"] <= annee_projection)
    ]
    if not sc_filter.empty:
        temp_projetee = sc_filter.iloc[-1]["temp_proj_C"]


# ==============================================================================
# ONGLETS PRINCIPAUX
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vue d'ensemble",
    "Historique",
    "Carte",
    "Projections",
    "Alertes"
])


# ==============================================================================
# ONGLET 1 — VUE D'ENSEMBLE
# ==============================================================================

with tab1:
    st.markdown("## Vue d'ensemble du climat en France")
    st.caption(f"Données jusqu'en {annee_recente} — Projection {scenario_choisi} à {annee_projection}")

    # --- Ligne de KPIs ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        couleur = "#ff5050" if anomalie_temp > 2 else "#ffa500" if anomalie_temp > 1 else "#00c864"
        st.markdown(f"""
        <div class="kpi-card" style="border-color:{couleur}">
            <p class="kpi-label"> Température moyenne France ({annee_recente})</p>
            <p class="kpi-value" style="color:{couleur}">{temp_recente:.1f}°C</p>
            <p class="kpi-delta" style="color:{couleur}">+{anomalie_temp:.2f}°C vs baseline 1900-1920</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        co2_str = f"{co2_actuel:.0f} ppm" if co2_actuel else "N/A"
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#ff8c00">
            <p class="kpi-label"> CO₂ atmosphérique (Mauna Loa)</p>
            <p class="kpi-value" style="color:#ff8c00">{co2_str}</p>
            <p class="kpi-delta" style="color:#aaa">Réf. pré-industrielle : ~280 ppm</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        niv_str = f"+{niveau_actuel:.0f} mm" if niveau_actuel else "N/A"
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#4d9fff">
            <p class="kpi-label"> Niveau de la mer (anomalie/1961-90)</p>
            <p class="kpi-value" style="color:#4d9fff">{niv_str}</p>
            <p class="kpi-delta" style="color:#aaa">Moyenne 36 ports français</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        emp_str = f"{empreinte_actuelle:.1f} t" if empreinte_actuelle else "N/A"
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#c084fc">
            <p class="kpi-value" style="color:#c084fc">{emp_str} CO₂/hab</p>
            <p class="kpi-label"> Empreinte carbone individuelle</p>
            <p class="kpi-delta" style="color:#aaa">Objectif ADEME : &lt; 2t/hab</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- Graphique central : température historique + projection scénario ---
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        st.markdown('<p class="section-title">Température historique + projection</p>', unsafe_allow_html=True)

        fig = go.Figure()

        # Historique
        df_hist_temp = df[["annee", "temp_moy_france"]].dropna()
        df_hist_temp = df_hist_temp[df_hist_temp["temp_moy_france"] > 9.5]
        fig.add_trace(go.Scatter(
            x=df_hist_temp["annee"], y=df_hist_temp["temp_moy_france"],
            mode="lines", name="Observations",
            line=dict(color="white", width=1.5)
        ))

        # Baseline
        fig.add_hline(y=baseline, line_dash="dot", line_color="gray",
                      annotation_text=f"Baseline 1900-1920 ({baseline:.1f}°C)")

        # Forecast Prophet
        if "temperature" in forecasts:
            fc = forecasts["temperature"]
            fc_fut = fc[fc["annee"] > annee_recente]
            fig.add_trace(go.Scatter(
                x=fc_fut["annee"], y=fc_fut["yhat"],
                mode="lines", name="Tendance Prophet",
                line=dict(color="steelblue", dash="dash", width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_fut["annee"], fc_fut["annee"][::-1]]),
                y=pd.concat([fc_fut["yhat_upper"], fc_fut["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(70,130,180,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="IC 95%"
            ))

        # Scénario sélectionné
        if scenarios is not None:
            couleurs = {"optimiste": "#00c864", "intermediaire": "#ffa500", "pessimiste": "#ff5050"}
            labels   = {"optimiste": "Optimiste +1.4°C", "intermediaire": "Intermédiaire +2.7°C", "pessimiste": "Pessimiste +4.4°C"}
            df_sc = scenarios[scenarios["scenario"] == scenario_choisi]
            pts = [{"annee": annee_recente, "temp_proj_C": temp_recente}]
            for _, r in df_sc[df_sc["annee"] > annee_recente].iterrows():
                pts.append({"annee": r["annee"], "temp_proj_C": r["temp_proj_C"]})
            df_pts = pd.DataFrame(pts)
            fig.add_trace(go.Scatter(
                x=df_pts["annee"], y=df_pts["temp_proj_C"],
                mode="lines+markers", name=labels[scenario_choisi],
                line=dict(color=couleurs[scenario_choisi], width=2.5),
                marker=dict(size=8)
            ))

        fig.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Année", yaxis_title="°C",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=20, t=20, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        st.markdown('<p class="section-title">CO₂ atmosphérique</p>', unsafe_allow_html=True)

        if "co2_ppm" in df.columns:
            df_co2 = df[["annee", "co2_ppm"]].dropna()
            fig_co2 = go.Figure()
            fig_co2.add_trace(go.Scatter(
                x=df_co2["annee"], y=df_co2["co2_ppm"],
                fill="tozeroy", line=dict(color="#ff8c00"),
                fillcolor="rgba(255,140,0,0.2)", name="CO₂"
            ))
            if "co2" in forecasts:
                fc_co2 = forecasts["co2"]
                fc_fut = fc_co2[fc_co2["annee"] > df_co2["annee"].max()]
                fig_co2.add_trace(go.Scatter(
                    x=fc_fut["annee"], y=fc_fut["yhat"],
                    line=dict(color="#ff8c00", dash="dash"), name="Projection"
                ))
            fig_co2.update_layout(
                template="plotly_dark", height=380,
                xaxis_title="Année", yaxis_title="ppm",
                showlegend=False, margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_co2, use_container_width=True)


# ==============================================================================
# ONGLET 2 — HISTORIQUE
# ==============================================================================

with tab2:
    st.markdown("## Évolution historique des indicateurs")

    # Sélecteur de variable
    variables_dispo = {
        "Température (°C)":             "temp_moy_france",
        "CO₂ (ppm)":                    "co2_ppm",
        "Niveau de la mer (mm)":        "niveau_mer_mm",
        "Jours chauds ≥30°C":           "jours_chauds_30",
        "Jours de gel ≤0°C":            "jours_gel",
        "Empreinte carbone (tCO₂/hab)": "empreinte_tCO2_hab",
        "Vendanges (jour de l'année)":  "jour_vendanges",
        "Coût catastrophes (Mrd USD)":  "dommages_Mrd_USD",
    }
    variables_dispo = {k: v for k, v in variables_dispo.items() if v in df.columns}

    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        var_label = st.selectbox("Variable à afficher", list(variables_dispo.keys()))
    with col_sel2:
        afficher_forecast = st.checkbox("Afficher la projection Prophet", value=True)

    col_y = variables_dispo[var_label]
    nom_forecast = {v: k2 for k2, v in {
        "temperature": "temp_moy_france", "co2": "co2_ppm",
        "niveau_mer": "niveau_mer_mm", "jours_chauds": "jours_chauds_30",
        "jours_gel": "jours_gel", "empreinte_carbone": "empreinte_tCO2_hab",
        "vendanges": "jour_vendanges", "cout_catastrophes": "dommages_Mrd_USD"
    }.items()}.get(col_y)

    df_var = df_filtre[["annee", col_y]].dropna()

    fig_hist = go.Figure()

    # Barres ou ligne selon la variable
    if col_y == "dommages_Mrd_USD":
        fig_hist.add_trace(go.Bar(
            x=df_var["annee"], y=df_var[col_y],
            name="Observations", marker_color="#ff5050"
        ))
    else:
        fig_hist.add_trace(go.Scatter(
            x=df_var["annee"], y=df_var[col_y],
            mode="lines", name="Observations",
            line=dict(color="white", width=1.5)
        ))
        # Ligne de tendance (moyenne mobile 10 ans)
        if len(df_var) >= 10:
            df_var = df_var.copy()
            df_var["tendance"] = df_var[col_y].rolling(10, center=True).mean()
            fig_hist.add_trace(go.Scatter(
                x=df_var["annee"], y=df_var["tendance"],
                mode="lines", name="Tendance (moy. 10 ans)",
                line=dict(color="#ffa500", width=2, dash="dash")
            ))

    # Forecast Prophet
    if afficher_forecast and nom_forecast in forecasts:
        fc = forecasts[nom_forecast]
        fc_fut = fc[fc["annee"] > df_var["annee"].max()]
        fig_hist.add_trace(go.Scatter(
            x=fc_fut["annee"], y=fc_fut["yhat"],
            mode="lines", name="Projection Prophet",
            line=dict(color="steelblue", dash="dash", width=1.5)
        ))
        fig_hist.add_trace(go.Scatter(
            x=pd.concat([fc_fut["annee"], fc_fut["annee"][::-1]]),
            y=pd.concat([fc_fut["yhat_upper"], fc_fut["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(70,130,180,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%", showlegend=True
        ))

    fig_hist.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Année", yaxis_title=var_label,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=20, b=80)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Stats rapides
    st.divider()
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Minimum", f"{df_var[col_y].min():.2f}")
    with col_s2:
        st.metric("Maximum", f"{df_var[col_y].max():.2f}")
    with col_s3:
        st.metric("Moyenne", f"{df_var[col_y].mean():.2f}")
    with col_s4:
        annee_max_val = int(df_var.loc[df_var[col_y].idxmax(), "annee"])
        st.metric("Année record", str(annee_max_val))


# ==============================================================================
# ONGLET 3 — CARTE INTERACTIVE
# ==============================================================================

with tab3:
    st.markdown("## Carte des ports français — Niveau de la mer")
    st.caption("Anomalie du niveau de la mer par rapport à la période de référence 1961-1990 (en mm)")

    col_carte, col_info = st.columns([3, 1])

    with col_info:
        st.markdown("### Légende")
        st.markdown("""
        🔵 **Bleu** : Niveau stable ou légèrement en hausse (< 50mm)

        🟡 **Jaune** : Hausse modérée (50-100mm)

        🔴 **Rouge** : Hausse significative (> 100mm)

        La taille du cercle est proportionnelle à la hausse mesurée.
        """)
        st.divider()
        st.markdown("### Stats ports")
        ports_avec_data = df_ports.dropna(subset=["niveau_mm"])
        if not ports_avec_data.empty:
            st.metric("Ports suivis", len(df_ports))
            st.metric("Hausse max", f"+{ports_avec_data['niveau_mm'].max():.0f} mm")
            st.metric("Hausse moyenne", f"+{ports_avec_data['niveau_mm'].mean():.0f} mm")

    with col_carte:
        # Carte centrée sur la France
        m = folium.Map(
            location=[46.5, 2.5],
            zoom_start=6,
            tiles="CartoDB dark_matter"
        )

        # Ajout des ports
        for _, port in df_ports.iterrows():
            niveau = port["niveau_mm"]
            if pd.isna(niveau):
                couleur = "gray"
                rayon   = 6
                popup_txt = f"<b>{port['ville']}</b><br>Données insuffisantes"
            else:
                # Couleur selon le niveau de hausse
                if niveau < 50:
                    couleur = "#4d9fff"
                elif niveau < 100:
                    couleur = "#ffa500"
                else:
                    couleur = "#ff5050"
                rayon = max(6, min(20, abs(niveau) / 8))
                popup_txt = f"<b>{port['ville']}</b><br>Anomalie : <b>{niveau:.0f} mm</b><br>par rapport à 1961-1990"

            folium.CircleMarker(
                location=[port["lat"], port["lon"]],
                radius=rayon,
                color=couleur,
                fill=True,
                fill_color=couleur,
                fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=200),
                tooltip=f"{port['ville']} : {niveau:.0f} mm" if not pd.isna(niveau) else port["ville"]
            ).add_to(m)

        st_folium(m, width=None, height=500)


# ==============================================================================
# ONGLET 4 — PROJECTIONS ET SCÉNARIOS
# ==============================================================================

with tab4:
    st.markdown("## Projections climatiques 2030 / 2050 / 2100")

    # --- Comparaison des 3 scénarios ---
    st.markdown('<p class="section-title">Comparaison des 3 scénarios de température</p>',
                unsafe_allow_html=True)

    if scenarios is not None:
        fig_sc = go.Figure()

        # Historique
        df_hist_temp = df[["annee", "temp_moy_france"]].dropna()
        df_hist_temp = df_hist_temp[df_hist_temp["temp_moy_france"] > 9.5]
        fig_sc.add_trace(go.Scatter(
            x=df_hist_temp["annee"], y=df_hist_temp["temp_moy_france"],
            mode="lines", name="Observations historiques",
            line=dict(color="white", width=2)
        ))

        # Tendance Prophet
        if "temperature" in forecasts:
            fc_t = forecasts["temperature"]
            fc_fut = fc_t[fc_t["annee"] > df_hist_temp["annee"].max()]
            fig_sc.add_trace(go.Scatter(
                x=fc_fut["annee"], y=fc_fut["yhat"],
                mode="lines", name="Tendance Prophet (sans contrainte)",
                line=dict(color="steelblue", dash="dot", width=1.5)
            ))

        # 3 scénarios
        config_sc = {
            "optimiste":     ("#00c864", "Optimiste +1.4°C"),
            "intermediaire": ("#ffa500", "Intermédiaire +2.7°C"),
            "pessimiste":    ("#ff5050", "Pessimiste +4.4°C"),
        }
        derniere_temp_filtre = df_hist_temp["temp_moy_france"].iloc[-1]
        derniere_annee_filtre = int(df_hist_temp["annee"].iloc[-1])

        for sc, (coul, label) in config_sc.items():
            df_sc = scenarios[scenarios["scenario"] == sc].copy()
            pts_x = [derniere_annee_filtre] + df_sc[df_sc["annee"] > derniere_annee_filtre]["annee"].tolist()
            pts_y = [derniere_temp_filtre] + df_sc[df_sc["annee"] > derniere_annee_filtre]["temp_proj_C"].tolist()
            fig_sc.add_trace(go.Scatter(
                x=pts_x, y=pts_y,
                mode="lines+markers", name=label,
                line=dict(color=coul, width=2.5),
                marker=dict(size=10, symbol="circle")
            ))

        # Ligne baseline
        fig_sc.add_hline(y=baseline, line_dash="dot", line_color="gray",
                         annotation_text=f"Baseline 1900-1920 ({baseline:.1f}°C)")

        # Marqueur sur l'année du slider
        fig_sc.add_vline(x=annee_projection, line_dash="dash", line_color="#888",
                         annotation_text=str(annee_projection))

        fig_sc.update_layout(
            template="plotly_dark", height=450,
            xaxis_title="Année", yaxis_title="Température (°C)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=20, t=20, b=80)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # --- Tableau récapitulatif ---
    st.divider()
    st.markdown('<p class="section-title">Projections à l\'horizon sélectionné</p>',
                unsafe_allow_html=True)

    if scenarios is not None:
        col_t1, col_t2, col_t3 = st.columns(3)
        cols_sc = [col_t1, col_t2, col_t3]
        config_display = [
            ("optimiste",     "🟢 Optimiste",     "#00c864"),
            ("intermediaire", "🟠 Intermédiaire", "#ffa500"),
            ("pessimiste",    "🔴 Pessimiste",    "#ff5050"),
        ]
        for col_widget, (sc, label, coul) in zip(cols_sc, config_display):
            df_sc = scenarios[scenarios["scenario"] == sc]
            val_proche = df_sc[df_sc["annee"] <= annee_projection].iloc[-1] if not df_sc[df_sc["annee"] <= annee_projection].empty else None
            with col_widget:
                if val_proche is not None:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-color:{coul}; text-align:center">
                        <p class="kpi-label">{label}</p>
                        <p class="kpi-value" style="color:{coul}">{val_proche['temp_proj_C']:.1f}°C</p>
                        <p class="kpi-delta" style="color:{coul}">+{val_proche['anomalie_C']:.2f}°C vs 1900-1920</p>
                        <p class="kpi-label">en {int(val_proche['annee'])}</p>
                    </div>""", unsafe_allow_html=True)

    # --- Forecasts individuels ---
    st.divider()
    st.markdown('<p class="section-title">Projections Prophet par indicateur</p>',
                unsafe_allow_html=True)

    noms_affichage = {
        "co2":               ("CO₂ (ppm)",              "#ff8c00"),
        "niveau_mer":        ("Niveau de la mer (mm)",   "#4d9fff"),
        "jours_chauds":      ("Jours chauds ≥30°C",      "#ff5050"),
        "vendanges":         ("Vendanges (DOY)",          "#a78bfa"),
        "empreinte_carbone": ("Empreinte carbone (t/hab)","#c084fc"),
    }
    fc_disponibles = {k: v for k, v in noms_affichage.items() if k in forecasts}

    if fc_disponibles:
        cols_fc = st.columns(min(3, len(fc_disponibles)))
        for i, (nom, (label, coul)) in enumerate(fc_disponibles.items()):
            with cols_fc[i % 3]:
                fc = forecasts[nom]
                fc_fut = fc[fc["annee"] <= annee_projection]
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=fc_fut["annee"], y=fc_fut["yhat"],
                    fill="tozeroy", line=dict(color=coul),
                    fillcolor=f"rgba({int(coul[1:3],16)},{int(coul[3:5],16)},{int(coul[5:7],16)},0.15)"
                ))
                val_fin = fc_fut["yhat"].iloc[-1]
                fig_mini.add_annotation(
                    x=fc_fut["annee"].iloc[-1], y=val_fin,
                    text=f"<b>{val_fin:.1f}</b>",
                    showarrow=False, font=dict(color=coul, size=12)
                )
                fig_mini.update_layout(
                    template="plotly_dark", height=200,
                    title=dict(text=label, font=dict(size=12)),
                    showlegend=False,
                    margin=dict(l=30, r=10, t=40, b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_mini, use_container_width=True)


# ==============================================================================
# ONGLET 5 — ALERTES
# ==============================================================================

with tab5:
    st.markdown("## Indicateurs d'alerte climatique")
    st.caption("Seuils définis d'après les recommandations GIEC / Accord de Paris")

    def alerte(label, valeur, seuil_orange, seuil_rouge, unite, description):
        if valeur is None:
            st.markdown(f"""<div class="alert-verte">
                <b>{label}</b> — Données non disponibles</div>""", unsafe_allow_html=True)
            return
        if valeur >= seuil_rouge:
            classe, emoji = "alert-rouge", "🔴"
        elif valeur >= seuil_orange:
            classe, emoji = "alert-orange", "🟡"
        else:
            classe, emoji = "alert-verte", "🟢"
        st.markdown(f"""<div class="{classe}">
            {emoji} <b>{label}</b> — Valeur actuelle : <b>{valeur:.2f} {unite}</b>
            <br><small>{description}</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Température")
    alerte("Anomalie de température",
           anomalie_temp, 1.5, 2.0, "°C",
           "Seuils : 🟡 +1.5°C (objectif Accord de Paris) — 🔴 +2.0°C (seuil critique GIEC)")

    st.markdown("### CO₂")
    alerte("Concentration CO₂",
           co2_actuel, 400, 450, "ppm",
           "Seuils : 🟡 400 ppm (franchis en 2013) — 🔴 450 ppm (compatible +2°C)")

    st.markdown("### Niveau de la mer")
    alerte("Hausse du niveau de la mer",
           niveau_actuel, 100, 200, "mm",
           "Anomalie vs 1961-1990. Seuils : 🟡 +10 cm — 🔴 +20 cm")

    st.markdown("### Empreinte carbone individuelle")
    alerte("Empreinte carbone individuelle",
           empreinte_actuelle, 5, 9, "t CO₂/hab",
           "Seuils : 🟡 5t (moyenne mondiale) — 🔴 9t (niveau actuel France ~2010). Objectif : 2t")

    st.divider()

    # Jauge température
    st.markdown("### Jauge de réchauffement")
    fig_jauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=anomalie_temp,
        delta={"reference": 1.5, "valueformat": ".2f",
               "increasing": {"color": "#ff5050"},
               "decreasing": {"color": "#00c864"}},
        title={"text": "Anomalie de température (°C) vs baseline 1900-1920"},
        gauge={
            "axis": {"range": [0, 4.5], "tickwidth": 1, "tickcolor": "white"},
            "bar":  {"color": "#ff8c00"},
            "steps": [
                {"range": [0, 1.5], "color": "#00c864"},
                {"range": [1.5, 2.0], "color": "#ffa500"},
                {"range": [2.0, 4.5], "color": "#ff3333"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 2.0
            }
        },
        number={"suffix": "°C", "valueformat": ".2f"}
    ))
    fig_jauge.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=40, r=40, t=60, b=20)
    )
    st.plotly_chart(fig_jauge, use_container_width=True)

    st.caption("La ligne blanche sur la jauge représente le seuil critique de +2°C (GIEC).")