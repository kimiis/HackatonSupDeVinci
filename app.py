# ==============================================================================
# HACKATHON #26 - app.py
# Dashboard Streamlit - Visualisation des données climatiques France
# Lancer avec : python -m streamlit run app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(
    page_title="Climat France",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .alerte {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CHARGEMENT DES DONNÉES
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
    out = {}
    for nom in noms:
        path = f"data/resultats/forecast_{nom}.csv"
        if os.path.exists(path):
            fc = pd.read_csv(path)
            fc["annee"] = pd.to_numeric(fc["annee"], errors="coerce")
            out[nom] = fc
    return out

@st.cache_data
def load_scenarios():
    path = "data/resultats/scenarios_temperature.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_ports():
    df_ports = pd.read_excel("data/raw/id_port_fr.xlsx")
    df_mer   = pd.read_csv("data/clean/niveau_mer.csv")
    derniere_ligne = df_mer[df_mer["annee"] == df_mer["annee"].max()].iloc[0]
    rows = []
    for _, row in df_ports.iterrows():
        col_match = next((c for c in df_mer.columns if row["Ville"].lower()[:5] in c), None)
        niveau = derniere_ligne[col_match] if col_match else None
        rows.append({"ville": row["Ville"], "lat": row["Lat"],
                     "lon": row["Long"], "niveau_mm": niveau})
    return pd.DataFrame(rows)

df        = load_historique()
forecasts = load_forecasts()
scenarios = load_scenarios()
df_ports  = load_ports()


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("Climat France")
    st.caption("Données officielles — Météo France, NOAA, CITEPA, PSMSL, INSEE, EM-DAT")
    st.divider()

    annee_min = int(df["annee"].min())
    annee_max = int(df["annee"].max())
    periode = st.slider("Période affichée",
                        min_value=annee_min, max_value=annee_max,
                        value=(1950, annee_max))

    scenario_choisi = st.selectbox(
        "Scénario climatique",
        options=["optimiste", "intermediaire", "pessimiste"],
        index=1,
        format_func=lambda x: {
            "optimiste":     "Optimiste — +1.4°C en 2100",
            "intermediaire": "Intermédiaire — +2.7°C en 2100",
            "pessimiste":    "Pessimiste — +4.4°C en 2100"
        }[x]
    )

    annee_projection = st.slider("Horizon", min_value=2026, max_value=2100,
                                 value=2050, step=5)


# ==============================================================================
# CALCUL DES VALEURS CLÉS
# ==============================================================================

df_filtre = df[(df["annee"] >= periode[0]) & (df["annee"] <= periode[1])]

baseline      = df.loc[(df["annee"] >= 1900) & (df["annee"] <= 1920), "temp_moy_france"].mean()
df_temp_completes = df.loc[df["temp_moy_france"] > 9.5]
temp_recente  = df_temp_completes["temp_moy_france"].iloc[-1]
annee_recente = int(df_temp_completes["annee"].iloc[-1])
hausse_temp   = temp_recente - baseline

co2_actuel        = df["co2_ppm"].dropna().iloc[-1]        if "co2_ppm"           in df.columns else None
ports_ok          = df_ports.dropna(subset=["niveau_mm"])
niveau_actuel     = ports_ok["niveau_mm"].mean() if not ports_ok.empty else None
empreinte_actuelle= df["empreinte_tCO2_hab"].dropna().iloc[-1] if "empreinte_tCO2_hab" in df.columns else None

COULEURS_SC = {"optimiste": "#4ade80", "intermediaire": "#fb923c", "pessimiste": "#f87171"}
LABELS_SC   = {"optimiste": "Optimiste +1.4°C", "intermediaire": "Intermédiaire +2.7°C",
               "pessimiste": "Pessimiste +4.4°C"}


# ==============================================================================
# ONGLETS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Vue d'ensemble", "Historique", "Carte", "Projections", "Modeles IA", "Alertes", "Que faire ?"
])


# ── TAB 1 : VUE D'ENSEMBLE ────────────────────────────────────────────────────

with tab1:
    st.header("Le climat en France")
    st.caption(f"Dernière observation : {annee_recente} — scénario affiché : {LABELS_SC[scenario_choisi]}")

    # Ligne 1 — 4 KPI climatiques
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Température moyenne",     f"{temp_recente:.1f} °C",
                f"+{hausse_temp:.2f}°C depuis 1900")
    co2_delta = round(co2_actuel - 280) if co2_actuel else None
    col2.metric("CO₂ dans l'air",
                f"{co2_actuel:.0f} ppm" if co2_actuel else "—",
                f"+{co2_delta} ppm vs avant l'ère industrielle" if co2_delta else "—")
    col3.metric("Niveau de la mer",        f"+{niveau_actuel:.0f} mm" if niveau_actuel else "—",
                "Hausse totale depuis 1900")
    col4.metric("Empreinte individuelle", f"{empreinte_actuelle:.1f} t CO₂/hab" if empreinte_actuelle else "—",
                "Objectif 2050 : 2 t/hab")

    # Ligne 2 — 4 KPI supplémentaires
    # On prend la dernière valeur > 0 et l'année correspondante
    def last_valid(col, exclude_current_year=False):
        if col not in df.columns:
            return None, None
        s = df[["annee", col]].dropna(subset=[col])
        s = s[s[col] > 0]
        if exclude_current_year:
            import datetime as dt
            s = s[s["annee"] < dt.date.today().year]
        if s.empty:
            return None, None
        return s.iloc[-1][col], int(s.iloc[-1]["annee"])

    jours_chauds_actuel, an_chauds     = last_valid("jours_chauds_30", exclude_current_year=True)
    jours_gel_actuel,    an_gel        = last_valid("jours_gel", exclude_current_year=True)
    vendanges_actuel,    an_vendanges  = last_valid("jour_vendanges")
    catastrophes_actuel, an_catastro   = last_valid("dommages_Mrd_USD")
    ges_actuel,          an_ges        = last_valid("ges_total_MtCO2eq")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Jours chauds ≥ 30°C",   f"{jours_chauds_actuel:.0f} j/an" if pd.notna(jours_chauds_actuel) else "—",
                f"Moyenne nationale ({an_chauds})" if an_chauds else "—")
    col6.metric("Jours de gel ≤ 0°C",    f"{jours_gel_actuel:.0f} j/an"    if pd.notna(jours_gel_actuel)    else "—",
                f"Moyenne nationale ({an_gel})" if an_gel else "—")
    if pd.notna(vendanges_actuel):
        import datetime
        date_vendanges = (datetime.date(2024, 1, 1) + datetime.timedelta(days=int(vendanges_actuel) - 1)).strftime("%d %b").lstrip("0")
    else:
        date_vendanges = "—"
    col7.metric("Début des vendanges",    date_vendanges,
                f"Plus tôt = été chaud ({an_vendanges})" if an_vendanges else "—")
    col8.metric("Coût des catastrophes", f"{catastrophes_actuel:.1f} Mrd $" if pd.notna(catastrophes_actuel) else "—",
                f"Total annuel France ({an_catastro})" if an_catastro else "—")

    # Ligne 3 — émissions totales tous secteurs
    col9, _, _, _ = st.columns(4)
    col9.metric("Émissions GES totales",
                f"{ges_actuel:.0f} Mt CO₂eq" if pd.notna(ges_actuel) else "—",
                f"Tous secteurs France ({an_ges})" if an_ges else "—")

    st.divider()

    col_g, col_co2 = st.columns([2, 1])

    with col_g:
        st.subheader("Température — passé et futur")
        fig = go.Figure()

        df_ht = df_filtre[["annee", "temp_moy_france"]].dropna()
        fig.add_trace(go.Scatter(
            x=df_ht["annee"], y=df_ht["temp_moy_france"],
            mode="lines", name="Mesures réelles",
            line=dict(color="#4FC3F7", width=2)
        ))
        fig.add_hline(y=baseline, line_dash="dot", line_color="#888",
                      annotation_text=f"Niveau de référence 1900-1920 ({baseline:.1f}°C)")

        if "temperature" in forecasts:
            fc = forecasts["temperature"]
            ff = fc[fc["annee"] > annee_recente]
            fig.add_trace(go.Scatter(
                x=pd.concat([ff["annee"], ff["annee"][::-1]]),
                y=pd.concat([ff["yhat_upper"], ff["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(70,130,180,0.12)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=ff["annee"], y=ff["yhat"],
                mode="lines", name="Projection tendancielle",
                line=dict(color="#60a5fa", dash="dash", width=1.5)
            ))

        if scenarios is not None:
            coul = COULEURS_SC[scenario_choisi]
            df_sc = scenarios[scenarios["scenario"] == scenario_choisi]
            pts_x = [annee_recente] + df_sc[df_sc["annee"] > annee_recente]["annee"].tolist()
            pts_y = [temp_recente]  + df_sc[df_sc["annee"] > annee_recente]["temp_proj_C"].tolist()
            fig.add_trace(go.Scatter(
                x=pts_x, y=pts_y, mode="lines+markers",
                name=LABELS_SC[scenario_choisi],
                line=dict(color=coul, width=2.5), marker=dict(size=7)
            ))

        fig.update_layout(template="plotly_dark", height=360,
                          xaxis_title="Année", yaxis_title="°C",
                          legend=dict(orientation="h", y=-0.25),
                          margin=dict(l=40, r=20, t=10, b=70))
        st.plotly_chart(fig, use_container_width=True)

    with col_co2:
        st.subheader("CO₂ dans l'atmosphère")
        if "co2_ppm" in df.columns:
            df_co2 = df_filtre[["annee", "co2_ppm"]].dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_co2["annee"], y=df_co2["co2_ppm"],
                fill="tozeroy", line=dict(color="#fb923c"),
                fillcolor="rgba(251,146,60,0.18)", name="CO₂"
            ))
            if "co2" in forecasts:
                ff = forecasts["co2"][forecasts["co2"]["annee"] > df_co2["annee"].max()]
                fig2.add_trace(go.Scatter(
                    x=ff["annee"], y=ff["yhat"],
                    line=dict(color="#fb923c", dash="dash"), name="Projection"
                ))
            fig2.update_layout(template="plotly_dark", height=360,
                               xaxis_title="Année", yaxis_title="ppm",
                               showlegend=False, margin=dict(l=40, r=10, t=10, b=40))
            st.plotly_chart(fig2, use_container_width=True)


# ── TAB 2 : HISTORIQUE ────────────────────────────────────────────────────────

with tab2:
    st.header("Évolution des indicateurs")

    VARIABLES = {
        "Température moyenne (°C)":         "temp_moy_france",
        "CO₂ atmosphérique (ppm)":          "co2_ppm",
        "Niveau de la mer (mm)":            "niveau_mer_mm",
        "Jours de chaleur ≥ 30°C":          "jours_chauds_30",
        "Jours de gel ≤ 0°C":               "jours_gel",
        "Empreinte carbone (t CO₂/hab)":    "empreinte_tCO2_hab",
        "Date de vendanges (jour de l'an)": "jour_vendanges",
        "Coût des catastrophes (Mrd USD)":  "dommages_Mrd_USD",
        "Émissions GES total (Mt CO₂eq)":  "ges_total_MtCO2eq",
    }
    VARIABLES = {k: v for k, v in VARIABLES.items() if v in df.columns}

    MAP_FC = {"temp_moy_france": "temperature", "co2_ppm": "co2",
              "niveau_mer_mm": "niveau_mer", "jours_chauds_30": "jours_chauds",
              "jours_gel": "jours_gel", "empreinte_tCO2_hab": "empreinte_carbone",
              "jour_vendanges": "vendanges", "dommages_Mrd_USD": "cout_catastrophes"}

    col_a, col_b = st.columns([3, 1])
    with col_a:
        var_label = st.selectbox("Indicateur", list(VARIABLES.keys()))
    with col_b:
        show_proj = st.checkbox("Afficher la pour ", value=True)

    col_y  = VARIABLES[var_label]
    nom_fc = MAP_FC.get(col_y)
    df_var = df_filtre[["annee", col_y]].dropna()

    fig3 = go.Figure()
    if col_y == "dommages_Mrd_USD":
        fig3.add_trace(go.Bar(x=df_var["annee"], y=df_var[col_y],
                              name="Valeur annuelle", marker_color="#f87171"))
    else:
        fig3.add_trace(go.Scatter(x=df_var["annee"], y=df_var[col_y],
                                  mode="lines", name="Données réelles",
                                  line=dict(color="white", width=1.5)))
        if len(df_var) >= 10:
            df_var = df_var.copy()
            df_var["moy10"] = df_var[col_y].rolling(10, center=True).mean()
            fig3.add_trace(go.Scatter(x=df_var["annee"], y=df_var["moy10"],
                                      mode="lines", name="Tendance sur 10 ans",
                                      line=dict(color="#fb923c", width=2, dash="dash")))

    if show_proj and nom_fc in forecasts:
        fc = forecasts[nom_fc]
        ff = fc[fc["annee"] > df_var["annee"].max()]
        fig3.add_trace(go.Scatter(
            x=pd.concat([ff["annee"], ff["annee"][::-1]]),
            y=pd.concat([ff["yhat_upper"], ff["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(70,130,180,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Fourchette de projection"
        ))
        fig3.add_trace(go.Scatter(x=ff["annee"], y=ff["yhat"],
                                  mode="lines", name="Projection",
                                  line=dict(color="#60a5fa", dash="dash", width=1.5)))

    fig3.update_layout(template="plotly_dark", height=430,
                       xaxis_title="Année", yaxis_title=var_label,
                       legend=dict(orientation="h", y=-0.25),
                       margin=dict(l=40, r=20, t=10, b=80))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Minimum",      f"{df_var[col_y].min():.2f}")
    c2.metric("Maximum",      f"{df_var[col_y].max():.2f}")
    c3.metric("Moyenne",      f"{df_var[col_y].mean():.2f}")
    c4.metric("Année record", str(int(df_var.loc[df_var[col_y].idxmax(), "annee"])))

    # ── Graphique GES par secteur ──────────────────────────────────────────────
    COLS_SECTEURS = {
        "industrie_energie":                      "Industrie énergie",
        "industrie_manufacturiere_construction":  "Industrie manuf.",
        "traitement_centralise_dechets":          "Déchets",
        "batiments_residentiel_tertiaire":        "Bâtiments",
        "agriculture":                            "Agriculture",
        "transports":                             "Transports",
        "emissions_naturelles":                   "Émissions naturelles",
        "emnr":                                   "EMNR",
    }
    secteurs_dispo = [c for c in COLS_SECTEURS if c in df.columns]
    if secteurs_dispo:
        st.divider()
        st.subheader("Émissions GES par secteur (Mt CO₂eq)")
        df_ges_filtre = df_filtre[["annee"] + secteurs_dispo].dropna(subset=secteurs_dispo, how="all")
        COULEURS_GES = ["#f87171","#fb923c","#fbbf24","#4ade80","#34d399","#60a5fa","#a78bfa","#f472b6"]
        fig_ges = go.Figure()
        for col_s, coul_s in zip(secteurs_dispo, COULEURS_GES):
            df_s = df_ges_filtre[["annee", col_s]].dropna()
            fig_ges.add_trace(go.Scatter(
                x=df_s["annee"], y=df_s[col_s],
                mode="lines", name=COLS_SECTEURS[col_s],
                stackgroup="one", line=dict(color=coul_s, width=0.5),
                fillcolor=coul_s
            ))
        fig_ges.update_layout(template="plotly_dark", height=400,
                              xaxis_title="Année", yaxis_title="Mt CO₂eq",
                              legend=dict(orientation="h", y=-0.3),
                              margin=dict(l=40, r=20, t=10, b=100))
        st.plotly_chart(fig_ges, use_container_width=True)


# ── TAB 3 : CARTE ─────────────────────────────────────────────────────────────

with tab3:
    st.header("Hausse du niveau de la mer — ports français")
    st.caption("Hausse du niveau de la mer en mm depuis 1900. Cliquer sur un port pour le détail.")

    col_m, col_leg = st.columns([3, 1])

    with col_leg:
        st.markdown("**Couleur des cercles**")
        st.markdown("🔵 Moins de 50 mm\n\n🟡 50 à 100 mm\n\n🔴 Plus de 100 mm")
        st.markdown("*La taille est proportionnelle à la hausse.*")
        st.divider()
        ports_ok = df_ports.dropna(subset=["niveau_mm"])
        if not ports_ok.empty:
            st.metric("Ports suivis", len(df_ports))
            st.metric("Hausse maximale", f"+{ports_ok['niveau_mm'].max():.0f} mm")
            st.metric("Hausse moyenne",  f"+{ports_ok['niveau_mm'].mean():.0f} mm")

    with col_m:
        m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="CartoDB dark_matter")
        for _, port in df_ports.iterrows():
            n = port["niveau_mm"]
            if pd.isna(n):
                couleur, rayon = "gray", 5
                txt = f"<b>{port['ville']}</b><br>Données insuffisantes"
            else:
                couleur = "#4d9fff" if n < 50 else ("#ffa500" if n < 100 else "#ff5050")
                rayon   = max(5, min(20, abs(n) / 8))
                txt     = f"<b>{port['ville']}</b><br>+{n:.0f} mm de hausse depuis 1900"
            folium.CircleMarker(
                location=[port["lat"], port["lon"]], radius=rayon,
                color=couleur, fill=True, fill_color=couleur, fill_opacity=0.7,
                popup=folium.Popup(txt, max_width=200),
                tooltip=f"{port['ville']} : +{n:.0f} mm" if not pd.isna(n) else port["ville"]
            ).add_to(m)
        st_folium(m, width=None, height=500)


# ── TAB 4 : PROJECTIONS ───────────────────────────────────────────────────────

with tab4:
    st.header("Et demain ?")
    st.caption("Les projections sont basées sur les tendances historiques et les scénarios du GIEC.")

    if scenarios is not None:
        st.subheader("Température selon les 3 scénarios")

        fig4 = go.Figure()
        df_ht2 = df_temp_completes[["annee", "temp_moy_france"]]
        fig4.add_trace(go.Scatter(x=df_ht2["annee"], y=df_ht2["temp_moy_france"],
                                  mode="lines", name="Mesures réelles",
                                  line=dict(color="white", width=2)))

        if "temperature" in forecasts:
            ff = forecasts["temperature"]
            ff = ff[ff["annee"] > annee_recente]
            fig4.add_trace(go.Scatter(x=ff["annee"], y=ff["yhat"],
                                      mode="lines", name="Si la tendance continue",
                                      line=dict(color="#60a5fa", dash="dot", width=1.5)))

        for sc, coul in COULEURS_SC.items():
            df_sc = scenarios[scenarios["scenario"] == sc]
            xs = [annee_recente] + df_sc[df_sc["annee"] > annee_recente]["annee"].tolist()
            ys = [temp_recente]  + df_sc[df_sc["annee"] > annee_recente]["temp_proj_C"].tolist()
            fig4.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                      name=LABELS_SC[sc],
                                      line=dict(color=coul, width=2.5),
                                      marker=dict(size=9)))

        fig4.add_hline(y=baseline, line_dash="dot", line_color="#555",
                       annotation_text=f"Niveau de référence 1900-1920 ({baseline:.1f}°C)")
        fig4.add_vline(x=annee_projection, line_dash="dash", line_color="#666",
                       annotation_text=str(annee_projection))

        fig4.update_layout(template="plotly_dark", height=430,
                           xaxis_title="Année", yaxis_title="°C",
                           legend=dict(orientation="h", y=-0.25),
                           margin=dict(l=40, r=20, t=10, b=80))
        st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader(f"Température projetée en {annee_projection}")
        c1, c2, c3 = st.columns(3)
        for col_w, sc in zip([c1, c2, c3], ["optimiste", "intermediaire", "pessimiste"]):
            df_sc = scenarios[scenarios["scenario"] == sc]
            row = df_sc[df_sc["annee"] <= annee_projection].iloc[-1]
            col_w.metric(LABELS_SC[sc],
                         f"{row['temp_proj_C']:.1f} °C",
                         f"+{row['anomalie_C']:.2f}°C depuis 1900")

    st.divider()
    st.subheader("Autres indicateurs projetés")
    MINI = {
        "co2":               ("CO₂ (ppm)",              "#fb923c"),
        "niveau_mer":        ("Niveau de la mer (mm)",   "#60a5fa"),
        "empreinte_carbone": ("Empreinte carbone (t/hab)","#c084fc"),
        "vendanges":         ("Vendanges (jour de l'an)","#a78bfa"),
    }
    disponibles = {k: v for k, v in MINI.items() if k in forecasts}
    if disponibles:
        cols_m = st.columns(len(disponibles))
        for col_w, (nom, (label, coul)) in zip(cols_m, disponibles.items()):
            with col_w:
                fc = forecasts[nom][forecasts[nom]["annee"] <= annee_projection]
                r, g, b = int(coul[1:3],16), int(coul[3:5],16), int(coul[5:7],16)
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(x=fc["annee"], y=fc["yhat"],
                                           fill="tozeroy", line=dict(color=coul),
                                           fillcolor=f"rgba({r},{g},{b},0.15)"))
                val = fc["yhat"].iloc[-1]
                fig_m.add_annotation(x=fc["annee"].iloc[-1], y=val,
                                     text=f"<b>{val:.1f}</b>",
                                     showarrow=False, font=dict(color=coul, size=11))
                fig_m.update_layout(template="plotly_dark", height=190,
                                    title=dict(text=label, font=dict(size=11)),
                                    showlegend=False,
                                    margin=dict(l=30, r=10, t=35, b=15),
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=False))
                st.plotly_chart(fig_m, use_container_width=True)

# ── TAB 5 : MODELES IA ────────────────────────────────────────────────────────

with tab5:
    st.header("Comparaison des modèles prédictifs")
    st.caption("Validation walk-forward : entraînement sur le passé, évaluation sur le futur.")

    path_csv = "data/resultats/comparaison_modeles.csv"
    path_png = "data/resultats/comparaison_modeles.png"

    if not os.path.exists(path_csv):
        st.warning("Lance d'abord `python modele_comparaison.py` pour générer les résultats.")
    else:
        df_comp = pd.read_csv(path_csv)

        # ── Explication des modèles ──────────────────────────────────────────
        st.subheader("Les 3 modèles comparés")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Régression Linéaire**")
            st.markdown("Modèle le plus simple : une droite de tendance `y = a × année + b`. "
                        "Rapide et explicable. Performant quand la tendance est constante.")
        with c2:
            st.markdown("**ARIMA(1,1,1)**")
            st.markdown("Modèle classique de séries temporelles. Capture les corrélations "
                        "entre années successives. Très efficace sur les séries régulières comme le CO₂.")
        with c3:
            st.markdown("**Prophet (Meta)**")
            st.markdown("Modèle avancé de décomposition de tendance. Robuste aux valeurs "
                        "manquantes et aux changements de tendance. Meilleur sur la température.")

        st.divider()

        # ── Tableau des métriques ────────────────────────────────────────────
        st.subheader("Performances — RMSE et MAPE par variable")
        st.caption("RMSE : erreur en unité de la variable (plus bas = meilleur). "
                   "MAPE : erreur en % (permet de comparer des variables d'unités différentes).")

        LABELS_MODELES = {
            "regression_lineaire": "Régression Linéaire",
            "arima":               "ARIMA(1,1,1)",
            "prophet":             "Prophet",
        }
        df_comp["modele"] = df_comp["modele"].map(LABELS_MODELES).fillna(df_comp["modele"])

        pivot_rmse = df_comp.pivot(index="variable", columns="modele", values="rmse").round(4)
        pivot_mape = df_comp.pivot(index="variable", columns="modele", values="mape").round(2)

        col_r, col_m = st.columns(2)
        with col_r:
            st.markdown("**RMSE**")
            st.dataframe(pivot_rmse, use_container_width=True)
        with col_m:
            st.markdown("**MAPE (%)**")
            st.dataframe(pivot_mape, use_container_width=True)

        st.divider()

        # ── Graphique barres ─────────────────────────────────────────────────
        st.subheader("Visualisation comparative — RMSE")
        COULEURS_MODELES = {
            "Régression Linéaire": "#60a5fa",
            "ARIMA(1,1,1)":        "#fb923c",
            "Prophet":             "#4ade80",
        }
        fig_comp = go.Figure()
        for modele in df_comp["modele"].unique():
            df_m = df_comp[df_comp["modele"] == modele]
            fig_comp.add_trace(go.Bar(
                name=modele,
                x=df_m["variable"],
                y=df_m["rmse"],
                marker_color=COULEURS_MODELES.get(modele, "#888"),
            ))
        fig_comp.update_layout(
            template="plotly_dark", barmode="group", height=380,
            xaxis_title="Variable", yaxis_title="RMSE (plus bas = meilleur)",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=10, b=80)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── Meilleur modèle par variable ─────────────────────────────────────
        st.divider()
        st.subheader("Meilleur modèle par variable")
        for var in df_comp["variable"].unique():
            df_v = df_comp[df_comp["variable"] == var].dropna(subset=["rmse"])
            if not df_v.empty:
                meilleur = df_v.loc[df_v["rmse"].idxmin(), "modele"]
                rmse_val = df_v["rmse"].min()
                mape_val = df_v.loc[df_v["rmse"].idxmin(), "mape"]
                st.markdown(f"**{var}** → {meilleur} "
                            f"(RMSE={rmse_val:.4f}, MAPE={mape_val:.2f}%)")

# ── TAB 6 : ALERTES ───────────────────────────────────────────────────────────

with tab6:
    st.header("Où en est-on ?")
    st.caption("Comparaison des valeurs actuelles aux seuils définis par l'Accord de Paris et le GIEC.")

    def afficher_alerte(label, valeur, seuil_orange, seuil_rouge, unite, explication):
        if valeur is None:
            st.info(f"**{label}** — données non disponibles")
            return
        if valeur >= seuil_rouge:
            couleur, icone = "#ef4444", "🔴"
            bg = "rgba(239,68,68,0.1)"
        elif valeur >= seuil_orange:
            couleur, icone = "#f97316", "🟡"
            bg = "rgba(249,115,22,0.1)"
        else:
            couleur, icone = "#22c55e", "🟢"
            bg = "rgba(34,197,94,0.1)"
        st.markdown(
            f'<div class="alerte" style="border-color:{couleur}; background:{bg}">'
            f'{icone} <b>{label}</b> — {valeur:.2f} {unite}'
            f'<br><small style="color:#aaa">{explication}</small></div>',
            unsafe_allow_html=True
        )

    afficher_alerte("Hausse de température",  hausse_temp,        1.5, 2.0, "°C",
        "Seuil orange : +1.5°C (objectif Accord de Paris) — Seuil rouge : +2.0°C")
    afficher_alerte("CO₂ dans l'atmosphère",  co2_actuel,         400, 450, "ppm",
        "400 ppm franchi en 2013 — 450 ppm correspond à un réchauffement de +2°C")
    afficher_alerte("Hausse du niveau de la mer", niveau_actuel,  100, 200, "mm",
        "Hausse depuis 1900 (réf. 1961-1990) — +10 cm : seuil orange, +20 cm : seuil rouge")
    afficher_alerte("Empreinte carbone par personne", empreinte_actuelle, 5, 9, "t CO₂",
        "Objectif ADEME pour 2050 : moins de 2 t par personne")

    st.divider()
    st.subheader("Jauge de réchauffement")

    fig_j = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=hausse_temp,
        delta={"reference": 1.5, "valueformat": ".2f",
               "increasing": {"color": "#f87171"},
               "decreasing": {"color": "#4ade80"}},
        title={"text": "Hausse de température depuis 1900 (°C)"},
        gauge={
            "axis": {"range": [0, 4.5], "tickwidth": 1},
            "bar":  {"color": "#fb923c"},
            "steps": [
                {"range": [0.0, 1.5], "color": "#166534"},
                {"range": [1.5, 2.0], "color": "#854d0e"},
                {"range": [2.0, 4.5], "color": "#7f1d1d"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": 2.0}
        },
        number={"suffix": " °C", "valueformat": ".2f"}
    ))
    fig_j.update_layout(template="plotly_dark", height=290,
                        margin=dict(l=40, r=40, t=50, b=10))
    st.plotly_chart(fig_j, use_container_width=True)
    st.caption("La ligne blanche marque le seuil critique de +2°C fixé par le GIEC.")

# ── TAB 7 : PRÉCONISATIONS CITOYENNES ─────────────────────────────────────────

with tab7:
    st.header("Que pouvons-nous faire ?")
    st.caption(
        "Préconisations issues du **Plan National d'Adaptation au Changement Climatique (PNACC-3)**, "
        "de l'**Earth Action Report 2025** et des **objectifs nationaux de neutralité carbone 2050**."
    )

    # ── Calcul des niveaux de risque à partir des données réelles ──────────────

    # Jours de chaleur : tendance sur les 10 dernières années
    jours_chauds_recents = None
    jours_chauds_anciens = None
    if "jours_chauds_30" in df.columns:
        serie_chauds = df[["annee", "jours_chauds_30"]].dropna()
        if len(serie_chauds) >= 20:
            jours_chauds_recents = serie_chauds[serie_chauds["annee"] >= serie_chauds["annee"].max() - 10]["jours_chauds_30"].mean()
            jours_chauds_anciens = serie_chauds[serie_chauds["annee"] <= serie_chauds["annee"].max() - 30]["jours_chauds_30"].mean()

    # Hausse de température
    risque_canicule = hausse_temp >= 1.5
    risque_eleve    = hausse_temp >= 2.0

    # Tendance jours chauds en hausse ?
    tendance_chaleur_forte = (
        jours_chauds_recents is not None and
        jours_chauds_anciens is not None and
        jours_chauds_recents > jours_chauds_anciens * 1.3  # +30% sur 30 ans
    )

    # Empreinte carbone vs objectif
    empreinte_elevee = empreinte_actuelle is not None and empreinte_actuelle > 5.0

    # ── Bandeau de synthèse ────────────────────────────────────────────────────

    st.subheader("Situation actuelle")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    def badge(col, label, actif, texte_actif, texte_inactif):
        if actif:
            col.error(f"**{label}**\n\n{texte_actif}")
        else:
            col.success(f"**{label}**\n\n{texte_inactif}")

    badge(col_r1, "Risque canicule",
          risque_canicule,
          f"+{hausse_temp:.1f}°C — seuil Accord de Paris dépassé",
          f"+{hausse_temp:.1f}°C — sous le seuil de 1.5°C")
    badge(col_r2, "Jours de chaleur extrême",
          tendance_chaleur_forte,
          f"{jours_chauds_recents:.0f} j/an en moyenne (hausse significative)" if jours_chauds_recents else "Données insuffisantes",
          f"{jours_chauds_recents:.0f} j/an — stable" if jours_chauds_recents else "Données insuffisantes")
    badge(col_r3, "CO₂ atmosphérique",
          co2_actuel is not None and co2_actuel > 400,
          f"{co2_actuel:.0f} ppm — au-dessus du seuil pré-industriel" if co2_actuel else "—",
          f"{co2_actuel:.0f} ppm" if co2_actuel else "—")
    badge(col_r4, "Empreinte carbone individuelle",
          empreinte_elevee,
          f"{empreinte_actuelle:.1f} t CO₂/hab — objectif : 2 t" if empreinte_actuelle else "—",
          f"{empreinte_actuelle:.1f} t CO₂/hab — sous le seuil" if empreinte_actuelle else "—")

    st.divider()

    # ── Préconisations thématiques ─────────────────────────────────────────────

    def section_preconisations(titre, icone, condition, urgence, actions, source):
        """
        Affiche un bloc de préconisations.
        urgence : "critique", "vigilance", "information"
        """
        couleurs = {"critique": "#ef4444", "vigilance": "#f97316", "information": "#60a5fa"}
        coul = couleurs.get(urgence, "#60a5fa")
        niveau_txt = {"critique": "⚠️ Risque élevé", "vigilance": "Vigilance recommandée", "information": "Bonnes pratiques"}[urgence]

        with st.expander(f"{icone} {titre} — {niveau_txt}", expanded=condition):
            st.markdown(f"<small style='color:{coul}'>Niveau : <b>{niveau_txt}</b></small>", unsafe_allow_html=True)
            st.markdown(f"*Source : {source}*")
            st.divider()
            cols = st.columns(len(actions))
            for col_a, (sous_titre, details) in zip(cols, actions):
                with col_a:
                    st.markdown(f"**{sous_titre}**")
                    for detail in details:
                        st.markdown(f"- {detail}")

    # ── 1. Risque de feux de forêt ─────────────────────────────────────────────
    section_preconisations(
        titre="Risque de feux de forêt",
        icone="🔥",
        condition=tendance_chaleur_forte or risque_eleve,
        urgence="critique" if risque_eleve else "vigilance",
        actions=[
            ("Débroussaillage", [
                "Débroussailler dans un rayon de 50 m autour des habitations (obligation légale L134-6)",
                "Éliminer les végétaux secs et broussailles avant l'été",
                "Maintenir les arbres à 3 m de hauteur de branche basse",
                "Tailler les haies mitoyennes avec les zones boisées",
            ]),
            ("Aménagement anti-incendie", [
                "Installer des citernes d'eau accessibles aux pompiers",
                "Créer des coupures vertes avec des espèces résistantes au feu",
                "Choisir des matériaux incombustibles pour les clôtures et terrasses",
                "Éloigner le bois de chauffage des bâtiments",
            ]),
            ("Procédures d'urgence", [
                "Consulter la carte Vigilance Météo France avant toute activité en forêt",
                "Préparer un plan d'évacuation familial avec deux itinéraires",
                "Tenir prêt un kit d'urgence (eau, médicaments, documents)",
                "Signaler tout départ de feu au 18 ou 112",
            ]),
        ],
        source="PNACC-3 axe 3 — Débroussaillement obligatoire (Code forestier L134-6)"
    )

    # ── 2. Sécheresse ─────────────────────────────────────────────────────────
    section_preconisations(
        titre="Sécheresse et tension sur l'eau",
        icone="🌵",
        condition=risque_canicule,
        urgence="critique" if risque_eleve else "vigilance",
        actions=[
            ("Réduire la consommation", [
                "Limiter l'arrosage aux heures fraîches (avant 8h, après 20h)",
                "Réparer immédiatement toute fuite — une fuite = 100 L/jour perdus",
                "Préférer les douches aux bains (5× moins d'eau)",
                "Couper l'eau lors du brossage des dents, savonnage",
            ]),
            ("Végétaux résistants", [
                "Remplacer les pelouses par des prairies fleuries sèches",
                "Privilégier les plantes méditerranéennes : lavande, romarin, olivier",
                "Pailler le pied des plantes pour réduire l'évaporation de 50%",
                "Regrouper les plantes à besoins similaires",
            ]),
            ("Récupération d'eau", [
                "Installer une cuve de récupération d'eau de pluie (300 à 1000 L)",
                "Réutiliser l'eau de cuisson refroidie pour arroser",
                "Collecter l'eau du lave-linge pour les toilettes si possible",
                "Orienter les gouttières vers le jardin ou une cuve",
            ]),
        ],
        source="PNACC-3 axe 1 — Gestion de l'eau / Earth Action Report 2025 objectif eau"
    )

    # ── 3. Canicule ───────────────────────────────────────────────────────────
    section_preconisations(
        titre="Adaptation aux vagues de chaleur",
        icone="🌡️",
        condition=risque_canicule,
        urgence="critique" if risque_eleve else "vigilance",
        actions=[
            ("Végétalisation urbaine", [
                "Planter des arbres à feuilles caduques côté sud des bâtiments",
                "Installer des toitures végétalisées (réduction jusqu'à 5°C)",
                "Créer des cours d'eau ou fontaines dans les espaces publics",
                "Participer aux initiatives de végétalisation de trottoirs (permis de végétaliser)",
            ]),
            ("Comportements individuels", [
                "Fermer volets et rideaux le jour, aérer la nuit",
                "S'hydrater régulièrement (1,5 L/jour minimum, plus en cas de chaleur)",
                "Repérer les climatisations publiques accessibles (mairies, médiathèques)",
                "Vérifier régulièrement les personnes âgées ou isolées du voisinage",
            ]),
            ("Rafraîchissement urbain", [
                "Soutenir les projets de désimperméabilisation des sols en mairie",
                "Favoriser les matériaux clairs à albédo élevé sur les toitures",
                "Promouvoir les couloirs de ventilation naturelle dans les PLU",
                "Utiliser les brumisateurs dans les espaces publics lors des pics",
            ]),
        ],
        source="PNACC-3 axe 2 — Santé / Plan canicule national / Earth Action Report 2025"
    )

    # ── 4. Empreinte carbone ──────────────────────────────────────────────────
    section_preconisations(
        titre="Réduire son empreinte carbone",
        icone="♻️",
        condition=True,  # toujours affiché
        urgence="critique" if empreinte_elevee else "vigilance",
        actions=[
            ("Mobilité douce", [
                "Remplacer 1 trajet voiture/semaine par le vélo ou les transports en commun",
                "Covoiturer : divise l'empreinte du trajet par le nombre de passagers",
                "Éviter l'avion pour les trajets < 4h en train (20× plus émetteur)",
                "Passer au véhicule électrique lors du prochain changement",
            ]),
            ("Alimentation bas carbone", [
                "Réduire la viande rouge à 1 repas/semaine (bœuf = 27 kg CO₂/kg)",
                "Privilégier les produits locaux et de saison",
                "Réduire le gaspillage alimentaire (8% de l'empreinte carbone française)",
                "Opter pour une alimentation végétale 3 jours sur 7",
            ]),
            ("Rénovation énergétique", [
                "Isoler les combles en priorité (30% des pertes de chaleur d'une maison)",
                "Remplacer une chaudière fioul/gaz par une pompe à chaleur",
                "Installer des panneaux solaires photovoltaïques",
                "Demander un bilan DPE et solliciter MaPrimeRénov' (aide de l'État)",
            ]),
        ],
        source="Neutralité carbone 2050 — Stratégie Nationale Bas-Carbone (SNBC) / ADEME"
    )

    st.divider()
    st.markdown(
        "**Aller plus loin :** "
        "[PNACC-3](https://www.ecologie.gouv.fr/pnacc) · "
        "[ADEME — Agir](https://agirpourlatransition.ademe.fr) · "
        "[Nos Gestes Climat](https://nosgestesclimat.fr) · "
        "[Earth Action Report 2025](https://www.unep.org)"
    )


