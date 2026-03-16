# ==============================================================================
# HACKATHON #26 - modele_prophet.py
#
# Objectif : construire des modèles prédictifs sur chaque indicateur climatique,
# évaluer leurs performances et produire des projections selon 3 scénarios.
#
# Ce fichier doit être lancé APRÈS pipeline.py qui produit dataset_final.csv.
#
# INSTALLATION :
#   pip install prophet matplotlib
# ==============================================================================

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import warnings
import os

# Prophet génère beaucoup de warnings internes (Stan, convergence, etc.)
# On les supprime pour garder une sortie console lisible.
warnings.filterwarnings("ignore")

# Dossier de sortie pour les fichiers CSV et PNG produits par la modélisation
os.makedirs("data/resultats", exist_ok=True)


# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

# On charge le fichier produit par pipeline.py.
# Chaque ligne est une année, chaque colonne un indicateur climatique.
# Les NaN représentent de vraies données manquantes (sources qui ne couvrent
# pas toute la période), ils seront filtrés dans preparer_df_prophet().
df = pd.read_csv("data/clean/dataset_final.csv")
df["annee"] = pd.to_numeric(df["annee"], errors="coerce")

# Prophet exige une colonne "ds" de type datetime.
# On convertit l'année entière en date au 1er janvier de chaque année.
# format="%Y" indique que l'entier représente uniquement une année.
df["ds"] = pd.to_datetime(df["annee"], format="%Y")

print(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Periode : {int(df['annee'].min())} - {int(df['annee'].max())}")
print(f"Colonnes : {list(df.columns)}")


# ==============================================================================
# VARIABLES À MODÉLISER
#
# Dictionnaire : nom_du_modele → colonne dans le dataset_final.csv
# Ce dictionnaire est le point central de configuration : pour ajouter ou
# retirer une variable de la modélisation, on modifie uniquement ce dict.
# ==============================================================================

VARIABLES = {
    "temperature":       "temp_moy_france",    # °C — série principale
    "co2":               "co2_ppm",            # ppm — depuis 1958
    "niveau_mer":        "niveau_mer_mm",       # mm anomalie/1961-1990
    "jours_chauds":      "jours_chauds_30",    # nb jours TX≥30°C — depuis 1950
    "jours_gel":         "jours_gel",           # nb jours TX≤0°C — depuis 1950
    "empreinte_carbone": "empreinte_tCO2_hab",  # tCO₂/hab — depuis 1990
    "vendanges":         "jour_vendanges",      # DOY (jour de l'année) — depuis 1900
    "cout_catastrophes": "dommages_Mrd_USD",    # milliards USD — depuis ~1950
}


# ==============================================================================
# HELPERS
# ==============================================================================

def preparer_df_prophet(df, col_y):
    """
    Prépare un DataFrame au format attendu par Prophet.

    Prophet exige exactement deux colonnes :
      - "ds" : timestamps (ici dates annuelles)
      - "y"  : valeurs à prédire

    Le dropna() est critique : Prophet ne tolère pas les NaN dans "y".
    On ne supprime que les lignes où la variable cible est manquante
    (les NaN des autres colonnes ne posent pas de problème ici).

    Paramètres :
        df    : DataFrame principal avec colonne "ds" et col_y
        col_y : nom de la colonne à modéliser

    Retourne :
        DataFrame avec colonnes ["ds", "y"], sans NaN, réindexé de 0 à N
    """
    sub = df[["ds", col_y]].dropna().copy()
    sub = sub.rename(columns={col_y: "y"})
    return sub.reset_index(drop=True)


def evaluer_modele(model, df_train):
    """
    Évalue un modèle Prophet par cross-validation temporelle (walk-forward).

    PRINCIPE DE LA CROSS-VALIDATION TEMPORELLE :
    Contrairement à la CV classique qui mélange aléatoirement les données,
    la CV temporelle respecte l'ordre chronologique. On entraîne toujours
    sur le passé et on prédit le futur, ce qui est la seule approche valide
    pour des séries temporelles.

    Paramètres de la CV Prophet :
      initial  : taille de la première fenêtre d'entraînement
                 (on prend 60% de la durée totale de la série)
      period   : décalage entre deux folds successifs (5 ans)
                 → on crée un nouveau fold tous les 5 ans
      horizon  : durée de prévision évaluée à chaque fold (5 ans)
                 → on mesure la précision sur les 5 années suivant l'entraînement

    MÉTRIQUES CALCULÉES (moyennées sur tous les folds) :
      RMSE : Root Mean Squared Error — sensible aux grandes erreurs
             = sqrt(mean((y_pred - y_true)²))
             → même unité que la variable (ex : °C pour la température)

      MAE  : Mean Absolute Error — robuste aux valeurs aberrantes
             = mean(|y_pred - y_true|)
             → même unité que la variable

      MAPE : Mean Absolute Percentage Error — erreur relative en %
             = mean(|y_pred - y_true| / y_true) × 100
             → utile pour comparer des variables d'unités différentes

    Paramètres :
        model    : modèle Prophet déjà entraîné
        df_train : DataFrame d'entraînement (colonnes ds, y)

    Retourne :
        dict {"rmse": float, "mae": float, "mape": float} ou None si échec
    """
    n_ans = df_train["ds"].dt.year.max() - df_train["ds"].dt.year.min()
    # Prophet attend les durées en jours pour les données annuelles.
    # On adapte les paramètres selon la longueur de la série :
    #   - Séries longues (>80 ans) : fenêtre initiale 60%, horizon 5 ans
    #   - Séries courtes (<80 ans) : fenêtre initiale 50%, horizon 3 ans
    #     pour garantir assez de folds d'évaluation
    if n_ans >= 80:
        initial_days = int(n_ans * 0.60 * 365)
        horizon_days = 5 * 365
    else:
        initial_days = int(n_ans * 0.50 * 365)
        horizon_days = 3 * 365
    initial_days = max(15 * 365, initial_days)  # minimum absolu : 15 ans
    initial  = f"{initial_days} days"
    period   = f"{horizon_days} days"
    horizon  = f"{horizon_days} days"
    try:
        df_cv  = cross_validation(model, initial=initial, period=period,
                                  horizon=horizon, disable_tqdm=True)
        m      = performance_metrics(df_cv, rolling_window=1)
        return {
            "rmse": round(m["rmse"].mean(), 4),
            "mae":  round(m["mae"].mean(),  4),
            "mape": round(m["mape"].mean() * 100, 2),  # Prophet retourne une fraction, on × 100
        }
    except Exception as e:
        print(f"      [!] Cross-validation impossible : {e}")
        return {"rmse": None, "mae": None, "mape": None}


# ==============================================================================
# ENTRAÎNEMENT DES MODÈLES + ÉVALUATION
#
# Pour chaque variable :
#   1. Préparation du DataFrame Prophet (ds, y) sans NaN
#   2. Configuration et entraînement du modèle Prophet
#   3. Projection jusqu'en 2100
#   4. Évaluation par cross-validation
#   5. Sauvegarde du forecast en CSV
# ==============================================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT DES MODELES INDIVIDUELS")
print("=" * 60)

modeles   = {}   # stocke les modèles entraînés {nom: Prophet}
forecasts = {}   # stocke les prédictions {nom: DataFrame}
metriques = {}   # stocke les métriques d'évaluation {nom: dict}

for nom, col in VARIABLES.items():
    if col not in df.columns:
        print(f"\n  [!] '{col}' absente - {nom} ignore")
        continue

    df_p = preparer_df_prophet(df, col)

    # Minimum de 20 points pour qu'un modèle Prophet soit statistiquement fiable
    if len(df_p) < 20:
        print(f"\n  [!] {nom} : seulement {len(df_p)} points, ignore")
        continue

    print(f"\n  -> {nom} ({len(df_p)} points : "
          f"{df_p['ds'].dt.year.min()}-{df_p['ds'].dt.year.max()})")

    # Transformation log pour les coûts de catastrophes :
    # Les données sont extrêmement irrégulières (0 certaines années, milliards
    # d'autres). log1p(x) = log(x+1) compresse les valeurs extrêmes et permet
    # à Prophet de converger. On retransformera avec expm1() pour interpréter.
    log_transform = (nom == "cout_catastrophes")
    if log_transform:
        df_p["y"] = np.log1p(df_p["y"])

    # Configuration Prophet :
    # yearly_seasonality=False  : inutile avec des données annuelles
    #                             (la saisonnalité mensuelle n'existe pas à cette granularité)
    # weekly_seasonality=False  : idem, pas de cycle hebdomadaire
    # daily_seasonality=False   : idem, pas de cycle journalier
    # interval_width=0.95       : intervalle de confiance à 95% sur les prédictions
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(df_p)
    modeles[nom] = m

    # Génération des dates futures jusqu'en 2100.
    # make_future_dataframe() crée un DataFrame avec toutes les dates historiques
    # + les dates futures. freq="YS" = Year Start = 1er janvier de chaque année.
    annees_restantes = 2100 - df_p["ds"].dt.year.max()
    future = m.make_future_dataframe(periods=annees_restantes, freq="YS")

    # m.predict() retourne un DataFrame avec :
    #   yhat       : prédiction centrale
    #   yhat_lower : borne inférieure de l'intervalle de confiance
    #   yhat_upper : borne supérieure de l'intervalle de confiance
    #   trend, trend_lower, trend_upper : composante de tendance
    fc = m.predict(future)
    fc["annee"] = fc["ds"].dt.year  # colonne pratique pour les jointures/filtres

    # Si on avait appliqué log1p, on retransforme avec expm1 pour revenir
    # en milliards USD lisibles. expm1(x) = exp(x) - 1, inverse exact de log1p.
    if log_transform:
        fc["yhat"]       = np.expm1(fc["yhat"])
        fc["yhat_lower"] = np.expm1(fc["yhat_lower"])
        fc["yhat_upper"] = np.expm1(fc["yhat_upper"])

    forecasts[nom] = fc

    # On sauvegarde uniquement les colonnes essentielles (le forecast complet
    # a ~20 colonnes de décomposition dont on n'a pas besoin ici)
    fc[["ds", "annee", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        f"data/resultats/forecast_{nom}.csv", index=False
    )

    # Évaluation par cross-validation temporelle
    print(f"     Evaluation en cours...")
    met = evaluer_modele(m, df_p)
    metriques[nom] = met

    rmse_str = f"{met['rmse']:.4f}" if met["rmse"] is not None else "N/A"
    mae_str  = f"{met['mae']:.4f}"  if met["mae"]  is not None else "N/A"
    mape_str = f"{met['mape']:.2f}%" if met["mape"] is not None else "N/A"
    print(f"     RMSE={rmse_str}  MAE={mae_str}  MAPE={mape_str}")


# ==============================================================================
# TABLEAU COMPARATIF DES PERFORMANCES
#
# Ce tableau permet de comparer objectivement les modèles entre eux.
# Un RMSE faible indique que le modèle est précis en valeur absolue.
# Un MAPE faible indique que l'erreur est faible en proportion de la valeur réelle.
# Pour comparer des variables d'unités différentes (°C vs mm vs ppm),
# le MAPE est la métrique la plus pertinente.
# ==============================================================================

print("\n" + "=" * 60)
print("COMPARAISON DES PERFORMANCES")
print("=" * 60)

# pd.DataFrame(dict).T : on transpose pour avoir les modèles en lignes
# et les métriques (rmse, mae, mape) en colonnes
df_perf = pd.DataFrame(metriques).T
df_perf.index.name = "modele"
df_perf = df_perf.reset_index()
print(df_perf.to_string(index=False))

# Note sur le MAPE du niveau de la mer :
# Les anomalies sont centrées sur zéro (par construction dans le pipeline).
# Le MAPE devient absurde quand la vraie valeur est proche de 0 (division par ~0).
# Pour ce modèle, se fier au RMSE uniquement.
if "niveau_mer" in df_perf["modele"].values:
    print("\n  [!] niveau_mer : MAPE non interprétable (anomalies centrées sur 0). "
          "Utiliser le RMSE.")

df_perf.to_csv("data/resultats/comparaison_performances.csv", index=False)


# ==============================================================================
# SCÉNARIOS CLIMATIQUES 2030 / 2050 / 2100
#
# DÉFINITION DES SCÉNARIOS :
# Les scénarios sont définis par une anomalie de température cible en 2100
# par rapport à la période pré-industrielle (1850-1900, référence GIEC).
# On utilise la moyenne 1900-1920 comme proxy de cette référence car
# les données de température disponibles débutent en 1900.
#
#   Optimiste     +1.4°C : correspond à l'objectif ambitieux de l'Accord de Paris
#                          (limiter le réchauffement à 1.5°C, avec une marge de
#                          ±0.1°C liée à l'incertitude de notre baseline)
#
#   Intermédiaire +2.7°C : correspond au scénario des politiques actuelles maintenues
#                          sans efforts supplémentaires (SSP2-4.5 du GIEC)
#
#   Pessimiste    +4.4°C : correspond au scénario "business as usual" sans aucune
#                          politique climatique (SSP5-8.5 du GIEC)
#
# MÉTHODE DE PROJECTION :
# On calcule une trajectoire linéaire entre la dernière température observée
# et la cible 2100. Les points 2030 et 2050 sont des interpolations sur cette droite.
# C'est une simplification : en réalité les trajectoires sont non-linéaires,
# mais cela donne une tendance indicative cohérente avec les cibles du GIEC.
#
# FORMULE :
# t = (annee - derniere_annee) / (2100 - derniere_annee)  ← progression de 0 à 1
# temp_proj = derniere_temp + t × (temp_cible_2100 - derniere_temp)
# ==============================================================================

print("\n" + "=" * 60)
print("SCENARIOS CLIMATIQUES")
print("=" * 60)

# Baseline pré-industrielle : moyenne des températures 1900-1920.
# C'est notre point de référence "zéro" pour calculer les anomalies.
baseline_temp = df.loc[
    (df["annee"] >= 1900) & (df["annee"] <= 1920), "temp_moy_france"
].mean()
print(f"\n  Baseline pre-industrielle (moy. 1900-1920) : {baseline_temp:.2f} °C")

# Dernière observation valide d'une ANNÉE COMPLÈTE.
# On filtre les valeurs < 9.5°C car une moyenne annuelle française est toujours
# supérieure à 10°C. Une valeur plus basse indique une année incomplète
# (ex : 2025/2026 avec seulement les mois d'hiver déjà publiés → fausse la baseline).
df_temp_obs   = df[["annee", "temp_moy_france"]].dropna()
df_temp_obs   = df_temp_obs[df_temp_obs["temp_moy_france"] > 9.5]
derniere_annee = int(df_temp_obs["annee"].iloc[-1])
derniere_temp  = df_temp_obs["temp_moy_france"].iloc[-1]
anomalie_actuelle = derniere_temp - baseline_temp
print(f"  Derniere observation ({derniere_annee}) : "
      f"{derniere_temp:.2f} °C  (anomalie : +{anomalie_actuelle:.2f} °C)")

# Définition des 3 scénarios avec leur couleur pour les graphiques
SCENARIOS = {
    "optimiste":     {"anomalie_2100": 1.4, "couleur": "green",  "label": "Optimiste (+1.4°C)"},
    "intermediaire": {"anomalie_2100": 2.7, "couleur": "orange", "label": "Intermediaire (+2.7°C)"},
    "pessimiste":    {"anomalie_2100": 4.4, "couleur": "red",    "label": "Pessimiste (+4.4°C)"},
}

ANNEES_CIBLES = [2030, 2050, 2100]  # horizons d'intérêt pour les projections
lignes_scenarios = []

for scenario, params in SCENARIOS.items():
    # Température absolue cible en 2100 = baseline + anomalie souhaitée
    temp_cible_2100 = baseline_temp + params["anomalie_2100"]

    for annee in ANNEES_CIBLES:
        if annee <= derniere_annee:
            # Pour les années déjà observées, on conserve la valeur réelle
            temp_proj = derniere_temp
        else:
            # Interpolation linéaire sur la trajectoire 2100
            # t varie de 0 (derniere_annee) à 1 (2100)
            t = (annee - derniere_annee) / (2100 - derniere_annee)
            temp_proj = derniere_temp + t * (temp_cible_2100 - derniere_temp)

        lignes_scenarios.append({
            "scenario":    scenario,
            "annee":       annee,
            "temp_proj_C": round(temp_proj, 2),
            "anomalie_C":  round(temp_proj - baseline_temp, 2),  # écart à la baseline
        })

df_scenarios = pd.DataFrame(lignes_scenarios)
print("\n  Projections de temperature par scenario :\n")

# pivot() : on réorganise le tableau pour avoir les scénarios en colonnes
# et les années en lignes → plus lisible pour comparer les scénarios
pivot = df_scenarios.pivot(index="annee", columns="scenario", values="temp_proj_C")
print(pivot.to_string())
df_scenarios.to_csv("data/resultats/scenarios_temperature.csv", index=False)


# ==============================================================================
# VISUALISATIONS
# ==============================================================================

# --- Graphique 1 : Températures historiques + tendance Prophet + 3 scénarios ---
#
# Ce graphique superpose 3 types d'information :
#   1. Les observations historiques (ligne noire) : ce qui s'est passé
#   2. La tendance Prophet extrapolée (ligne bleue pointillée) :
#      ce que le modèle prédit en prolongeant la tendance actuelle
#   3. Les 3 scénarios GIEC (lignes colorées) :
#      des trajectoires contraintes par des cibles de politique climatique

if "temperature" in forecasts:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Observations historiques : toutes les années avec une mesure réelle
    ax.plot(df_temp_obs["annee"], df_temp_obs["temp_moy_france"],
            "k-", linewidth=1.5, label="Observations", zorder=5)

    # Tendance Prophet : uniquement la partie future (après la dernière observation)
    fc_t = forecasts["temperature"]
    fc_future = fc_t[fc_t["annee"] > derniere_annee]

    # L'intervalle de confiance (zone bleue) représente l'incertitude du modèle.
    # Il s'élargit avec le temps car l'incertitude augmente sur le long terme.
    ax.fill_between(fc_future["annee"], fc_future["yhat_lower"], fc_future["yhat_upper"],
                    alpha=0.1, color="blue")
    ax.plot(fc_future["annee"], fc_future["yhat"],
            "b--", linewidth=1, alpha=0.6, label="Prophet (tendance extrapolee)")

    # Tracé des 3 scénarios climatiques
    for scenario, params in SCENARIOS.items():
        df_sc = df_scenarios[df_scenarios["scenario"] == scenario].copy()

        # On ajoute le point de départ (dernière observation) pour que la courbe
        # de scénario parte exactement de la fin des observations → raccord visuel
        depart = pd.DataFrame([{
            "scenario": scenario, "annee": derniere_annee,
            "temp_proj_C": derniere_temp, "anomalie_C": anomalie_actuelle
        }])
        df_sc = pd.concat([depart, df_sc[df_sc["annee"] > derniere_annee]], ignore_index=True)

        ax.plot(df_sc["annee"], df_sc["temp_proj_C"],
                color=params["couleur"], linewidth=2.5,
                marker="o", markersize=7, label=params["label"])

    # Ligne horizontale de référence : niveau pré-industriel
    ax.axhline(baseline_temp, color="gray", linestyle=":", alpha=0.6,
               label=f"Baseline pre-industrielle ({baseline_temp:.1f} °C)")

    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel("Temperature moyenne France (°C)", fontsize=12)
    ax.set_title("Projections climatiques France - 3 scenarios (2030 / 2050 / 2100)", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/resultats/scenarios_temperature.png", dpi=150)
    plt.close()
    print("\n  Graphique : data/resultats/scenarios_temperature.png")


# --- Graphique 2 : Forecast individuel pour chaque variable ---
#
# Pour chaque variable modélisée, on produit un graphique avec :
#   - Les points d'observation (points noirs)
#   - La prédiction centrale Prophet (ligne bleue)
#   - L'intervalle de confiance 95% (zone bleue transparente)
# Ces graphiques permettent d'évaluer visuellement la qualité du modèle
# et la plausibilité de l'extrapolation jusqu'en 2100.

for nom, fc in forecasts.items():
    col = VARIABLES.get(nom)
    if col not in df.columns:
        continue

    df_hist = df[["annee", col]].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Zone d'incertitude : plus elle est large, moins le modèle est certain
    ax.fill_between(fc["annee"], fc["yhat_lower"], fc["yhat_upper"],
                    alpha=0.2, color="steelblue", label="Intervalle de confiance 95%")

    # Prédiction centrale
    ax.plot(fc["annee"], fc["yhat"], color="steelblue", linewidth=1.5,
            label="Prediction Prophet")

    # Points d'observation réels (petits points pour ne pas surcharger le graphique)
    ax.plot(df_hist["annee"], df_hist[col], "k.", markersize=3, label="Observations")

    ax.set_title(f"Prophet - {nom}", fontsize=13)
    ax.set_xlabel("Annee")
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"data/resultats/forecast_{nom}.png", dpi=150)
    plt.close()


# ==============================================================================
# RÉSUMÉ FINAL
# ==============================================================================

print("\n" + "=" * 60)
print("Modelisation terminee.")
print("Fichiers produits dans data/resultats/ :")
for f in sorted(os.listdir("data/resultats")):
    print(f"  - {f}")
print("=" * 60)