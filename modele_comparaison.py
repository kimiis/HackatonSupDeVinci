# ==============================================================================
# HACKATHON #26 - modele_comparaison.py
#
# Objectif : comparer 5 modèles prédictifs sur les indicateurs climatiques
# et produire un tableau de performances (RMSE, MAE, MAPE).
#
# Modèles comparés :
#   1. Régression Linéaire  — baseline simple (tendance pure)
#   2. ARIMA(1,1,1)         — modèle classique de séries temporelles
#   3. Prophet              — modèle avancé de Meta
#   4. Random Forest        — ensemble d'arbres de décision
#   5. XGBoost              — gradient boosting
#
# Note sur les modèles arbre (RF, XGBoost) :
#   On utilise le numéro d'année comme seule feature, ce qui les rend comparables
#   à la régression linéaire. Leur limite principale en séries temporelles est
#   l'extrapolation : ils ne peuvent pas prédire au-delà du max/min vu à
#   l'entraînement. Ils sont donc moins adaptés aux projections long terme.
#
# Ce fichier doit être lancé APRÈS pipeline.py et modele_prophet.py.
#
# INSTALLATION :
#   pip install pandas numpy scikit-learn statsmodels prophet matplotlib mlflow xgboost
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.makedirs("data/resultats", exist_ok=True)

mlflow.set_experiment("hackathon26_comparaison_modeles")

# ==============================================================================
# CHARGEMENT
# ==============================================================================

df = pd.read_csv("data/clean/dataset_final.csv")
df["annee"] = pd.to_numeric(df["annee"], errors="coerce")

VARIABLES = {
    "temperature":  "temp_moy_france",
    "co2":          "co2_ppm",
    "niveau_mer":   "niveau_mer_mm",
    "vendanges":    "jour_vendanges",
}

print("=" * 70)
print("COMPARAISON DE MODELES PREDICTIFS")
print("Modeles : Regression Lineaire | ARIMA | Prophet | Random Forest | XGBoost")
print("=" * 70)


# ==============================================================================
# VALIDATION WALK-FORWARD
#
# Principe : on entraîne sur les N premières années et on prédit les K suivantes.
# On répète en avançant d'une fenêtre à chaque fois.
# → Respecte l'ordre chronologique (pas de fuite d'information du futur vers le passé)
#
# Exemple avec initial=70%, horizon=5 ans :
#   Fold 1 : train [1900-1970], test [1971-1975]
#   Fold 2 : train [1900-1975], test [1976-1980]
#   ...
# ==============================================================================

def walk_forward_validation(serie_annee, serie_y, initial_frac=0.70, horizon=5):
    """
    Évalue 5 modèles par validation walk-forward.

    Paramètres :
        serie_annee  : array des années
        serie_y      : array des valeurs cibles
        initial_frac : fraction des données pour le premier entraînement
        horizon      : nombre d'années à prédire à chaque fold

    Retourne :
        dict {modele: {"rmse": float, "mae": float, "mape": float}}
    """
    n = len(serie_y)
    start = int(n * initial_frac)

    erreurs = {
        "regression_lineaire": {"y_true": [], "y_pred": []},
        "arima":               {"y_true": [], "y_pred": []},
        "prophet":             {"y_true": [], "y_pred": []},
        "random_forest":       {"y_true": [], "y_pred": []},
        "xgboost":             {"y_true": [], "y_pred": []},
    }

    for i in range(start, n - horizon + 1, horizon):
        y_train = serie_y[:i]
        y_test  = serie_y[i:i + horizon]
        x_train = serie_annee[:i].reshape(-1, 1)
        x_test  = serie_annee[i:i + horizon].reshape(-1, 1)

        # ── Régression Linéaire ───────────────────────────────────────────────
        # Modèle : y = a × annee + b — droite de tendance.
        # Avantage : rapide, explicable, bon pour les tendances monotones.
        # Limite : ne capture pas les accélérations ou décélérations.
        try:
            reg = LinearRegression()
            reg.fit(x_train, y_train)
            preds_reg = reg.predict(x_test)
            erreurs["regression_lineaire"]["y_true"].extend(y_test)
            erreurs["regression_lineaire"]["y_pred"].extend(preds_reg)
        except Exception:
            pass

        # ── ARIMA(1,1,1) ─────────────────────────────────────────────────────
        # p=1 : terme autorégressif | d=1 : différenciation | q=1 : moyenne mobile
        # Avantage : capture la dynamique locale et les corrélations temporelles.
        # Limite : peut diverger sur long terme.
        try:
            arima = ARIMA(y_train, order=(1, 1, 1))
            arima_fit = arima.fit()
            preds_arima = arima_fit.forecast(steps=horizon)
            erreurs["arima"]["y_true"].extend(y_test)
            erreurs["arima"]["y_pred"].extend(preds_arima)
        except Exception:
            pass

        # ── Prophet ──────────────────────────────────────────────────────────
        # Décomposition additive : tendance + saisonnalité + résidus.
        # Avantage : robuste aux valeurs manquantes, gère les ruptures de tendance.
        # Limite : plus lent à entraîner.
        try:
            annees_train = serie_annee[:i]
            ds_train = pd.to_datetime(annees_train.astype(int), format="%Y")
            df_p = pd.DataFrame({"ds": ds_train, "y": y_train})

            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, interval_width=0.95)
            m.fit(df_p)

            annees_test = serie_annee[i:i + horizon]
            ds_test = pd.to_datetime(annees_test.astype(int), format="%Y")
            df_future = pd.DataFrame({"ds": ds_test})
            preds_prophet = m.predict(df_future)["yhat"].values

            erreurs["prophet"]["y_true"].extend(y_test)
            erreurs["prophet"]["y_pred"].extend(preds_prophet)
        except Exception:
            pass

        # ── Random Forest ─────────────────────────────────────────────────────
        # Ensemble de 200 arbres de décision entraînés sur des sous-échantillons.
        # Avantage : robuste au bruit, capture les relations non-linéaires.
        # Limite : pas de vrai mécanisme d'extrapolation — prédit une valeur
        #   constante (max vu à l'entraînement) pour les années futures.
        #   Moins adapté aux projections long terme que Prophet ou ARIMA.
        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(x_train, y_train)
            preds_rf = rf.predict(x_test)
            erreurs["random_forest"]["y_true"].extend(y_test)
            erreurs["random_forest"]["y_pred"].extend(preds_rf)
        except Exception:
            pass

        # ── XGBoost ───────────────────────────────────────────────────────────
        # Gradient boosting : construction séquentielle d'arbres pour corriger
        # les erreurs du modèle précédent.
        # Avantage : très performant sur les données tabulaires, gère bien les
        #   non-linéarités et les interactions.
        # Limite : même problème d'extrapolation que Random Forest.
        try:
            xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                               max_depth=3, random_state=42, verbosity=0)
            xgb.fit(x_train, y_train)
            preds_xgb = xgb.predict(x_test)
            erreurs["xgboost"]["y_true"].extend(y_test)
            erreurs["xgboost"]["y_pred"].extend(preds_xgb)
        except Exception:
            pass

    # ── Calcul des métriques ──────────────────────────────────────────────────
    resultats = {}
    for modele, data in erreurs.items():
        if not data["y_true"]:
            resultats[modele] = {"rmse": None, "mae": None, "mape": None}
            continue

        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else None

        resultats[modele] = {
            "rmse": round(rmse, 4),
            "mae":  round(mae, 4),
            "mape": round(mape, 2) if mape is not None else None,
        }

    return resultats


# ==============================================================================
# BOUCLE SUR TOUTES LES VARIABLES
# ==============================================================================

tous_resultats = []

for nom_var, col in VARIABLES.items():
    if col not in df.columns:
        print(f"\n  [!] '{col}' absente, ignoree")
        continue

    serie = df[["annee", col]].dropna()
    if len(serie) < 30:
        print(f"\n  [!] {nom_var} : seulement {len(serie)} points, ignore")
        continue

    annees  = serie["annee"].values.astype(float)
    valeurs = serie[col].values.astype(float)

    print(f"\n  -> {nom_var} ({len(serie)} points : "
          f"{int(serie['annee'].min())}-{int(serie['annee'].max())})")

    with mlflow.start_run(run_name=f"comparaison_{nom_var}"):
        mlflow.log_param("variable", col)
        mlflow.log_param("n_observations", len(serie))

        resultats = walk_forward_validation(annees, valeurs)

        for modele, metriques in resultats.items():
            rmse_str = f"{metriques['rmse']:.4f}" if metriques["rmse"] is not None else "N/A"
            mae_str  = f"{metriques['mae']:.4f}"  if metriques["mae"]  is not None else "N/A"
            mape_str = f"{metriques['mape']:.2f}%" if metriques["mape"] is not None else "N/A"
            print(f"     {modele:<25} RMSE={rmse_str}  MAE={mae_str}  MAPE={mape_str}")

            mlflow_met = {f"{modele}_{k}": v for k, v in metriques.items() if v is not None}
            if mlflow_met:
                mlflow.log_metrics(mlflow_met)

            tous_resultats.append({
                "variable": nom_var,
                "modele":   modele,
                **metriques
            })


# ==============================================================================
# TABLEAU COMPARATIF FINAL
# ==============================================================================

df_comparaison = pd.DataFrame(tous_resultats)
df_comparaison.to_csv("data/resultats/comparaison_modeles.csv", index=False)

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF — RMSE (plus bas = meilleur)")
print("=" * 70)
pivot_rmse = df_comparaison.pivot(index="variable", columns="modele", values="rmse")
print(pivot_rmse.to_string())

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF — MAPE % (plus bas = meilleur)")
print("=" * 70)
pivot_mape = df_comparaison.pivot(index="variable", columns="modele", values="mape")
print(pivot_mape.to_string())


# ==============================================================================
# GRAPHIQUE COMPARATIF
# ==============================================================================

df_plot       = df_comparaison.dropna(subset=["rmse"])
variables_dispo = df_plot["variable"].unique()
modeles_dispo   = df_plot["modele"].unique()
n_modeles     = len(modeles_dispo)
x     = np.arange(len(variables_dispo))
width = 0.15

COULEURS = {
    "regression_lineaire": "#60a5fa",
    "arima":               "#fb923c",
    "prophet":             "#4ade80",
    "random_forest":       "#c084fc",
    "xgboost":             "#f472b6",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, metric in enumerate(["rmse", "mae"]):
    ax = axes[ax_idx]
    for i, modele in enumerate(modeles_dispo):
        vals = [df_plot[(df_plot["variable"] == v) & (df_plot["modele"] == modele)][metric].values
                for v in variables_dispo]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        offset = (i - n_modeles / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=modele,
               color=COULEURS.get(modele, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(variables_dispo, rotation=15)
    ax.set_title(f"{metric.upper()} par modèle et par variable\n(plus bas = meilleur)")
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Comparaison des 5 modèles prédictifs — Hackathon #26", fontsize=13)
plt.tight_layout()
plt.savefig("data/resultats/comparaison_modeles.png", dpi=150)
plt.close()

print("\nGraphique : data/resultats/comparaison_modeles.png")
print("CSV       : data/resultats/comparaison_modeles.csv")
print("\n" + "=" * 70)
print("Meilleur modele par variable (RMSE) :")
for var in df_plot["variable"].unique():
    df_v = df_plot[df_plot["variable"] == var].dropna(subset=["rmse"])
    if not df_v.empty:
        meilleur = df_v.loc[df_v["rmse"].idxmin(), "modele"]
        rmse_val = df_v["rmse"].min()
        print(f"  {var:<20} -> {meilleur}  (RMSE={rmse_val:.4f})")
print("=" * 70)
