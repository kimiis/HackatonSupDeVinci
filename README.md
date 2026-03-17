# Hackathon #26 — Dashboard Climatique France

Analyse et visualisation de l'évolution du climat en France de **1900 à 2026**, avec projections jusqu'en 2100.
Données officielles exclusivement : Météo France, NOAA, CITEPA, PSMSL, INSEE, EM-DAT.

---

## Lancer le projet

### 1. Installer les dépendances

```bash
pip install pandas requests openpyxl prophet matplotlib mlflow streamlit plotly folium streamlit-folium
```

### 2. Générer le dataset

```bash
python pipeline.py
```

Télécharge, nettoie et fusionne toutes les sources → produit `data/clean/dataset_final.csv`.

### 3. Entraîner les modèles Prophet

```bash
python modele_prophet.py
```

Entraîne un modèle par indicateur, génère les projections → produit `data/resultats/forecast_*.csv`.

### 4. Lancer le dashboard

```bash
python -m streamlit run app.py
```

### 5. Ouvrir le rapport analytique

```bash
jupyter notebook rapport_analytique.ipynb
```

---

## Structure du projet

```
HackatonSupDeVinci/
│
├── pipeline.py                  # Collecte et transformation des données
├── modele_prophet.py            # Modélisation et projections (Prophet + MLflow)
├── app.py                       # Dashboard Streamlit interactif
├── rapport_analytique.ipynb     # Analyse exploratoire et interprétation
│
├── data/
│   ├── raw/                     # Fichiers sources bruts
│   │   ├── co2_secteur/         # CITEPA Secten 2025 — GES par secteur (8 fichiers .xlsx)
│   │   ├── id_port_fr.xlsx      # Liste des ports français avec ID PSMSL
│   │   ├── vendanges_france_1900_2026_estimations.xlsx
│   │   └── public_emdat_*_cout_eco_catastrophe.xlsx
│   │
│   ├── clean/                   # Fichiers nettoyés produits par pipeline.py
│   │   ├── dataset_final.csv    # Dataset principal (entrée de modele_prophet.py)
│   │   ├── temperatures.csv
│   │   ├── co2.csv
│   │   ├── ges.csv
│   │   ├── niveau_mer.csv
│   │   ├── empreinte_carbone.csv
│   │   ├── vendanges.csv
│   │   └── cout_eco_catastrophe.csv
│   │
│   └── resultats/               # Projections produites par modele_prophet.py
│       ├── forecast_temperature.csv
│       ├── forecast_co2.csv
│       └── ...
│
└── mlruns/                      # Logs MLflow (expériences, métriques, artefacts)
```

---

## Sources de données

| Indicateur | Source | Période | Détails dans le code |
|---|---|---|---|
| Température moyenne (°C) | [Météo France](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles) | 1900–2026 | `pipeline.py` → `charger_temperatures()` |
| Jours chauds ≥ 30°C / Jours de gel | Météo France MENSQ — colonnes NBJTX30/NBJTX0 | 1950–2026 | `pipeline.py` → `charger_jours_extremes()` |
| CO₂ atmosphérique (ppm) | [NOAA Mauna Loa](https://gml.noaa.gov/ccgg/trends/) | 1958–2026 | `pipeline.py` → `charger_co2()` |
| Émissions GES par secteur (Mt CO₂eq) | [CITEPA Secten 2025](https://www.citepa.org/fr/secten/) | 1960–2023 | `pipeline.py` → `charger_ges()` |
| Niveau de la mer (mm depuis 1900) | [PSMSL](https://psmsl.org) — 10 ports français | 1900–2025 | `pipeline.py` → `charger_niveau_mer()` |
| Empreinte carbone individuelle (t CO₂/hab) | [INSEE ip2077](https://www.insee.fr/fr/statistiques/8654458) | 1990–2024 | `pipeline.py` → `charger_empreinte_carbone()` |
| Dates de vendanges (jour de l'année) | Estimations régionales France | 1900–2026 | `pipeline.py` → `charger_vendanges()` |
| Coût des catastrophes naturelles (Mrd USD) | [EM-DAT](https://www.emdat.be) | 1950–2023 | `pipeline.py` → `charger_cout_eco_catastrophe()` |

---

## Dataset final — colonnes de `dataset_final.csv`

| Colonne | Description | Unité |
|---|---|---|
| `annee` | Année | — |
| `temp_moy_france` | Température moyenne annuelle (7 départements) | °C |
| `jours_chauds_30` | Jours où TX ≥ 30°C (moyenne nationale) | jours/an |
| `jours_gel` | Jours où TX ≤ 0°C (moyenne nationale) | jours/an |
| `co2_ppm` | Concentration CO₂ atmosphérique | ppm |
| `industrie_energie` | Émissions GES — Industrie énergie | Mt CO₂eq |
| `industrie_manufacturiere_construction` | Émissions GES — Industrie manufacturière | Mt CO₂eq |
| `traitement_centralise_dechets` | Émissions GES — Déchets | Mt CO₂eq |
| `batiments_residentiel_tertiaire` | Émissions GES — Bâtiments | Mt CO₂eq |
| `agriculture` | Émissions GES — Agriculture | Mt CO₂eq |
| `transports` | Émissions GES — Transports | Mt CO₂eq |
| `emissions_naturelles` | Émissions GES — Émissions naturelles | Mt CO₂eq |
| `emnr` | Émissions GES — EMNR | Mt CO₂eq |
| `ges_total_MtCO2eq` | Total GES tous secteurs | Mt CO₂eq |
| `niveau_mer_brest` … | Hausse niveau de la mer par port depuis 1900 | mm |
| `niveau_mer_mm` | Hausse moyenne nationale depuis 1900 | mm |
| `empreinte_tCO2_hab` | Empreinte carbone individuelle | t CO₂/hab |
| `jour_vendanges` | Jour de début des vendanges (DOY moyen) | jour de l'année |
| `dommages_Mrd_USD` | Coût total des catastrophes naturelles | Mrd USD |

---

## Choix techniques

### Pipeline de données (`pipeline.py`)
- **LEFT JOIN** sur l'année : toutes les années de température sont conservées, les sources plus courtes génèrent des NaN (pas d'interpolation — on ne fabrique pas de données).
- **7 départements** Météo France : couvrent les grandes zones climatiques (Méditerranée, Atlantique, Nord, Est, Bassin parisien) sans télécharger les ~100 fichiers nationaux.
- **Niveau de la mer** : centrage sur 1900 (valeur 1900 = 0 mm) pour afficher directement la hausse totale depuis 1900. Nécessaire car chaque marégraphe a un zéro physique local différent.

### Modélisation (`modele_prophet.py`)
- **Prophet (Meta)** : adapté aux séries temporelles avec tendance et saisonnalité annuelle.
- **3 scénarios** calés sur les trajectoires GIEC : optimiste (+1.4°C en 2100), intermédiaire (+2.7°C), pessimiste (+4.4°C).
- **MLflow** : traçabilité des expériences, métriques MAE/RMSE, artefacts (modèles, graphiques).

### Dashboard (`app.py`)
- **Streamlit + Plotly** : visualisations interactives.
- **Folium** : carte géographique des ports avec hausse du niveau de la mer.
- Seuils d'alerte basés sur l'**Accord de Paris** (+1.5°C / +2°C) et les recommandations **GIEC**.

---

## Rapport analytique

Le notebook [`rapport_analytique.ipynb`](rapport_analytique.ipynb) contient :
- Analyse exploratoire des données (distributions, corrélations)
- Interprétation des tendances observées
- Validation des projections Prophet
- Conclusions et recommandations

---

## Références

- [Accord de Paris — UNFCCC](https://unfccc.int/process-and-meetings/the-paris-agreement)
- [Rapports du GIEC (AR6)](https://www.ipcc.ch/assessment-report/ar6/)
- [ADEME — Agir pour la transition](https://agirpourlatransition.ademe.fr)