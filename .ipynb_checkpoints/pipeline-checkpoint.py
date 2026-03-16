# ==============================================================================
# HACKATHON #26 - pipeline.py
#
# Objectif : collecter, nettoyer et fusionner toutes les sources de données
# climatiques en un seul fichier CSV prêt pour la modélisation Prophet.
#
# Ce fichier doit être exécuté EN PREMIER. Il produit :
#   data/clean/dataset_final.csv  <- fichier d'entrée de modele_prophet.py
#
# INSTALLATION (dans le terminal IntelliJ) :
#   pip install pandas requests openpyxl
# ==============================================================================

import pandas as pd
import requests
import os

# On s'assure que les dossiers de travail existent avant tout traitement.
# exist_ok=True évite une erreur si le dossier existe déjà.
# data/raw   : fichiers téléchargés tels quels depuis les sources officielles
# data/clean : fichiers après nettoyage et transformation
os.makedirs("data/raw",   exist_ok=True)
os.makedirs("data/clean", exist_ok=True)


# ==============================================================================
# SOURCE 1 - Météo France (meteo.data.gouv.fr)
#
# DONNÉES BRUTES :
#   Format : fichiers CSV compressés (.csv.gz) au format MENSQ (Mensuel Quotidien)
#   Contenu : une ligne par mois par station météo, avec des dizaines de colonnes
#             mesurant différentes variables climatiques.
#   Colonne clé : TM = température moyenne mensuelle en °C
#   Colonne date : AAAAMM = année+mois concaténés (ex : 202401 = janvier 2024)
#   Granularité : mensuelle, par station (identifiée par NUM_POSTE)
#
# TRANSFORMATION :
#   1. On télécharge les fichiers pour 7 départements représentatifs des grandes
#      zones climatiques de France (Méditerranée, Atlantique, Nord, Est, etc.)
#   2. On extrait l'année depuis AAAAMM en prenant les 4 premiers caractères
#   3. On fait la moyenne de TM sur TOUTES les stations et TOUS les mois d'une
#      même année → 1 valeur annuelle nationale représentative
#
# POURQUOI CES CHOIX :
#   - 7 départements au lieu de la France entière : évite de télécharger ~100 fichiers
#     tout en couvrant les différents régimes climatiques
#   - Moyenne sur les stations : chaque station a ses propres conditions locales,
#     la moyenne atténue ces effets et donne une tendance nationale
#   - On conserve df_all (données brutes) pour charger_jours_extremes() qui a
#     besoin des mêmes fichiers sans les retélécharger
# ==============================================================================

def charger_temperatures():
    print("Chargement temperatures Meteo France...")

    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/"

    # 7 départements choisis pour couvrir les grandes zones climatiques françaises :
    # 13=Bouches-du-Rhône (Méditerranée), 33=Gironde (Atlantique Sud),
    # 44=Loire-Atlantique (Atlantique Nord), 59=Nord (Nord), 67=Bas-Rhin (Est/Continental),
    # 69=Rhône (Centre/Alpin), 75=Paris (Bassin Parisien)
    departements = ["13", "33", "44", "59", "67", "69", "75"]

    dfs = []
    for dept in departements:
        # Météo France découpe ses archives en plusieurs fichiers par période.
        # On essaie de télécharger chaque période disponible pour ce département.
        # Le try/except permet de continuer si un fichier n'existe pas encore
        # (ex : le fichier 2025-2026 n'est pas toujours publié en temps réel).
        for period_url, label in [
            (f"{base_url}MENSQ_{dept}_previous-1950-2024.csv.gz", "1950-2024"),
            (f"{base_url}MENSQ_{dept}_latest-2025-2026.csv.gz",   "2025-2026"),
        ]:
            try:
                # sep=";" : les CSV Météo France utilisent le point-virgule comme séparateur
                # compression="gzip" : décompression automatique du .gz par pandas
                # encoding="latin-1" : encodage des fichiers français (caractères accentués)
                df = pd.read_csv(period_url, sep=";", compression="gzip", encoding="latin-1")
                dfs.append(df)
                print(f"   ? Departement {dept} ({label})")
            except Exception:
                pass  # fichier absent ou pas encore publié, on continue

        # Avant 1950, l'année de début des archives varie selon le département
        # (certains ont des données depuis 1851, d'autres depuis 1929 seulement).
        # On essaie les dates dans l'ordre jusqu'à trouver le fichier qui existe.
        for debut in ["1851", "1852", "1860", "1864", "1870", "1875", "1880", "1900", "1920", "1929"]:
            url_hist = f"{base_url}MENSQ_{dept}_{debut}-1949.csv.gz"
            try:
                df_hist = pd.read_csv(url_hist, sep=";", compression="gzip", encoding="latin-1")
                dfs.append(df_hist)
                print(f"   ? Departement {dept} (historique depuis {debut})")
                break  # on a trouvé le bon fichier, inutile d'essayer les suivants
            except Exception:
                continue  # ce fichier n'existe pas, on essaie la date suivante

    if not dfs:
        raise RuntimeError("Impossible de charger les donnees de temperature Meteo France")

    # On rassemble tous les DataFrames téléchargés en un seul.
    # ignore_index=True réindexe de 0 à N pour éviter les doublons d'index.
    df_all = pd.concat(dfs, ignore_index=True)

    # errors="coerce" convertit en NaN les valeurs non numériques (ex : "mq" = manquant
    # dans les fichiers Météo France). Sans ça, la colonne resterait en type object (str).
    df_all["TM"] = pd.to_numeric(df_all["TM"], errors="coerce")

    # AAAAMM est un entier de type 202401. On le convertit en str pour découper
    # les 4 premiers caractères (l'année), puis on le repasse en int pour les calculs.
    df_all["annee"] = df_all["AAAAMM"].astype(str).str[:4].astype(int)

    # On filtre sur notre fenêtre temporelle d'intérêt.
    # On supprime aussi les lignes sans température mesurée (TM = NaN) car une
    # moyenne avec trop de NaN serait peu représentative.
    df_all = df_all[(df_all["annee"] >= 1900) & (df_all["annee"] <= 2026)]
    df_all = df_all.dropna(subset=["TM"])

    # Agrégation : on calcule la température moyenne annuelle nationale en faisant
    # la moyenne de TM sur toutes les stations et tous les mois de chaque année.
    # groupby("annee")["TM"].mean() → une valeur par année, moyennée sur toutes
    # les lignes de cette année (toutes stations confondues, tous mois confondus).
    df_annuel = df_all.groupby("annee")["TM"].mean().reset_index()
    df_annuel.columns = ["annee", "temp_moy_france"]

    # reset_index() après set_index() ne fait rien de visible ici mais assure
    # que l'index est un entier standard (utile pour les merges ultérieurs).
    df_annuel = df_annuel.set_index("annee").reset_index()

    print(f"Temperatures chargees : {len(df_annuel)} annees")
    df_annuel.to_csv("data/clean/temperatures.csv", index=False)

    # On retourne aussi df_all (données brutes mensuelles par station) car
    # charger_jours_extremes() en a besoin. Cela évite de retélécharger les fichiers.
    return df_annuel, df_all


# ==============================================================================
# SOURCE  - Météo France : jours extrêmes
#
# DONNÉES BRUTES :
#   Même fichiers MENSQ que pour les températures (df_all_mensq passé en paramètre).
#   Colonnes clés :
#     NBJTX30 = nombre de jours dans le mois où TX (température maximale) >= 30°C
#     NBJTX0  = nombre de jours dans le mois où TX <= 0°C (jours de gel)
#   Ces colonnes existent seulement dans les fichiers depuis ~1950.
#
# TRANSFORMATION :
#   Pour chaque station et chaque année :
#     1. On somme les valeurs mensuelles → total annuel par station
#        (ex : 3 jours chauds en juin + 5 en juillet + 2 en août = 10 pour l'année)
#     2. On fait la moyenne des totaux annuels entre toutes les stations
#        → 1 valeur nationale représentative par an
#
# POURQUOI CES DEUX ÉTAPES DANS CET ORDRE :
#   Si on faisait la moyenne des mois avant de sommer les années, on biaiserait
#   le résultat (une station avec peu de données fausserait la moyenne mensuelle).
#   La logique est : compter d'abord par station, puis comparer les stations.
# ==============================================================================

def charger_jours_extremes(df_all_mensq):
    print("Extraction jours extremes (TX>=30C / TX<=0C)...")

    df = df_all_mensq.copy()
    df["annee"] = df["AAAAMM"].astype(str).str[:4].astype(int)

    # Les colonnes NBJTX30 et NBJTX0 ne sont disponibles que depuis ~1950 dans
    # les fichiers MENSQ. On filtre pour ne garder que la période couverte.
    df = df[(df["annee"] >= 1950) & (df["annee"] <= 2026)]

    # On vérifie quelles colonnes sont présentes dans les fichiers téléchargés
    # (elles peuvent varier selon les départements ou les périodes).
    cols_dispo = []

    if "NBJTX30" in df.columns:
        df["NBJTX30"] = pd.to_numeric(df["NBJTX30"], errors="coerce")
        cols_dispo.append("NBJTX30")
    else:
        print("   Colonne NBJTX30 absente dans les fichiers MENSQ")

    if "NBJTX0" in df.columns:
        df["NBJTX0"] = pd.to_numeric(df["NBJTX0"], errors="coerce")
        cols_dispo.append("NBJTX0")
    else:
        print("   Colonne NBJTX0 absente dans les fichiers MENSQ")

    if not cols_dispo:
        print("Aucun indicateur de jours extremes disponible")
        return None

    # Étape 1 : groupby(["NUM_POSTE", "annee"]).sum()
    #   → pour chaque station (NUM_POSTE) et chaque année, on additionne
    #     les valeurs mensuelles pour obtenir le total annuel de jours extrêmes.
    #   Exemple : station Paris, 2022 → sum(0, 0, 0, 0, 0, 3, 8, 5, 0, 0, 0, 0) = 16 jours chauds
    #
    # Étape 2 : groupby("annee").mean()
    #   → on fait la moyenne de ces totaux entre toutes les stations pour
    #     obtenir une valeur nationale : en 2022, en moyenne 16 jours chauds par station.
    df_annuel = (
        df.groupby(["NUM_POSTE", "annee"])[cols_dispo].sum()   # étape 1 : total/station/an
          .reset_index()
          .groupby("annee")[cols_dispo].mean()                  # étape 2 : moyenne nationale
          .reset_index()
    )

    # Renommage pour des noms de colonnes explicites dans le dataset final.
    df_annuel = df_annuel.rename(columns={
        "NBJTX30": "jours_chauds_30",
        "NBJTX0":  "jours_gel"
    })

    print(f"Jours extremes charges : {len(df_annuel)} annees "
          f"({', '.join(df_annuel.columns[1:].tolist())})")
    df_annuel.to_csv("data/clean/jours_extremes.csv", index=False)
    return df_annuel


# ==============================================================================
# SOURCE 2 - NOAA Mauna Loa (gml.noaa.gov)
#
# DONNÉES BRUTES :
#   Format : CSV texte avec des lignes de commentaires commençant par #
#   Contenu : concentrations atmosphériques annuelles de CO₂ en ppm
#             (parties par million) mesurées à la station Mauna Loa (Hawaï)
#   Colonnes : année | co2_ppm | incertitude
#   Période : 1958 à aujourd'hui (premières mesures directes en continu au monde)
#
# TRANSFORMATION :
#   - On ignore les lignes de commentaires (comment="#")
#   - On garde uniquement les colonnes année et co2_ppm
#   - On force le typage numérique et on filtre jusqu'en 2026
#
# ==============================================================================

def charger_co2():
    print("Chargement CO2 NOAA...")

    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"

    # comment="#" indique à pandas d'ignorer toutes les lignes commençant par #
    # (métadonnées, informations de licence, description des colonnes)
    # header=None car la ligne d'en-tête est absente après filtrage des commentaires
    df = pd.read_csv(url, comment="#", header=None)
    df.columns = ["annee", "co2_ppm", "incertitude"]

    # On ne conserve que les deux colonnes utiles pour la modélisation.
    # L'incertitude de mesure n'est pas utilisée dans notre modèle Prophet.
    df = df[["annee", "co2_ppm"]]

    # Données disponibles uniquement depuis 1958 (premières mesures directes).
    # On ne génère PAS de valeurs pour les années antérieures afin de ne pas
    # introduire de données fabriquées dans le dataset.
    df_complet = df.copy()

    # Int64 (avec majuscule) est le type entier pandas qui supporte les NaN,
    # contrairement au int64 numpy standard qui ne le supporte pas.
    df_complet["annee"]   = pd.to_numeric(df_complet["annee"],   errors="coerce").astype("Int64")
    df_complet["co2_ppm"] = pd.to_numeric(df_complet["co2_ppm"], errors="coerce")
    df_complet = df_complet[df_complet["annee"] <= 2026]

    print(f"CO2 charge : {len(df_complet)} annees - "
          f"derniere valeur : {df_complet.iloc[-1]['co2_ppm']:.1f} ppm")
    df_complet.to_csv("data/clean/co2.csv", index=False)
    return df_complet


# ==============================================================================
# SOURCE 3 - CITEPA Secten 2025 - Émissions GES par secteur
#
# DONNÉES BRUTES :
#   Format : fichiers Excel (.xlsx) un par secteur économique
#            (Transports, Énergie, Agriculture, Industrie, Résidentiel, etc.)
#   Structure de chaque fichier :
#     - Un onglet dont le nom contient "mission" (ex : "Transports-Emissions")
#     - Ligne 9 (index=9) : en-têtes avec les années en colonnes à partir de col 6
#     - Colonnes 0-5 : métadonnées (périmètre, substance, secteur, unité)
#     - Colonne 4 : nom du secteur
#     - Colonne 5 : unité de mesure ("Mt CO2e" pour les totaux)
#     - Colonnes 6+ : valeurs annuelles d'émissions en Mt CO₂ équivalent
#   Période : environ 1960 à 2023
#
# TRANSFORMATION :
#   _extraire_total_secteur() : lit un fichier et retourne la ligne "total secteur"
#     1. Détection de l'onglet Emissions
#     2. Lecture des années disponibles (ligne 9, col 6+) → dictionnaire {année: colonne}
#     3. Sélection de la ligne dont l'unité est "Mt CO2e" et qui n'est pas
#        "Total national" ni "Autres secteurs" (on veut le total du secteur lui-même)
#     4. Extraction des valeurs → pandas Series indexée par année
#
#   charger_ges() : orchestre la lecture de tous les fichiers secteur
#     1. Lecture de chaque fichier → une Series par secteur
#     2. Assemblage en DataFrame : une colonne par secteur
#     3. Calcul du total national = somme horizontale de tous les secteurs
#
# POURQUOI RECONSTITUER LE TOTAL AU LIEU DE LE LIRE DIRECTEMENT :
#   Chaque fichier Excel ne contient qu'un secteur. Il n'existe pas de fichier
#   "total national" : on doit additionner les secteurs pour le reconstruire.
# ==============================================================================

def _extraire_total_secteur(chemin):
    """
    Lit un fichier Excel CITEPA Secten et retourne le total d'émissions du secteur.

    Retourne :
        nom (str)          : nom du secteur (ex : "Transports")
        serie (pd.Series)  : émissions annuelles en Mt CO₂eq, indexées par année
        (None, None)       : si le fichier ne correspond pas au format attendu
    """
    xl = pd.ExcelFile(chemin)

    # On cherche l'onglet dont le nom contient "mission" (pour "Emissions")
    # car le nom exact peut varier selon le secteur (ex : "Transports-Emissions")
    sheet = next((s for s in xl.sheet_names if "mission" in s), None)
    if sheet is None:
        return None, None

    # header=None : on lit le fichier sans interpréter de ligne comme en-tête
    # car la structure est irrégulière (métadonnées avant les données)
    df = pd.read_excel(chemin, sheet_name=sheet, header=None)

    # La ligne 9 (index=9) contient les années en colonnes à partir de la colonne 6.
    # On construit un dictionnaire {année: indice_colonne} pour récupérer
    # facilement la valeur d'une année donnée.
    year_to_col = {}
    for c in range(6, df.shape[1]):
        v = pd.to_numeric(df.iloc[9, c], errors="coerce")
        if pd.notna(v) and 1900 <= v <= 2026:
            year_to_col[int(v)] = c

    if not year_to_col:
        return None, None

    # Colonne 4 : nom du secteur (ex : "Transports", "Énergie")
    # Colonne 5 : unité de mesure ("Mt CO2e", "kt", "%", etc.)
    col_secteur = df.iloc[:, 4].astype(str).str.strip()
    col_unite   = df.iloc[:, 5].astype(str).str.strip()

    # On cherche la ligne dont :
    # - l'unité est exactement "Mt CO2e" → c'est le total du secteur (pas les sous-catégories)
    # - le secteur n'est pas "Total national" ni "Autres secteurs" ni une ligne en %
    #   → on veut le total propre au secteur, pas une agrégation ou un résidu
    exclure = r"total national|autres secteurs|%|nan"
    mask = (
        (col_unite == "Mt CO2e") &
        ~col_secteur.str.lower().str.contains(exclure, na=True, regex=True)
    )
    candidats = df[mask]
    if candidats.empty:
        return None, None

    # Si plusieurs lignes correspondent (cas rares), on prend la dernière
    # qui est généralement la ligne de total la plus agrégée
    idx_row = candidats.index[-1]
    nom = col_secteur.iloc[idx_row]

    # Extraction des valeurs numériques pour chaque année identifiée
    values = {}
    for year, c in year_to_col.items():
        val = pd.to_numeric(df.iloc[idx_row, c], errors="coerce")
        if pd.notna(val):
            values[year] = val

    if not values:
        return None, None

    # pd.Series avec les années comme index → format pratique pour créer
    # ensuite un DataFrame multi-colonnes dans charger_ges()
    serie = pd.Series(values, name=nom)
    return nom, serie


def charger_ges():
    print("... Chargement GES CITEPA Secten (par secteur)...")

    dossier = "data/raw/co2_secteur"
    if not os.path.exists(dossier):
        print("[!]  Dossier data/raw/co2_secteur introuvable")
        return None

    fichiers = [f for f in os.listdir(dossier) if f.endswith(".xlsx")]
    if not fichiers:
        print("[!]  Aucun fichier .xlsx dans data/raw/co2_secteur")
        return None

    # On lit chaque fichier secteur et on stocke la série dans un dictionnaire
    # {nom_colonne: serie_annuelle}
    series = {}
    for f in sorted(fichiers):
        chemin = os.path.join(dossier, f)
        nom, serie = _extraire_total_secteur(chemin)
        if serie is not None:
            # On génère un nom de colonne court à partir du nom de fichier
            # ex : "GES_Transports_secten2025.xlsx" → "transports"
            col = f.split("_")[1].replace("-", "_").lower()
            series[col] = serie
            print(f"   ? {col} ({serie.dropna().shape[0]} annees)")

    if not series:
        print("[!]  Aucun secteur extrait")
        return None

    # pd.DataFrame(dict_de_series) : pandas aligne automatiquement les series
    # sur leur index commun (les années). Les années absentes d'une série
    # reçoivent NaN → comportement correct, on ne fabrique pas de données.
    df_ges = pd.DataFrame(series)
    df_ges.index.name = "annee"
    df_ges = df_ges.reset_index()

    # Total national = somme ligne par ligne (axis=1) de tous les secteurs.
    # skipna=True : si un secteur n'a pas de valeur pour une année,
    # on additionne quand même les secteurs disponibles (plutôt que NaN total).
    cols_secteurs = [c for c in df_ges.columns if c != "annee"]
    df_ges["ges_total_MtCO2eq"] = df_ges[cols_secteurs].sum(axis=1, skipna=True)

    print(f"[OK] GES charge : {len(df_ges)} annees - "
          f"secteurs : {cols_secteurs}")
    df_ges.to_csv("data/clean/ges.csv", index=False)
    return df_ges


# ==============================================================================
# SOURCE 4 - PSMSL (Permanent Service for Mean Sea Level)
#
# DONNÉES BRUTES :
#   Format : fichier texte séparé par ";" sans en-tête, une ligne par mois
#   URL : https://psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata
#   Colonnes : date_décimale | hauteur_mm | flag1 | flag2
#     - date_décimale : ex 2024.042 = début janvier 2024
#     - hauteur_mm    : hauteur du niveau de la mer en millimètres,
#                       mesurée depuis le zéro local du marégraphe (RLR datum)
#     - flag -99999   : valeur sentinelle indiquant une donnée manquante
#   Source des ports : fichier data/raw/id_port_fr.xlsx (ID, Ville, Lat, Long)
#
# TRANSFORMATION :
#   Pour chaque port :
#     1. Filtrage des valeurs sentinelles (< -99999) et de la période 1900-2026
#     2. Extraction de l'année depuis la date décimale (partie entière)
#     3. Moyenne des mesures mensuelles → 1 valeur annuelle par port
#     4. Centrage sur la période de référence 1961-1990 :
#        hauteur_centree = hauteur_brute - moyenne(1961-1990)
#        → transforme la hauteur absolue (dépendante du zéro local)
#          en anomalie relative (comparable entre ports)
#   Puis : moyenne des anomalies de tous les ports → 1 valeur nationale par an
#
# POURQUOI LE CENTRAGE SUR 1961-1990 :
#   Chaque marégraphe a son propre zéro physique (un repère au fond de l'eau).
#   Sans centrage, Brest à 6800mm et Marseille à 4200mm ne peuvent pas être
#   moyennés (ils ne mesurent pas la même chose). Après centrage, les deux
#   ports indiquent "combien de mm par rapport à leur niveau habituel sur
#   1961-1990", ce qui est comparable et peut être moyenné.
#   La période 1961-1990 est la référence climatique internationale standard (OMM).
# ==============================================================================

def charger_niveau_mer():
    print("Chargement niveau de la mer (PSMSL - ports francais)...")

    base_url = "https://psmsl.org/data/obtaining/rlr.monthly.data/{}.rlrdata"
    ports_file = "data/raw/id_port_fr.xlsx"

    if not os.path.exists(ports_file):
        print("[!] Fichier id_port_fr.xlsx introuvable - fallback sur Brest seul")
        ports = [{"ID": 1, "Ville": "BREST"}]
    else:
        df_ports = pd.read_excel(ports_file)
        ports = df_ports[["ID", "Ville"]].to_dict("records")

    series = []
    for port in ports:
        pid, ville = port["ID"], port["Ville"]
        try:
            # Lecture du fichier texte brut sans en-tête.
            # On nomme les colonnes manuellement selon la documentation PSMSL.
            df = pd.read_csv(base_url.format(pid), sep=";", header=None,
                             names=["date_decimal", "hauteur_mm", "flag1", "flag2"])

            df["hauteur_mm"] = pd.to_numeric(df["hauteur_mm"], errors="coerce")

            # -99999 est la valeur sentinelle PSMSL pour "donnée manquante".
            # On la filtre avant toute moyenne pour ne pas biaiser les calculs.
            df = df[df["hauteur_mm"] > -99999]

            # La date décimale 2024.042 donne l'année en prenant la partie entière.
            # int() tronque (ne pas utiliser round qui donnerait 2024.5 → 2025).
            df["annee"] = df["date_decimal"].astype(int)
            df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

            if df.empty:
                continue

            # Étape 1 : Moyenne mensuelle → annuelle pour ce port
            df_a = df.groupby("annee")["hauteur_mm"].mean()

            # Étape 2 : Centrage sur 1961-1990
            # On vérifie qu'on a au minimum 5 années de référence (seuil minimum
            # pour que la moyenne de référence soit statistiquement valide).
            ref_mask = (df_a.index >= 1961) & (df_a.index <= 1990)
            if ref_mask.sum() >= 5:
                # Soustraction de la moyenne de référence → anomalie en mm
                df_a = df_a - df_a[ref_mask].mean()
                series.append(df_a.rename(ville))
                print(f"   OK {ville} (ID={pid}) : {int(df_a.index.min())}-{int(df_a.index.max())}")
            else:
                print(f"   [!] {ville} (ID={pid}) : pas assez de donnees ref 1961-1990, ignore")
        except Exception as e:
            print(f"   [!] {ville} (ID={pid}) : {e}")

    if not series:
        print("[!] Aucun port charge")
        return None

    # pd.concat(series, axis=1) : on aligne les séries sur l'axe des années.
    # Les années où un port n'a pas de données reçoivent NaN pour ce port.
    df_all = pd.concat(series, axis=1)
    df_all.index.name = "annee"
    df_all = df_all.reset_index()

    # On préfixe les noms de colonnes de villes pour éviter les conflits
    # lors de la fusion finale (ex : "BREST" → "niveau_mer_brest")
    ville_cols = {v: f"niveau_mer_{v.lower().replace(' ', '_').replace('-', '_')}"
                  for v in df_all.columns if v != "annee"}
    df_all = df_all.rename(columns=ville_cols)

    # Moyenne nationale : mean(axis=1) calcule la moyenne horizontale,
    # c'est-à-dire la moyenne entre tous les ports pour chaque année.
    # C'est la colonne principale utilisée dans la modélisation.
    cols_villes = [c for c in df_all.columns if c != "annee"]
    df_all["niveau_mer_mm"] = df_all[cols_villes].mean(axis=1)

    print(f"[OK] Niveau de la mer : {len(df_all)} annees, "
          f"{len(series)} ports (anomalie/ref. 1961-1990)")
    print(f"   Colonnes : {cols_villes + ['niveau_mer_mm']}")
    df_all.to_csv("data/clean/niveau_mer.csv", index=False)
    return df_all


# ==============================================================================
# SOURCE 5 - INSEE ip2077 - Empreinte carbone individuelle
#
# DONNÉES BRUTES :
#   Format : fichier Excel téléchargé dynamiquement via l'API INSEE
#   URL : https://insee.fr/fr/statistiques/fichier/8654458/ip2077.xlsx
#   Onglet utilisé : "Figure 3" (L'empreinte carbone et ses composantes 1990-2024)
#   Structure : ligne 3 à 37 = données 1990-2024
#     - Colonne 0 : année
#     - Colonne 4 : "Empreinte totale" en Mt CO₂eq (émissions nationales totales
#                  incluant les importations, contrairement aux émissions territoriales)
#
# TRANSFORMATION :
#   1. Lecture du fichier Excel en mémoire (io.BytesIO évite de sauvegarder en local)
#   2. Extraction des lignes/colonnes utiles par position (iloc)
#   3. Estimation de la population française par interpolation linéaire
#      sur 5 points de référence INSEE connus (1990, 2000, 2010, 2020, 2024)
#   4. Calcul de l'empreinte individuelle :
#      empreinte_tCO2_hab = empreinte_Mt / population_M
#      (Mt et millions s'annulent → résultat directement en tCO₂/habitant)
#
# POURQUOI DIVISER PAR LA POPULATION :
#   L'empreinte totale en Mt reflète la taille du pays. Pour comparer dans le
#   temps ou entre pays, on ramène à l'individu. C'est la métrique standard
#   utilisée dans les rapports du GIEC et par l'ADEME.
# ==============================================================================

def charger_empreinte_carbone():
    """
    Source : INSEE ip2077 - "Figure 3 : L'empreinte carbone et ses composantes de 1990 a 2024"

    Colonne utilisee : "Empreinte totale" (col 4) en Mt CO2eq
    Calcul : empreinte_tCO2_hab = Empreinte totale (Mt) / population (millions)
             → t CO2eq/habitant (les millions s'annulent)
    """
    print("Chargement empreinte carbone individuelle (INSEE ip2077)...")

    import io

    url = "https://insee.fr/fr/statistiques/fichier/8654458/ip2077.xlsx"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()  # lève une exception si le serveur renvoie une erreur HTTP

        # io.BytesIO permet de lire le contenu binaire de la réponse HTTP
        # directement comme un fichier, sans le sauvegarder sur le disque.
        # header=None car on sélectionne les lignes manuellement avec iloc.
        df_raw = pd.read_excel(io.BytesIO(resp.content),
                               sheet_name="Figure 3", header=None)

        # iloc[3:38] : lignes 3 à 37 (index 3 inclus, 38 exclu) = données 1990-2024
        # iloc[:, [0, 4]] : on garde uniquement la colonne année (0) et empreinte totale (4)
        df_data = df_raw.iloc[3:38, [0, 4]].copy()
        df_data.columns = ["annee", "empreinte_Mt"]
        df_data["annee"]        = pd.to_numeric(df_data["annee"],        errors="coerce")
        df_data["empreinte_Mt"] = pd.to_numeric(df_data["empreinte_Mt"], errors="coerce")
        df_data = df_data.dropna()

        # Population française : on dispose de 5 points de référence INSEE.
        # On remplit les années intermédiaires par interpolation linéaire.
        # C'est acceptable car la population évolue de façon quasi-linéaire
        # sur des intervalles de 10 ans (pas de choc démographique).
        pop_ref = {1990: 58.0, 2000: 60.5, 2010: 64.6, 2020: 67.4, 2024: 68.4}
        pop_series = pd.Series(pop_ref).reindex(range(1990, 2027))
        # method="index" : interpole en tenant compte de l'écart entre les index
        # (important si les points de référence ne sont pas équidistants)
        pop_series = pop_series.interpolate(method="index")  # en millions d'habitants

        # On associe la population estimée à chaque année du dataset
        df_data["population_M"] = df_data["annee"].map(pop_series)

        # Conversion Mt → t/hab :
        # empreinte_Mt × 1 000 000 t/Mt ÷ (population_M × 1 000 000 hab/M)
        # = empreinte_Mt / population_M  (les 10^6 s'annulent)
        df_data["empreinte_tCO2_hab"] = df_data["empreinte_Mt"] / df_data["population_M"]

        df_out = df_data[["annee", "empreinte_tCO2_hab"]].copy()
        df_out["annee"] = df_out["annee"].astype(int)

        print(f" Empreinte carbone chargee : {len(df_out)} annees "
              f"({int(df_out['annee'].min())}-{int(df_out['annee'].max())}, "
              f"derniere valeur : {df_out.iloc[-1]['empreinte_tCO2_hab']:.1f} t CO2/hab)")
        df_out.to_csv("data/clean/empreinte_carbone.csv", index=False)
        return df_out

    except Exception as e:
        print(f" INSEE non disponible ({e}) - empreinte carbone ignoree")
        return None


# ==============================================================================
# SOURCE 6 - Dates de vendanges (indicateur phénologique)
#
# DONNÉES BRUTES :
#   Format : fichier Excel local
#   Chemin : data/raw/vendanges_france_1900_2026_estimations.xlsx
#   Onglet : "Vendanges_estimations"
#   Contenu : une ligne par couple (année, région viticole)
#     - Colonne année : l'année
#     - Colonne DOY (Day Of Year) : le jour de l'année du début des vendanges
#                                   (ex : 258 = 15 septembre)
#   Couverture : plusieurs régions viticoles françaises, 1900-2026
#
# TRANSFORMATION :
#   - Détection automatique des noms de colonnes (évite les problèmes d'accents)
#   - Moyenne du DOY sur toutes les régions → 1 valeur nationale par an
#
# POURQUOI C'EST UN INDICATEUR CLIMATIQUE :
#   Les vendanges débutent quand le raisin a atteint un degré de maturité
#   déterminé par l'accumulation de chaleur depuis le printemps. Un DOY plus
#   petit (vendanges plus tôt dans l'année) indique un été plus chaud.
#   C'est un indicateur phénologique historique fiable, utilisé dans de
#   nombreuses études sur le réchauffement climatique en Europe.
# ==============================================================================

def charger_vendanges():
    print("... Chargement dates de vendanges...")

    chemin = "data/raw/vendanges_france_1900_2026_estimations.xlsx"

    if not os.path.exists(chemin):
        print("[!]  Fichier vendanges introuvable :", chemin)
        return None

    df = pd.read_excel(chemin, sheet_name="Vendanges_estimations")

    # strip() supprime les espaces en début/fin de nom de colonne,
    # fréquents dans les fichiers Excel issus de saisies manuelles.
    df.columns = [c.strip() for c in df.columns]

    # Détection flexible des colonnes : le nom exact peut varier selon
    # la version du fichier (ex : "Année", "annee", "Year", "ANNEE")
    col_annee = next(c for c in df.columns if "ann" in c.lower() or "year" in c.lower())
    col_doy   = next(c for c in df.columns if "doy" in c.lower() or "jour" in c.lower() and "ann" in c.lower())

    df = df[[col_annee, col_doy]].rename(
        columns={col_annee: "annee", col_doy: "jour_vendanges"}
    )
    df["annee"]          = pd.to_numeric(df["annee"],          errors="coerce")
    df["jour_vendanges"] = pd.to_numeric(df["jour_vendanges"], errors="coerce")

    # dropna() supprime les lignes où l'une des deux colonnes est NaN
    # (lignes vides, séparateurs ou notes de bas de tableau dans l'Excel)
    df = df.dropna()
    df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

    # Moyenne des DOY sur toutes les régions → indicateur national annuel.
    # On n'agrège pas en somme car le DOY est une position dans l'année,
    # pas une quantité cumulable.
    df_annuel = df.groupby("annee")["jour_vendanges"].mean().reset_index()

    print(f" Vendanges chargees : {len(df_annuel)} annees - "
          f"derniere valeur : jour {df_annuel.iloc[-1]['jour_vendanges']:.0f} "
          f"(an {int(df_annuel.iloc[-1]['annee'])})")
    df_annuel.to_csv("data/clean/vendanges.csv", index=False)
    return df_annuel


# ==============================================================================
# SOURCE 7 - EM-DAT : Coût économique des catastrophes naturelles
#
# DONNÉES BRUTES :
#   Format : fichier Excel local (export personnalisé EM-DAT)
#   Nom dynamique : data/raw/*cout_eco_catastrophe*.xlsx
#   Onglet : "EM-DAT Data"
#   Contenu : une ligne par événement catastrophique (inondation, canicule,
#             tempête, etc.) avec son année de début et ses dommages économiques
#   Colonne clé : "Total Damage, Adjusted ('000 US$)"
#     - "Adjusted" = ajusté à l'inflation (valeur réelle, comparable dans le temps)
#     - Unité : milliers de dollars US (k$)
#
# TRANSFORMATION :
#   1. On extrait uniquement l'année et les dommages
#   2. groupby("annee").sum() : on additionne les dommages de tous les
#      événements d'une même année → coût total annuel des catastrophes
#   3. Conversion d'unité : k$ ÷ 1 000 000 = milliards USD (Mrd USD)
#      pour avoir des valeurs lisibles (ex : 17 Mrd USD au lieu de 17 000 000 k$)
#
# LIMITE :
#   EM-DAT recense surtout les catastrophes majeures documentées. Les petits
#   événements et les années anciennes sont sous-représentés → la série
#   historique avant 1990 est peu fiable.
# ==============================================================================

def charger_cout_eco_catastrophe():
    print("Chargement cout economique des catastrophes (EM-DAT)...")

    import glob

    # Le nom du fichier inclut une date de téléchargement variable
    # (ex : "public_emdat_custom_request_2024-01-15_cout_eco_catastrophe.xlsx")
    # glob avec * permet de le trouver quel que soit ce préfixe de date.
    fichiers = glob.glob("data/raw/*cout_eco_catastrophe*.xlsx")
    if not fichiers:
        print("  Fichier EM-DAT introuvable dans data/raw/")
        return None
    chemin = fichiers[0]

    df = pd.read_excel(chemin, sheet_name="EM-DAT Data")

    col_annee  = "Start Year"
    col_dommag = "Total Damage, Adjusted ('000 US$)"

    df = df[[col_annee, col_dommag]].rename(
        columns={col_annee: "annee", col_dommag: "dommages_k_usd"}
    )
    df["annee"]          = pd.to_numeric(df["annee"],          errors="coerce")
    df["dommages_k_usd"] = pd.to_numeric(df["dommages_k_usd"], errors="coerce")

    # dropna sur l'année uniquement : on garde les lignes sans dommages connus
    # (dommages NaN = non documentés, pas forcément nuls). Pour la somme annuelle,
    # sum() ignore les NaN par défaut, ce qui est le comportement voulu.
    df = df.dropna(subset=["annee"])
    df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

    # Somme des dommages par année (toutes catastrophes confondues)
    # puis conversion k$ → Mrd$ pour avoir des valeurs plus lisibles
    df_annuel = df.groupby("annee")["dommages_k_usd"].sum().reset_index()
    df_annuel["dommages_Mrd_USD"] = df_annuel["dommages_k_usd"] / 1_000_000
    df_annuel = df_annuel[["annee", "dommages_Mrd_USD"]]
    df_annuel["annee"] = df_annuel["annee"].astype(int)

    print(f" Cout catastrophes charge : {len(df_annuel)} annees - "
          f"derniere valeur : {df_annuel.iloc[-1]['dommages_Mrd_USD']:.2f} Mrd USD "
          f"(an {int(df_annuel.iloc[-1]['annee'])})")
    df_annuel.to_csv("data/clean/cout_eco_catastrophe.csv", index=False)
    return df_annuel


# ==============================================================================
# FUSION FINALE
#
# OBJECTIF :
#   Assembler les 8 sources nettoyées en un seul DataFrame prêt pour Prophet.
#   Chaque ligne = une année. Chaque colonne = un indicateur climatique.
#
# MÉTHODE :
#   On utilise les températures comme table de base (série la plus longue :
#   toutes les années pour lesquelles on a au moins une mesure de température).
#   On effectue des LEFT JOIN successifs sur la colonne "annee".
#   → Toutes les années de température sont conservées.
#   → Les colonnes des autres sources ont NaN pour les années hors de leur
#     période de couverture (ex : CO₂ aura NaN avant 1958, empreinte avant 1990).
#
# POURQUOI LEFT JOIN ET NON INNER JOIN :
#   Un INNER JOIN ne garderait que les années couvertes par TOUTES les sources,
#   soit seulement 1990-2023. On perdrait tout l'historique des températures.
#   Le LEFT JOIN permet de conserver l'historique complet tout en acceptant
#   des NaN sur les colonnes moins longues.
# ==============================================================================

def fusionner_sources(df_temp, df_co2, df_ges, df_jours_ext,
                      df_niveau_mer, df_empreinte, df_vendanges, df_catastrophes):
    print("\n Fusion de toutes les sources...")

    # Table de base : températures (série la plus longue disponible)
    df_final = df_temp.copy()

    sources = [
        (df_co2,          "CO2"),
        (df_ges,          "GES CITEPA"),
        (df_jours_ext,    "jours extremes"),
        (df_niveau_mer,   "niveau mer"),
        (df_empreinte,    "empreinte carbone"),
        (df_vendanges,    "vendanges"),
        (df_catastrophes, "cout catastrophes"),
    ]

    for df_src, nom in sources:
        if df_src is not None:
            df_src = df_src.copy()
            # On force le type numérique de la colonne annee dans chaque source
            # pour éviter des erreurs de jointure dues aux types mixtes (int vs float vs Int64)
            df_src["annee"] = pd.to_numeric(df_src["annee"], errors="coerce")
            # how="left" : on garde toutes les lignes de df_final (toutes les années
            # de température) et on ajoute les colonnes de df_src quand elles existent.
            df_final = df_final.merge(df_src, on="annee", how="left")
            print(f"   ? {nom} fusionne")

    # Sauvegarde du dataset final. C'est ce fichier que modele_prophet.py lit.
    # Les NaN sont conservés tels quels : ils représentent de vraies données manquantes
    # et non des valeurs à inventer.
    df_final.to_csv("data/clean/dataset_final.csv", index=False)

    print(f"\n Dataset final cree : {df_final.shape[0]} lignes ? "
          f"{df_final.shape[1]} colonnes")
    print(f"   Colonnes : {list(df_final.columns)}")
    print(f"   Periode  : {df_final['annee'].min()} -> {df_final['annee'].max()}")
    print(f"\n Fichier pret : data/clean/dataset_final.csv")
    print("   -> Tu peux maintenant lancer modele_prophet.py")
    return df_final


# ==============================================================================
# POINT D'ENTRÉE
# Lance ce fichier avec : python pipeline.py
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HACKATHON #26 - Pipeline de donnees climatiques")
    print("=" * 60)

    # Étape 1 : températures (retourne aussi df_all_mensq pour les jours extrêmes)
    df_temp, df_all_mensq = charger_temperatures()

    # Étape 2 : jours extrêmes (réutilise les fichiers déjà téléchargés)
    df_jours_ext          = charger_jours_extremes(df_all_mensq)

    # Étapes 3-8 : sources indépendantes
    df_co2                = charger_co2()
    df_ges                = charger_ges()
    df_niveau_mer         = charger_niveau_mer()
    df_empreinte          = charger_empreinte_carbone()
    df_vendanges          = charger_vendanges()
    df_catastrophes       = charger_cout_eco_catastrophe()

    # Étape finale : fusion de toutes les sources en un seul fichier
    df_final = fusionner_sources(
        df_temp, df_co2, df_ges, df_jours_ext,
        df_niveau_mer, df_empreinte, df_vendanges, df_catastrophes
    )

    print("\n" + "=" * 60)
    print("Pipeline termine. Apercu des donnees :")
    print("=" * 60)
    print(df_final.tail(10).to_string(index=False))