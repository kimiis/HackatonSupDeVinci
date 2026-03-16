# ==============================================================================
# HACKATHON #26 - pipeline.py
# Ce fichier charge et nettoie les donnees depuis les sources officielles
# du cahier des charges. À lancer EN PREMIER avant modele_prophet.py
# ==============================================================================
# INSTALLATION (dans le terminal IntelliJ) :
#   pip install pandas requests openpyxl
# ==============================================================================

import pandas as pd
import requests
import os

# Dossier ou seront sauvegardes tous les fichiers telecharges et nettoyes
os.makedirs("data/raw",   exist_ok=True)
os.makedirs("data/clean", exist_ok=True)


# ==============================================================================
# SOURCE 1 - Meteo France (meteo.data.gouv.fr)
# Temperatures moyennes annuelles + jours chauds (TX>=30C) + jours de gel (TX<=0C)
# Donnees mensuelles par departement, format MENSQ
# https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles
# ==============================================================================

def charger_temperatures():
    print("Chargement temperatures Meteo France...")

    base_url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/"

    # Departements representatifs couvrant les grandes zones climatiques de France
    # metropolitaine (nord, parisien, loire-atlantique, alsacien, atlantique, mediterraneen, alpin)
    departements = ["13", "33", "44", "59", "67", "69", "75"]

    dfs = []
    for dept in departements:
        # Periode 1950-2024
        for period_url, label in [
            (f"{base_url}MENSQ_{dept}_previous-1950-2024.csv.gz", "1950-2024"),
            (f"{base_url}MENSQ_{dept}_latest-2025-2026.csv.gz",   "2025-2026"),
        ]:
            try:
                df = pd.read_csv(period_url, sep=";", compression="gzip", encoding="latin-1")
                dfs.append(df)
                print(f"   ? Departement {dept} ({label})")
            except Exception:
                pass  # fichier absent ou pas encore publie, on continue

        # Periode historique avant 1950 - l'annee de debut varie selon le departement
        for debut in ["1851", "1852", "1860", "1864", "1870", "1875", "1880", "1900", "1920", "1929"]:
            url_hist = f"{base_url}MENSQ_{dept}_{debut}-1949.csv.gz"
            try:
                df_hist = pd.read_csv(url_hist, sep=";", compression="gzip", encoding="latin-1")
                dfs.append(df_hist)
                print(f"   ? Departement {dept} (historique depuis {debut})")
                break
            except Exception:
                continue

    if not dfs:
        raise RuntimeError("Impossible de charger les donnees de temperature Meteo France")

    df_all = pd.concat(dfs, ignore_index=True)

    # TM dans les fichiers MENSQ est en degC
    df_all["TM"] = pd.to_numeric(df_all["TM"], errors="coerce")

    # On extrait l'annee depuis le format AAAAMM (ex : 202401 -> 2024)
    df_all["annee"] = df_all["AAAAMM"].astype(str).str[:4].astype(int)
    df_all = df_all[(df_all["annee"] >= 1900) & (df_all["annee"] <= 2026)]
    df_all = df_all.dropna(subset=["TM"])

    # Moyenne nationale par annee
    df_annuel = df_all.groupby("annee")["TM"].mean().reset_index()
    df_annuel.columns = ["annee", "temp_moy_france"]

    # Remplissage des eventuelles annees manquantes par interpolation
    df_annuel = df_annuel.set_index("annee").reindex(range(1900, 2027))
    df_annuel = df_annuel.interpolate(method="linear").reset_index()

    print(f"Temperatures chargees : {len(df_annuel)} annees")
    df_annuel.to_csv("data/clean/temperatures.csv", index=False)

    # On conserve df_all pour charger_jours_extremes() sans re-telecharger
    return df_annuel, df_all


# ==============================================================================
# SOURCE 1b - Meteo France : jours de chaleur (TX>=30C) et jours de gel (TX<=0C)
# Colonnes NBJTX30 et NBJTX0 des fichiers MENSQ
# ==============================================================================

def charger_jours_extremes(df_all_mensq):
    print("Extraction jours extremes (TX>=30C / TX<=0C)...")

    df = df_all_mensq.copy()
    df["annee"] = df["AAAAMM"].astype(str).str[:4].astype(int)
    df = df[(df["annee"] >= 1950) & (df["annee"] <= 2026)]

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

    # NBJTX30/NBJTX0 = nb de jours dans le mois pour une station
    # 1 - somme des mois -> total annuel par station
    # 2 - moyenne entre toutes les stations -> valeur nationale
    df_annuel = (
        df.groupby(["NUM_POSTE", "annee"])[cols_dispo].sum()   # 1
          .reset_index()
          .groupby("annee")[cols_dispo].mean()                  # 2
          .reset_index()
    )
    df_annuel = df_annuel.rename(columns={
        "NBJTX30": "jours_chauds_30",
        "NBJTX0":  "jours_gel"
    })

    print(f"Jours extremes charges : {len(df_annuel)} annees "
          f"({', '.join(df_annuel.columns[1:].tolist())})")
    df_annuel.to_csv("data/clean/jours_extremes.csv", index=False)
    return df_annuel


# ==============================================================================
# SOURCE 2 - NOAA (noaa.gov)
# Concentrations atmospheriques de CO2 depuis 1958 (station Mauna Loa)
# Donnees annuelles officielles, format CSV simple
# ==============================================================================

def charger_co2():
    print("Chargement CO2 NOAA...")

    # URL du fichier annuel CO2 Mauna Loa - NOAA GML
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"

    # Ce fichier a des lignes de commentaires qui commencent par #
    df = pd.read_csv(url, comment="#", header=None)
    df.columns = ["annee", "co2_ppm", "incertitude"]
    df = df[["annee", "co2_ppm"]]

    # Valeurs pre-industrielles estimees (avant 1958, pas de mesures directes)
    # On utilise des valeurs de reference issues des carottes glaciaires IPCC
    annees_avant = list(range(1900, 1958))
    # CO2 en 1900 ≈ 296 ppm, en 1957 ≈ 315 ppm - progression lineaire estimee
    co2_avant = [296 + (315 - 296) * (a - 1900) / (1957 - 1900) for a in annees_avant]
    df_avant = pd.DataFrame({"annee": annees_avant, "co2_ppm": co2_avant})

    # On fusionne les deux periodes
    df_complet = pd.concat([df_avant, df], ignore_index=True)
    df_complet["annee"]   = pd.to_numeric(df_complet["annee"],   errors="coerce").astype("Int64")
    df_complet["co2_ppm"] = pd.to_numeric(df_complet["co2_ppm"], errors="coerce")
    df_complet = df_complet[df_complet["annee"] <= 2026]

    print(f"CO2 charge : {len(df_complet)} annees - "
          f"derniere valeur : {df_complet.iloc[-1]['co2_ppm']:.1f} ppm")
    df_complet.to_csv("data/clean/co2.csv", index=False)
    return df_complet


# ==============================================================================
# SOURCE 3 - CITEPA Secten edition 2025 - GES par secteur
# Fichiers dans : data/raw/co2_secteur/
# Structure de chaque fichier :
#   - Onglet "*-Emissions"
#   - Ligne 9 (index) : en-tete avec les annees en colonnes (a partir de col 5)
#   - La ligne dont col 4 = nom du secteur (ex: "Transports") = total du secteur
# ==============================================================================

def _extraire_total_secteur(chemin):
    """Lit un fichier CITEPA Secten et retourne (nom_secteur, serie_annuelle).

    Structure des fichiers :
      - Ligne 9 : en-tete  col0=NaN col1=Perimetre col2=Substance col3=NaN
                            col4=Secteur col5=Unite col6+=annees (1960, 1961…)
      - Ligne secteur total : col4=nom_secteur, col5="Mt CO2e", col6+=valeurs
    On identifie la ligne total via le mot-cle du nom de l'onglet (ex : "Energie").
    """
    xl = pd.ExcelFile(chemin)
    sheet = next((s for s in xl.sheet_names if "mission" in s), None)
    if sheet is None:
        return None, None

    df = pd.read_excel(chemin, sheet_name=sheet, header=None)

    # Mapping annee -> index de colonne (annees en col 6+)
    year_to_col = {}
    for c in range(6, df.shape[1]):
        v = pd.to_numeric(df.iloc[9, c], errors="coerce")
        if pd.notna(v) and 1900 <= v <= 2026:
            year_to_col[int(v)] = c

    if not year_to_col:
        return None, None

    col_secteur = df.iloc[:, 4].astype(str).str.strip()
    col_unite   = df.iloc[:, 5].astype(str).str.strip()

    # Ligne cible : unite "Mt CO2e", hors "Total national", "Autres secteurs", "%"
    exclure = r"total national|autres secteurs|%|nan"
    mask = (
        (col_unite == "Mt CO2e") &
        ~col_secteur.str.lower().str.contains(exclure, na=True, regex=True)
    )
    candidats = df[mask]
    if candidats.empty:
        return None, None

    idx_row = candidats.index[-1]  # prend la derniere si plusieurs
    nom = col_secteur.iloc[idx_row]

    # Extraction des valeurs pour chaque annee
    values = {}
    for year, c in year_to_col.items():
        val = pd.to_numeric(df.iloc[idx_row, c], errors="coerce")
        if pd.notna(val):
            values[year] = val

    if not values:
        return None, None

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

    series = {}
    for f in sorted(fichiers):
        chemin = os.path.join(dossier, f)
        nom, serie = _extraire_total_secteur(chemin)
        if serie is not None:
            # Nom de colonne court depuis le nom du fichier (ex: "Transports")
            col = f.split("_")[1].replace("-", "_").lower()
            series[col] = serie
            print(f"   ? {col} ({serie.dropna().shape[0]} annees)")

    if not series:
        print("[!]  Aucun secteur extrait")
        return None

    df_ges = pd.DataFrame(series)
    df_ges.index.name = "annee"
    df_ges = df_ges.reset_index()

    # Total national = somme de tous les secteurs disponibles
    cols_secteurs = [c for c in df_ges.columns if c != "annee"]
    df_ges["ges_total_MtCO2eq"] = df_ges[cols_secteurs].sum(axis=1, skipna=True)

    print(f"[OK] GES charge : {len(df_ges)} annees - "
          f"secteurs : {cols_secteurs}")
    df_ges.to_csv("data/clean/ges.csv", index=False)
    return df_ges


# ==============================================================================
# SOURCE 4 - PSMSL (Permanent Service for Mean Sea Level)
# Ports francais depuis data/raw/id_port_fr.xlsx (ID, Lat, Long, Ville)
# URL : https://psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata
# On centre chaque port sur 1961-1990 puis on moyenne les anomalies par annee.
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
            df = pd.read_csv(base_url.format(pid), sep=";", header=None,
                             names=["date_decimal", "hauteur_mm", "flag1", "flag2"])
            df["hauteur_mm"] = pd.to_numeric(df["hauteur_mm"], errors="coerce")
            df = df[df["hauteur_mm"] > -99999]
            df["annee"] = df["date_decimal"].astype(int)
            df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

            if df.empty:
                continue

            df_a = df.groupby("annee")["hauteur_mm"].mean()

            # Centrage individuel sur 1961-1990 pour rendre les ports comparables
            ref_mask = (df_a.index >= 1961) & (df_a.index <= 1990)
            if ref_mask.sum() >= 5:
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

    # DataFrame avec une colonne par port + moyenne nationale
    df_all = pd.concat(series, axis=1)
    df_all.index.name = "annee"
    df_all = df_all.reset_index()

    # Renommage des colonnes avec prefixe pour eviter les conflits
    ville_cols = {v: f"niveau_mer_{v.lower().replace(' ', '_').replace('-', '_')}"
                  for v in df_all.columns if v != "annee"}
    df_all = df_all.rename(columns=ville_cols)

    # Moyenne nationale en colonne supplementaire
    cols_villes = [c for c in df_all.columns if c != "annee"]
    df_all["niveau_mer_mm"] = df_all[cols_villes].mean(axis=1)

    print(f"[OK] Niveau de la mer : {len(df_all)} annees, "
          f"{len(series)} ports (anomalie/ref. 1961-1990)")
    print(f"   Colonnes : {cols_villes + ['niveau_mer_mm']}")
    df_all.to_csv("data/clean/niveau_mer.csv", index=False)
    return df_all


# ==============================================================================
# SOURCE 5 - Empreinte carbone individuelle
# INSEE ip2077 - emissions et empreinte carbone 2024
#   https://www.insee.fr/fr/statistiques/8654458
# ==============================================================================

def charger_empreinte_carbone():
    """
    Source : INSEE ip2077 - "Figure 3 : L'empreinte carbone et ses composantes de 1990 a 2024"
    https://www.insee.fr/fr/statistiques/8654458

    Colonne utilisee : "Empreinte totale" (col 4) en Mt CO2eq
    Calcul : empreinte_tCO2_hab = Empreinte totale (Mt) × 1 000 000 / population francaise

    """
    print("Chargement empreinte carbone individuelle (INSEE ip2077)...")

    import io

    url = "https://insee.fr/fr/statistiques/fichier/8654458/ip2077.xlsx"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Figure 3 : ligne 2 = en-tetes, lignes 3-37 = donnees 1990-2024
        df_raw = pd.read_excel(io.BytesIO(resp.content),
                               sheet_name="Figure 3", header=None)
        df_data = df_raw.iloc[3:38, [0, 4]].copy()
        df_data.columns = ["annee", "empreinte_Mt"]
        df_data["annee"]        = pd.to_numeric(df_data["annee"],        errors="coerce")
        df_data["empreinte_Mt"] = pd.to_numeric(df_data["empreinte_Mt"], errors="coerce")
        df_data = df_data.dropna()

        # Population francaise par interpolation sur points INSEE connus
        pop_ref = {1990: 58.0, 2000: 60.5, 2010: 64.6, 2020: 67.4, 2024: 68.4}
        pop_series = pd.Series(pop_ref).reindex(range(1990, 2027))
        pop_series = pop_series.interpolate(method="index")  # millions

        df_data["population_M"] = df_data["annee"].map(pop_series)

        # t CO2eq/hab = Mt × 1 000 000 tonnes / (population_M × 1 000 000 hab)
        #             = Mt / population_M
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
# SOURCE 6 - Dates de vendanges (indicateur phenologique fort)
# Fichier : data/raw/vendanges_france_1900_2026_estimations.xlsx
# Colonnes : Annee | Region | Date debut vendanges | Jour de l'annee (DOY)
# ==============================================================================

def charger_vendanges():
    print("... Chargement dates de vendanges...")

    chemin = "data/raw/vendanges_france_1900_2026_estimations.xlsx"

    if not os.path.exists(chemin):
        print("[!]  Fichier vendanges introuvable :", chemin)
        return None

    df = pd.read_excel(chemin, sheet_name="Vendanges_estimations")
    df.columns = [c.strip() for c in df.columns]

    # Detection automatique des colonnes annee et DOY (noms avec accents variables)
    col_annee = next(c for c in df.columns if "ann" in c.lower() or "year" in c.lower())
    col_doy   = next(c for c in df.columns if "doy" in c.lower() or "jour" in c.lower() and "ann" in c.lower())
    df = df[[col_annee, col_doy]].rename(
        columns={col_annee: "annee", col_doy: "jour_vendanges"}
    )
    df["annee"]          = pd.to_numeric(df["annee"],          errors="coerce")
    df["jour_vendanges"] = pd.to_numeric(df["jour_vendanges"], errors="coerce")
    df = df.dropna()
    df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

    # Moyenne sur toutes les regions -> 1 valeur par an
    df_annuel = df.groupby("annee")["jour_vendanges"].mean().reset_index()

    print(f" Vendanges chargees : {len(df_annuel)} annees - "
          f"derniere valeur : jour {df_annuel.iloc[-1]['jour_vendanges']:.0f} "
          f"(an {int(df_annuel.iloc[-1]['annee'])})")
    df_annuel.to_csv("data/clean/vendanges.csv", index=False)
    return df_annuel


# ==============================================================================
# SOURCE 7 - Cout economique des catastrophes naturelles (EM-DAT)
# Fichier : data/raw/public_emdat_custom_request_*_cout_eco_catastrophe.xlsx
# Base EM-DAT (Emergency Events Database) - Centre de Recherche sur l'Épidemiologie
# Colonne cle : "Total Damage, Adjusted ('000 US$)" - dommages ajustes a l'inflation
# Agrege par annee : somme des dommages en milliards USD (Mrd USD)
# ==============================================================================

def charger_cout_eco_catastrophe():
    print("Chargement cout economique des catastrophes (EM-DAT)...")

    # Recherche du fichier (nom dynamique avec date)
    import glob
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
    df["annee"]         = pd.to_numeric(df["annee"],         errors="coerce")
    df["dommages_k_usd"] = pd.to_numeric(df["dommages_k_usd"], errors="coerce")
    df = df.dropna(subset=["annee"])
    df = df[(df["annee"] >= 1900) & (df["annee"] <= 2026)]

    # Somme annuelle, conversion en milliards USD (÷ 1 000 000)
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
# On assemble toutes les sources en un seul fichier pret pour Prophet
# ==============================================================================

def fusionner_sources(df_temp, df_co2, df_ges, df_jours_ext,
                      df_niveau_mer, df_empreinte, df_vendanges, df_catastrophes):
    print("\n Fusion de toutes les sources...")

    # Base : les temperatures (la serie la plus longue - 1900-2026)
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
            df_src["annee"] = pd.to_numeric(df_src["annee"], errors="coerce")
            df_final = df_final.merge(df_src, on="annee", how="left")
            print(f"   ? {nom} fusionne")

    # Interpolation des valeurs manquantes restantes
    df_final = df_final.set_index("annee")
    df_final = df_final.interpolate(method="linear")
    df_final = df_final.reset_index()

    # Sauvegarde du fichier final - c'est ce fichier que modele_prophet.py va lire
    df_final.to_csv("data/clean/dataset_final.csv", index=False)

    print(f"\n Dataset final cree : {df_final.shape[0]} lignes ? "
          f"{df_final.shape[1]} colonnes")
    print(f"   Colonnes : {list(df_final.columns)}")
    print(f"   Periode  : {df_final['annee'].min()} -> {df_final['annee'].max()}")
    print(f"\n Fichier pret : data/clean/dataset_final.csv")
    print("   -> Tu peux maintenant lancer modele_prophet.py")
    return df_final


# ==============================================================================
# POINT D'ENTRÉE - Lance ce fichier avec : python pipeline.py
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HACKATHON #26 - Pipeline de donnees climatiques")
    print("=" * 60)

    df_temp, df_all_mensq = charger_temperatures()
    df_jours_ext          = charger_jours_extremes(df_all_mensq)
    df_co2                = charger_co2()
    df_ges                = charger_ges()
    df_niveau_mer         = charger_niveau_mer()
    df_empreinte          = charger_empreinte_carbone()
    df_vendanges          = charger_vendanges()
    df_catastrophes       = charger_cout_eco_catastrophe()

    df_final = fusionner_sources(
        df_temp, df_co2, df_ges, df_jours_ext,
        df_niveau_mer, df_empreinte, df_vendanges, df_catastrophes
    )

    print("\n" + "=" * 60)
    print("Pipeline termine. Apercu des donnees :")
    print("=" * 60)
    print(df_final.tail(10).to_string(index=False))