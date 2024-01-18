import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import folium
from folium.features import DivIcon
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV


################# FONCTIONS POUR L'APPLICATION #################

colors_list = [
    "#E01E5A",
    "#ECB22E",
    "#2EB67D",
    "#36C5F0",
    "#7C2852",
    "#4A154B",
    "#1E328F",
    "#78D7DD",
]

prepro = load("preprocessor.joblib")
prepro2 = load("preprocessor2.joblib")


def pie_spec(df: pd.DataFrame, feat: str, ax: plt.Axes, titre: str, seuil: float):
    """Retourne un diagramme circulaire customisé : l'idée est de ne pas montrer les catégories qui seraient trop "petites" et
        qui rendent la compréhension du graph difficile, sans apporter plus de valeur
    Args:
        df (DataFrame): dataframe issu du scrapping
        feat (str): le nom de la feature catégorielle que l'on souhaite visualiser sous forme de pie chart
        ax (_type_): pour intégrer le pie plot dans subplot
        titre (str): titre du pie plot
        seuil (float): seuil à partir duquel on ne veut pas voir la catégorie apparaitre sur le pie plot
    """

    collec = ax.pie(
        df[feat].value_counts().values,
        labels=df[feat].value_counts().index,
        autopct="%1.0f%%",
        colors=colors_list,
    )
    ax.set_title(titre)
    list_val = df[feat].value_counts(normalize=True).tolist()
    if len(list_val) > 3:
        for i in range(len(df[feat].value_counts().index)):
            if list_val[i] < seuil:
                collec[1][i].remove()
                collec[2][i].remove()


def graphs(df: pd.DataFrame):
    """Retourne un ensemble de graph (distribution, classification) permettant de prendre connaissance des données récoltées via le scrapping

    Args:
        df (DataFrame): dataframe issu du scrapping
    """

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    df["prix"].plot(
        kind="hist",
        title="Distribution du prix des véhicules",
        ax=axs[0, 0],
        color="#1E328F",
    ).set_ylabel("")
    df[["name", "immat_year"]].groupby("immat_year").count().plot(
        kind="bar",
        legend=None,
        ax=axs[0, 1],
        title="Nombre de véhicules par année d'immatriculation",
        color="#1E328F",
    ).set_xlabel("")
    axs[0, 1].tick_params(axis="both", which="major", rotation=45, labelsize=8)
    df["kilometrage"].plot(
        kind="hist",
        title="Distribution du kilométrage des véhicules",
        ax=axs[0, 2],
        color="#1E328F",
    ).set_ylabel("")
    axs[0, 2].tick_params(axis="both", which="major", labelsize=8)
    pie_spec(df, "carburant", axs[1, 0], "Répartition des types de carburant", 0.04)
    pie_spec(
        df, "transmission", axs[1, 1], "Répartition des types de transmission", 0.04
    )
    pie_spec(
        df, "seller_location", axs[1, 2], "Répartition de l'origine des véhicules", 0.04
    )
    fig.tight_layout(pad=2)
    return fig


def price_corr(df: pd.DataFrame):
    """Retourne un ensemble de graphs mettant en évidence la corrélation entre le prix des véhicules et le reste des variables du dataset

    Args:
        df (DataFrame): dataframe issu du scrapping
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    df.plot.scatter(
        x="kilometrage",
        y="prix",
        ax=axs[0, 0],
        title="Prix en fonction du nombre de kilomètre",
        color="#1E328F",
    )
    axs[0, 0].tick_params(axis="x", which="major", labelsize=8)

    df.plot.scatter(
        x="immat_year",
        y="prix",
        ax=axs[0, 1],
        title="Prix en fonction de l'age du véhicule",
        color="#1E328F",
    )

    sns.violinplot(x=df["seller_type"], y=df["prix"], ax=axs[0, 2], color="#1E328F")
    axs[0, 2].set_title("Distribution du prix en fonction du type de vendeur")

    sns.violinplot(x=df["seller_location"], y=df["prix"], ax=axs[1, 0], color="#1E328F")
    axs[1, 0].set_title(
        "Distribution du prix en fonction de \nla localisation du vendeur"
    )

    sns.violinplot(x=df["transmission"], y=df["prix"], ax=axs[1, 1], color="#1E328F")
    axs[1, 1].set_title("Distribution du prix en fonction \ndu type de transmission")
    axs[1, 1].tick_params(axis="x", which="major", labelsize=8)

    sns.violinplot(x=df["carburant"], y=df["prix"], ax=axs[1, 2], color="#1E328F")
    axs[1, 2].set_title("Distribution du prix en fonction \ndu type de carburant")
    axs[1, 2].tick_params(axis="x", which="major", labelsize=8)

    fig.tight_layout(pad=3)
    return fig


def viz_seller_geo(df: pd.DataFrame):
    """Visualisation des vendeurs sur une carte d'Europe

    Args:
        df (DataFrame): dataframe issu du scrapping
    """

    # on récupère les répartitions par pays dans un nouveau dataframe
    countries = df["seller_location"].value_counts().index
    nb_vehicules = df["seller_location"].value_counts().values
    dic_seller_loc = {"country": countries, "vehicule_for_sale": nb_vehicules}
    df_byloc = pd.DataFrame(dic_seller_loc)

    # on trouve les coordonnées des pays à avec geolocator
    geolocator = Nominatim(user_agent="me")

    def geolocate(country):
        try:
            # Geolocalise le centre du pays
            loc = geolocator.geocode(country)
            # et retourne longitude et latitude
            return (loc.latitude, loc.longitude)
        except:
            return np.nan

    df_byloc["geolocation"] = df_byloc["country"].map(
        geolocate
    )  # coordonnées (long,lat)
    df_byloc.geolocation = df_byloc.geolocation.astype("str")

    # création d'un dataframe avec deux colonnes (long et lat) à partir de la colonne créée précedemment
    detailed_coord = df_byloc.geolocation.str.split(",", expand=True)
    detailed_coord.rename(columns={0: "latitude", 1: "longitude"}, inplace=True)

    # concaténation des deux dataframe et nettoyage des données
    final_df = pd.concat([df_byloc, detailed_coord], axis=1, join="inner")
    final_df["latitude"] = final_df["latitude"].str.replace("(", "", regex=True)
    final_df["longitude"] = final_df["longitude"].str.replace(")", "", regex=True)
    final_df[["latitude", "longitude"]] = final_df[["latitude", "longitude"]].apply(
        pd.to_numeric
    )

    # création de la map
    world_map = folium.Map(
        location=[47, 8], tiles="cartodbpositron", zoom_start=5, width=700, height=600
    )
    nb_total = final_df.vehicule_for_sale.sum()

    for i in range(len(final_df)):
        lat = final_df.iloc[i]["latitude"]
        long = final_df.iloc[i]["longitude"]
        nb_i = final_df.iloc[i]["vehicule_for_sale"]
        radius = 600000 * (nb_i / nb_total)
        text = f"{nb_i} voitures en vente"
        folium.Circle([lat, long], radius=radius, fill=True).add_to(world_map)
        folium.map.Marker(
            [lat + 0.4, long - 0.5],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html='<div style="font-size: 15pt; font-weight:700; color: #ff6600;">%s</div>'
                % nb_i,
            ),
        ).add_to(world_map)

    map_html = world_map._repr_html_()
    return map_html


def filtre(
    df: pd.DataFrame,
    trans: str,
    carb: str,
    seller_loc: str,
    seller_t: str,
    sort: str,
    croissant: str,
):
    """Retourne un dataframe filtré en fonction des éléments indiqués en input

    Args:
        df (pd.DataFrame): datframe conetnant les données récoltées
        trans (str): type de transmission qu'on veut filtrer
        carb (str): type de carburant qu'on veut filtrer
        seller_loc (str): type de localisation geo qu'on veut filtrer
        seller_t (str): type de vendeur qu'on veut filtrer
        sort (str): feature numérique avec laquelle on trier le dataset
        croissant (str): Oui si trie croissante, non sinon

    Returns:
        _type_: _description_
    """
    if trans == "Pas de filtre":
        df1 = df.copy()
    else:
        df1 = df[df["transmission"] == trans]

    if carb == "Pas de filtre":
        df2 = df1.copy()
    else:
        df2 = df1[df1["carburant"] == carb]

    if seller_loc == "Pas de filtre":
        df3 = df2.copy()
    else:
        df3 = df2[df2["seller_location"] == seller_loc]

    if seller_t == "Pas de filtre":
        df4 = df3.copy()
    else:
        df4 = df3[df3["seller_type"] == seller_t]

    if croissant == "Oui":
        df4 = df4.sort_values(by=sort, ascending=True)
    else:
        df4 = df4.sort_values(by=sort, ascending=False)

    return df4


def train_test_preprocess1(df: pd.DataFrame, preprocessor):
    """Preprocess les données collectées pour pouvoir les utiliser dans des algo de ML
    (1) sépare les features de la target (X et y)
    (2) scinde le dataset en training set et test set
    (3) preprocess les données en fonction de leur type (num ou cat) et de leur nature (gestion des valeurs manquantes différentes)

    Args:
        df (DataFrame): données collectées
        preprocessor (pipeline sklearn): pour le preprocessing des données
    Return:
        X_train_scl et X_test_scl = les données du trainig et test set prétraitées
        y_train et y_test = les response variables pour les deux sets
    """
    dfml = pd.DataFrame(
        df
    )  # pour pouvoir prend en compte les nan qui sont remplacés par un dico vide sinon
    n = len(dfml)

    drop_missing = ["kilometrage", "seller_location"]
    dfml.dropna(subset=drop_missing, inplace=True)
    print(f"Suppression de {n-len(dfml)} véhicules")

    target = "prix"
    X = dfml.drop(
        columns=[target, "name"]
    )  # suppression de la variable 'name' qui n'a pas d'interet pour le modèle
    y = dfml[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=12
    )
    X_train_scl = preprocessor.transform(X_train)
    X_test_scl = preprocessor.transform(X_test)

    return X_train_scl, X_test_scl, y_train, y_test


def train_test_preprocess2(df: pd.DataFrame, preprocessor):
    """Preprocess les données collectées pour pouvoir les utiliser dans des algo de ML
    (1) sépare les features de la target (X et y)
    (2) One Hot encode les variables catégorielles (avant split du dataset pour être sûr qu'il n'y a pas de données inconnues dans test set) -- litérature pas clair sur le sujet
    (3) scinde le dataset en training set et test set
    (4) preprocess les données numériques en fonction de leur nature (gestion des valeurs manquantes différentes)

    Args:
        df (DataFrame): données collectées
        preprocessor (pipeline sklearn): pour le preprocessing des données
    Return:
        X_train_scl et X_test_scl = les données du trainig et test set prétraitées
        y_train et y_test = les response variables pour les deux sets
    """
    dfml = pd.DataFrame(
        df
    )  # pour pouvoir prend en compte les nan qui sont remplacés par un dico vide sinon
    n = len(dfml)

    drop_missing = ["kilometrage", "seller_location"]
    dfml.dropna(subset=drop_missing, inplace=True)
    print(f"Suppression de {n-len(dfml)} véhicules")

    target = "prix"
    X = dfml.drop(
        columns=[target, "name"]
    )  # suppression de la variable 'name' qui n'a pas d'interet pour le modèle
    y = dfml[target]

    cat = X.select_dtypes(include=['object']).columns.to_list()
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',sparse_output=False)) 
        ])
    
    OHE_preprocess = ColumnTransformer(transformers=[
        ('cat',cat_pipeline,cat)],
        verbose_feature_names_out=False,
        remainder='passthrough')
    
    OHE_preprocess.set_output(transform='pandas')
    X_ohe = OHE_preprocess.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_ohe, y, test_size=0.1, random_state=12
    )
    X_train_scl = preprocessor.transform(X_train)
    X_test_scl = preprocessor.transform(X_test)

    return X_train_scl, X_test_scl, y_train, y_test, OHE_preprocess


def search_best_model(model, X_train: np.array, y_train: np.array, param=None):
    """A partir d'un modèle et d'un ensemble de paramètre, retourne les paramètres donnant lieu aux meilleures performances

    Args:
        model (model sklearn): model à entrainer
        X_train (np.array): features
        y_train (np.array): target
        param (dict, optional): dictionnaire de paramètre à essayer. Defaults to None.

    Returns:
        dict: dictionnaire des meilleurs paramètres
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(model, param, scoring="neg_root_mean_squared_error", cv=kfold)
    result = search.fit(X_train, y_train)
    best_param = result.best_params_
    return best_param


def fit_model(model, best_para: dict, X_train: np.array, y_train: np.array):
    """Fit le modèle avec les meilleurs paramètres

    Args:
        model (model sklearn): modèle à fitter
        best_para (dict): dictionnaire de paramètres
        X_train (np.array): features
        y_train (np.array): target

    Returns:
        model: model fitté
    """
    modelf = model.set_params(**best_para)
    modelf.fit(X_train, y_train)
    return modelf


def pricer(
    kilometrage: int,
    year: int,
    transmission: str,
    carburant: str,
    puissance: int,
    location: str,
    type_vendeur: str,
    preprocessing,
    model_fitted,
    OHE_preprocess=None
):
    """Prend les données input comme des features et les soumet au modèle pour obtenir la prédiction sur cette nouvelle instance

    Args:
        kilometrage (int): kilometrage du véhicule dont on souhaite connaitre la valeur de marché
        year (int): année d'immatriculation du véhicule dont on souhaite connaitre la valeur de marché
        transmission (str): type de transmission du véhicule dont on souhaite connaitre la valeur de marché
        carburant (str): type de carburant du véhicule dont on souhaite connaitre la valeur de marché
        puissance (int): puissance du véhicule dont on souhaite connaitre la valeur de marché
        location (str): lieu d'immatriculation du véhicule dont on souhaite connaitre la valeur de marché
        type_vendeur (str): type de vendeur
        preprocessing (pipeline sklearn): pour le preprocessing des nouvelles données avant de les soumettre au modèle
        model_fitted (model sklearn): modele fitté

    Returns:
        _type_: _description_
    """

    # on récupère dans le dictionnaire new_data uniquement les features qui vont servir pour la prédiction (par ex, on ne garde par les features qui ont été supprimé lors du preprocessing)
    new_data = {
        "kilometrage": [kilometrage],
        "promo_%": [
            0
        ],  # cette feature a été donnée au modèle pour qu'il puisse apprendre des patterns (notamment au niveau de la "cote" de certains modèles sur certains marchés), mais ce n'est pas une feature pertinente pour un vendeur qui cherche à connaitre la valeur de marché de son véhicule
        "immat_year": [year],
        "puissance_chv": [puissance],
        "transmission": [transmission],
        "carburant": [carburant],
        "seller_location": [location],
        "seller_type": [type_vendeur],
    }

    X_new = pd.DataFrame(new_data)
    if OHE_preprocess:
        X_ohe = OHE_encoder.transform(X_new)
        X_new_scl = preprocessing.transform(X_ohe)
    else:
        X_new_scl = preprocessing.transform(X_new)
    predictions = model_fitted.predict(X_new_scl)
    return predictions


################# SETTING DE L'APPLICATION #################

st.set_page_config(
    page_title="Valeur de marché d'un véhicule d'occasion",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# JS code to modify te decoration on top
# changer le param decoration.style.right pour couvrir (20) ou découvrir le logo (à partir de 200) 'running'
st.components.v1.html(
    """
    <script>
    // Modify the decoration on top to reuse as a banner

    // Locate elements
    var decoration = window.parent.document.querySelectorAll('[data-testid="stDecoration"]')[0];
    var sidebar = window.parent.document.querySelectorAll('[data-testid="stSidebar"]')[0];

    // Observe sidebar size
    function outputsize() {
        decoration.style.left = `${sidebar.offsetWidth}px`;
    }

    new ResizeObserver(outputsize).observe(sidebar);

    // Adjust sizes
    outputsize();
    decoration.style.height = "5.0rem";
    decoration.style.right = "200px"; 

    // Adjust image decorations
    decoration.style.backgroundImage = "url(https://i.etsystatic.com/14685610/r/il/d2a944/2138438916/il_1588xN.2138438916_oqby.jpg)";
    decoration.style.backgroundSize = "contain";
    </script>        
    """,
    width=0,
    height=0,
)


with st.container():
    st.title("Valeur de marché d'un véhicule d'occasion")
    st.markdown(
        "🚘 *Cet outil permet d'obtenir des informations concernant le marché de l'occasion du véhicule de votre choix* 🚘"
    )
    st.markdown("##")

list_tab = [
    "Informations concernant \nle véhicule recherché",
    "Pays d'origine",
    "Analyses graphiques",
    "Filtres personnalisés",
    "Valeur de marché du \nvéhicule (pricing)",
]
# style des tabs
st.markdown(
    """
  <style>
    .stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
        margin: 0px 2px 0px 2px;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 2px;
		padding-bottom: 2px;
        padding-right: 5px;
        padding-left: 5px;
    }
  </style>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([s for s in list_tab])


with st.sidebar:
    st.markdown(
        "Veuillez spécifier le marque et le modèle du véhicule, puis cochez la case 'Analyse'"
    )
    user_marque = st.text_input("Marque")
    user_modele = st.text_input("Modèle")
    mycheckb = st.checkbox("Analyse", key="one")
    st.markdown(
        "<h1 style='font-size:13px;font-style:italic'>Pour faire une nouvelle recherche, veuillez dé-cocher la case 'Analyse' avant de modifier la marque et le modèle</h1>",
        unsafe_allow_html=True,
    )
    schema = {"marque": user_marque, "modele": user_modele}

# with Session State, it's possible to have values persist across reruns for those instances when you don't want your variables reinitialized
# cela permet de ne pas avoir à faire une request vers fastapi à chaque fois que l'utilisateur clique sur un bouton
# on conserve les valeurs récupérés lors de la recherche initiale

if mycheckb:
    if "img_url" not in st.session_state:
        req_img = requests.post("http://backend:8000/car_pic", json=schema)
        resultat_img = req_img.json()
        img_url = resultat_img["pic_url"]
        st.session_state.img_url = img_url

    with st.sidebar:
        st.markdown(f"![Alt Text]({st.session_state.img_url})")

    if "request" not in st.session_state:
        req = requests.post("http://backend:8000/car_info", json=schema)
        resultat = req.json()
        st.session_state.request = resultat
    data_car = pd.DataFrame(st.session_state.request["df_info"])
    nb = st.session_state.request["nb_vec"]
    prix = st.session_state.request["prix"]
    km = st.session_state.request["km"]
    age = st.session_state.request["age"]

    with st.sidebar:
        st.markdown(
            f"<h1 style='font-size:15px;font-weight:bold'>Nombre de véhicules en vente</h1> {nb}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h1 style='font-size:15px;font-weight:bold'>Prix moyen</h1> {prix}€",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h1 style='font-size:15px;font-weight:bold'>Kilometrage moyen</h1> {km} km",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h1 style='font-size:15px;font-weight:bold'>Ancienneté moyenne</h1> {age} ans",
            unsafe_allow_html=True,
        )

    with tab1:
        st.markdown(
            f"Vous pouvez trier le tableau en cliquant sur la colonne qui vous intéresse"
        )
        for col in data_car.columns:
            data_car[col] = data_car[col].apply(
                lambda x: x if not type(x) == dict else np.nan
            )
        st.dataframe(data_car)

    with tab2:
        fig3 = viz_seller_geo(data_car)
        st.components.v1.html(fig3, height=1000)

    with tab3:
        fig1 = graphs(data_car)
        st.pyplot(fig1)
        fig2 = price_corr(data_car)
        st.pyplot(fig2)

    with tab4:
        with st.form(key="info_for_filter"):
            col1, col2 = st.columns([0.5, 0.5], gap="medium")
            with col1:
                fil_trans = st.selectbox(
                    "Transmission",
                    data_car["transmission"].value_counts().index.tolist()
                    + ["Pas de filtre"],
                )
                fil_carb = st.selectbox(
                    "Carburant",
                    data_car["carburant"].value_counts().index.tolist()
                    + ["Pas de filtre"],
                )
                fil_loc = st.selectbox(
                    "Pays d'immatriculation",
                    data_car["seller_location"].value_counts().index.tolist()
                    + ["Pas de filtre"],
                )
                submit_button_filtre = st.form_submit_button(label="Filtrer")
            with col2:
                fil_vtyp = st.selectbox(
                    "Type de vendeur",
                    data_car["seller_type"].value_counts().index.tolist()
                    + ["Pas de filtre"],
                )
                sort_by = st.selectbox(
                    "Trier par", ["prix", "kilometrage", "puissance_chv"]
                )
                sort_how = st.selectbox("Ordre croissant", ["Oui", "Non"])
        if submit_button_filtre:
            data_car_filtred = filtre(
                data_car, fil_trans, fil_carb, fil_loc, fil_vtyp, sort_by, sort_how
            )
            st.markdown(
                f"{len(data_car_filtred)} véhicules correspondent à votre recherche."
            )
            st.dataframe(data_car_filtred)

    with tab5:
        X_train_scl, X_test_scl, y_train, y_test = train_test_preprocess1(
            data_car, prepro
        )

        param_grid1 = {"learning_rate": [0.1, 0.2, 0.4, 0.6, 0.8, 1]}
        best_params = search_best_model(
            model=XGBRegressor(),
            X_train=X_train_scl,
            y_train=y_train,
            param=param_grid1,
        )
        param_grid2 = {
            "learning_rate": [best_params["learning_rate"]],
            "lambda": [20, 50, 100, 120, 250],
        }
        xgb_params = search_best_model(
            model=XGBRegressor(),
            X_train=X_train_scl,
            y_train=y_train,
            param=param_grid2,
        )

        xgboost = fit_model(XGBRegressor(), xgb_params, X_train_scl, y_train)

        pred_final = xgboost.predict(X_test_scl)
        final_rmse = mean_squared_error(y_test, pred_final, squared=False)

        with st.form(key="info_to_submit"):
            col1, col2 = st.columns([0.5, 0.5], gap="medium")
            with col1:
                user_km = st.text_input("Kilométrage du véhicule")
                user_age = st.text_input("Année d'immatriculation du véhicule")
                user_transmission = st.selectbox(
                    "Type de transmission",
                    (data_car["transmission"].value_counts().index.tolist()),
                )
                user_carburant = st.selectbox(
                    "Carburant", (data_car["carburant"].value_counts().index.tolist())
                )
                submit_button = st.form_submit_button(label="Estimation")
            with col2:
                user_puissance = st.text_input("Puissance en chevaux")
                user_geo = st.selectbox(
                    "Pays d'immatriculation",
                    (data_car["seller_location"].value_counts().index.tolist()),
                )
                user_vendeur = st.selectbox(
                    "Etes vous un particulier ou un professionnel ?",
                    (data_car["seller_type"].value_counts().index.tolist()),
                )

        if submit_button:
            estimation = pricer(
                user_km,
                user_age,
                user_transmission,
                user_carburant,
                user_puissance,
                user_geo,
                user_vendeur,
                prepro,
                xgboost,
            )
            current_date = datetime.date.today()
            st.markdown(
                f'<p style="font-size:20px">Etant donné les informations communiquées, à la date du {current_date.day} {current_date.strftime("%b")} {current_date.year}, la valeur de marché de votre <span style="font-weight:bold">{user_marque} {user_modele}</span> est <span style="font-size:25px;font-weight:bold;color: #36C5F0">{estimation[0]:,.2f}€</span> <span style="font-size:12px">(marge erreur +/-{final_rmse/y_test.mean()*100:.2f}%)</span></p>',
                unsafe_allow_html=True,
            )
