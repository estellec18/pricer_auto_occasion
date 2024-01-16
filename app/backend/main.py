from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import datetime

# uvicorn main:app
app = FastAPI()


class UserInput(BaseModel):
    marque: str
    modele: str


@app.get("/")
def visit():
    return JSONResponse(status_code=200, content="la connection à l'API fonctionne")


# si GET http://127.0.0.1:8000/car_info?marque=bmw&modele=320
@app.post("/car_info")
def get_car_info(item: UserInput):
    """A partir d'une marque (ex:Mercedes) et d'un model (ex:amg-gt), retourne un dataframe contenant les informations collectées sur le site autoscoot24
    Pour chaque instance, les features sont les suivantes : nom (détail donné par le vendeur), kilométrage, prix, promotion en cours, type de carburant, type de transmission, type de vendeur, localisation du vendeur

    Args:
        marque (str): marque du véhicule recherché
        modele (str): modèle du véhicule recherché (doit etre en adéquation avec la marque sinon pas de résultat)
        cf. la nomenclature du site autoscoot24 pour bien définir le couple marque-modèle
    """

    if (type(item.marque) != str) or (type(item.modele) != str):
        print(
            "La marque et le modèle du véhicule doivent être indiqués sous format string : entre guillemet"
        )
        return

    i = 1
    url = f"https://www.autoscout24.fr/lst/{item.marque}/{item.modele}?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&page={i}&powertype=kw&search_id=29u2akin6i7&sort=standard&source=homepage_search-mask&ustate=N%2CU"
    r = requests.get(url)

    if r.status_code == 200:
        soup = BeautifulSoup(r.content, "html.parser")

        # permet de trouver le nombre max de pages [attention : le site autoscoot24 se limite à 20 pages par recherche -> donc max de 400 instances collectées]
        pages = soup.find("div", class_="ListPage_pagination__4Vw9q").text
        number = re.findall(r"\d+", pages.split("/")[1].strip())
        nb_pages = int(number[0])

        nom = []
        prix = []
        promo = []
        list_km = []
        list_transmission = []
        list_carburant = []
        list_date = []
        puissance_ch = []
        location = []
        vendeur = []

        for i in range(1, nb_pages + 1):
            url = f"https://www.autoscout24.fr/lst/{item.marque}/{item.modele}?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&page={i}&powertype=kw&search_id=29u2akin6i7&sort=standard&source=homepage_search-mask&ustate=N%2CU"
            r = requests.get(url)
            soup = BeautifulSoup(r.content, "html.parser")

            voitures = soup.find_all("div", class_="ListItem_wrapper__TxHWu")
            for voiture in voitures:
                nom.append(
                    voiture.find("span", class_="ListItem_version__5EWfi").text.strip()
                )

                all_price = voiture.find(
                    "div", class_="PriceAndSeals_wrapper__BMNaJ"
                ).text
                detailed = all_price.split(",")
                if len(detailed) < 3:
                    price = detailed[0].strip("\u202f").strip("€")
                    price = int("".join(list(map(lambda x: x.strip(), price.split()))))
                    prix.append(price)
                    promo.append(0)
                else:
                    price2 = detailed[1].strip("\u202f").strip("-€")
                    price2 = int(
                        "".join(list(map(lambda x: x.strip(), price2.split())))
                    )
                    prix.append(price2)
                    price1 = int(detailed[0].strip("€").replace(".", ""))
                    promo.append(round(((price2 - price1) / price1) * 100, 2))

                detail = (
                    voiture.find("div", class_="VehicleDetailTable_container__XhfV1")
                    .get_text()
                    .replace("km", "")
                )
                new = detail.split(" ")

                if new[0] == "-":
                    km = np.nan
                else:
                    km = new[0].strip("\u202f")
                    km = int("".join(list(map(lambda x: x.strip(), km.split()))))
                list_km.append(km)

                if new[-1] == "CH)":
                    if new[-2].strip("(").isdigit():
                        ch = int(new[-2].strip("("))
                    else:
                        ch = np.nan
                else:
                    if new[-5].strip("("):
                        ch = int(new[-5].strip("("))
                    else:
                        ch = np.nan
                puissance_ch.append(ch)

                if new[1] == "-":
                    transmission = np.nan
                    if new[3].startswith("(A"):
                        date = np.nan
                    else:
                        date = (
                            re.findall(r"\d+", new[2])[0]
                            + "/"
                            + re.findall(r"\d+", new[2])[1]
                        )
                    if len(re.findall(r"[a-zA-Z]+", new[2])) > 2:
                        carburant = (
                            "E"
                            + re.findall(r"[a-zA-Z]+", new[2])[1]
                            + "/"
                            + re.findall(r"[a-zA-Z]+", new[2])[2]
                        )
                    elif len(re.findall(r"[a-zA-Z]+", new[2])) == 2:
                        carburant = re.findall(r"[a-zA-Z]+", new[2])[1]
                    else:
                        carburant = np.nan

                elif new[1] == "Boîte":
                    transmission = new[1] + " " + re.findall(r"[a-zA-Z]+", new[2])[0]
                    if new[3].startswith("(A"):
                        date = np.nan
                    else:
                        date = (
                            re.findall(r"\d+", new[2])[0]
                            + "/"
                            + re.findall(r"\d+", new[2])[1]
                        )
                    if len(re.findall(r"[a-zA-Z]+", new[2])) > 2:
                        carburant = (
                            "E"
                            + re.findall(r"[a-zA-Z]+", new[2])[1]
                            + "/"
                            + re.findall(r"[a-zA-Z]+", new[2])[2]
                        )
                    elif len(re.findall(r"[a-zA-Z]+", new[2])) == 2:
                        carburant = re.findall(r"[a-zA-Z]+", new[2])[1]
                    else:
                        carburant = np.nan

                else:
                    detail_spec = re.findall(r"[a-zA-Z]+", new[1])
                    transmission = detail_spec[0] + "-" + detail_spec[1]
                    carburant = detail_spec[2]
                    date = (
                        re.findall(r"\d+", new[1])[0]
                        + "/"
                        + re.findall(r"\d+", new[1])[1]
                    )
                list_transmission.append(transmission)
                list_carburant.append(carburant)
                list_date.append(date)

            regex = re.compile("leRMu$|THzvQ$")
            sellers = soup.find_all("span", {"class": regex})
            for seller in sellers:
                address = seller.text
                address_detail = re.split("• |,", address)
                if address_detail[0] == "Particuliers":
                    vendeur.append("Particulier")
                else:
                    vendeur.append("Professionnel")
                if len(address_detail) > 1:
                    address_final = address_detail[1]
                else:
                    address_final = address_detail[0]
                country = address_final[:2]
                location.append(country)

        # création du dataframe à partir des données collectées lors du scrapping
        dico_df = {
            "name": nom,
            "kilometrage": list_km,
            "prix": prix,
            "promo_%": promo,
            "transmission": list_transmission,
            "1ere_immat": list_date,
            "carburant": list_carburant,
            "puissance_chv": puissance_ch,
            "seller_location": location,
            "seller_type": vendeur,
        }
        df = pd.DataFrame(dico_df)

        # on récupère seulement l'année de la date d'immatriculation et on supprime la colonne "1ere_imat"
        df["1ere_immat"] = pd.to_datetime(df["1ere_immat"], format="%m/%Y")
        df.insert(4, "immat_year", df["1ere_immat"].dt.year)
        df.drop(["1ere_immat"], axis=1, inplace=True)

        # mise en forme de certaines colonnes
        df[["puissance_chv", "immat_year", "kilometrage", "prix"]] = df[
            ["puissance_chv", "immat_year", "kilometrage", "prix"]
        ].astype("Int64")

        # corrections manuelles de certaines erreures récurentes liées au scrapping
        df["carburant"] = df["carburant"].replace("Ete/lectrique", "Electrique/Essence")
        df["carburant"] = df["carburant"].replace("Ete/Essence", "Essence")
        df["carburant"] = df["carburant"].replace("Ete/Autres", "Autres")
        df["carburant"] = df["carburant"].replace("lectrique", "Electrique")
        df["carburant"] = df["carburant"].replace("te", "Electrique")
        df["seller_location"] = df["seller_location"].str.strip()
        df.drop(df.loc[df["seller_location"].str.len() == 1].index, inplace=True)

        current_year = datetime.date.today().year
        float_formater = "{:,}".format
        info_prix = float_formater(df.prix.mean().round(2))
        info_km = float_formater(df.kilometrage.mean().round(2))
        info_age = float_formater(round(current_year - df.immat_year.mean(), 2))

        # pour s'assurer que toutes les valeurs manquantes sont au même format np.nan (sinon <NA>)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if not pd.isna(x) else np.nan)
        # les nan sont remplacé par None car Json ne reconnait pas le format nan et renvoie une erreur
        clean_df = df.replace(np.nan, None)
        # clean_df = df.dropna() --> version hard à éviter car supprime potentiellement beaucoup de ligne
        df_data = clean_df.to_dict()

        return {
            "prix": info_prix,
            "km": info_km,
            "age": info_age,
            "df_info": df_data,
            "nb_vec": len(df),
        }

        # return HTMLResponse(df.to_html())

    else:
        return f"erreur de connection au site ; error code : {r.status_code}"


@app.post("/car_pic")
def show_car_image(item: UserInput):
    """A partir d'une marque et d'un modèle, retourne une image du véhicule recherché (par simplicité, il s'agit de l'image du premier véhicule disponible à la vente sur le site autoscoot)

    Args:
        marque (str): marque du véhicule recherché
        modele (str): modèle du véhicule recherché (doit etre en adéquation avec la marque sinon pas de résultat)
        cf. la nomenclature du site autoscoot24 pour bien définir le couple marque-modèle
    """

    if (type(item.marque) != str) or (type(item.modele) != str):
        print(
            "La marque et le modèle du véhicule doivent être indiqués sous format string : entre parenthèse"
        )
        return

    url = f"https://www.autoscout24.fr/lst/{item.marque}/{item.modele}?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&powertype=kw&search_id=29u2akin6i7&sort=standard&source=homepage_search-mask&ustate=N%2CU"
    r = requests.get(url)

    if r.status_code == 200:
        soup = BeautifulSoup(r.content, "html.parser")
        voitures = soup.find_all("div", class_="ListItem_wrapper__TxHWu")
        for voiture in voitures:
            image = voiture.find("img")
            break
        img_url = image.attrs["src"]
        return {"pic_url": img_url}
