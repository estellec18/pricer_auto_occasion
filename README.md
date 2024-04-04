# pricer_auto_occasion

*Projet élaboré afin de se familiariser au webscrapping*

**OBJECTIF** : implémenter un pricer qui retourne la valeur de marché d'un véhicule d'occasion

L'idée est de créer un dashboard sur lequel l'utilisateur pourra sélectionner le modèle de véhicule qui l'interesse (qu'il soit acheteur ou vendeur).
Via un algorithme de webscrapping, des informations relatives à ce véhicule seront récupérées sur internet et des analyses visuelles seront présentées à l'utilisateur.
Sur la base de ces données, une valeur de marché sera également proposée (calculée via un algorithme de ML).

Etapes du projet
1. Réalisation d'un POC sur jupyter notebook : [scrapping_eda](/POC/Webscrapping_&_Viz.ipynb) et [model_pricer](/POC/Model_training_&_pricer.ipynb)
2. Programmation du dashboard (sur la base des travaux développés dans les notebooks) : [script fastapi](/app/backend/main.py) et [script streamlit](/app/frontend/frontend.py), le tout dockerisé (avec docker compose)

Le dashboard se compose de 4 onglets:
- l'onglet général sur lequel l'utilisateur a accès à toutes les informations collectées via le scrapping dans un tableau
- l'onglet "pays d'origine" sur lequel l'utilisateur peut visualiser sur une carte le pays d'origine des véhicules disponibles
- l'onglet "analyses graphiques" qui propose un certain nombres d'analyses graphiques à partir des données collectées
- l'onglet "filtres personnalisés" qui permet à l'utilisateur de filtrer les données selon ses besoins
- l'onglet "pricer" qui renvoie une valeur de marché du véhicule à partir des inputs de l'utilisateurs (kilométrage, puissance...)

#

<img width="1422" alt="Vision_gene" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/9cc72df7-b105-4b01-af59-945ea74672d8">

#

<img width="957" alt="geo_precis" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/e348d743-a5fc-4a86-a3f5-183fbdbd08e3">

#

<img width="1006" alt="pricer" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/ea87c0c2-8021-4a5f-b7eb-b2f76f0655ad">

#

**LIMITES** :
1. Les quantités de données récoltées pour un véhicules précis via le scrapping sont insuffisantes pour pouvoir entrainer convenablement un modèle, la marge d'erreur est donc beaucoup trop importante, ce qui rend le pricer pas suffisament pertinent.
  - une solution serait de scrapper un nombre beaucoup plus important de données, mais dans la pratique ce n'est pas très réaliste (beaucoup de temps de scrapping + source limitée car de nombreux sites empechent le scrapping)
2. Il est assez courant que la valeur d'une feature catégorielle n'apparaissent qu'une fois dans le dataset (ex: pays d'origine = Luxembourg). Dans ces cas là, on court le risque que cette valeur soit dans le test set et donc inconnue du modèle.
  - à date, pour gérer cette situation, j'ai choisi pour le one_hot_encoding l'option handle_unknown='ignore' qui met des 0 partout lorsqu'un cas inconnu est rencontré par le modèle. CEPENDANT, ce n'est pas une solution parfaite étant donné que j'ai également choisi l'option drop='first' (pour limiter la colinéarité): il est donc possible que des situations différentes soient modélisées de la meme manière (avec des 0 dans toutes les colonnes)
  - une autre solution consisterait à one_hot_encoder les variables catégorielles avant le split du dataset train/test. La littérature n'est pas claire sur le sujet, mais a priori, il n'est pas souhaitable d'intervenir sur les données avant le split sous peine de data leakage.
  - cf. solutions proposées pour le "Design Pattern 1 : Hashed Feature" dans l'ouvrage *Machine Learning Design Patterns* de V. Lakshmanan, S. Robinson & M. Munn
3. Le prix d'une voiture est lié à sa cote et donc, non seulement à des éléments tangibles (kilométrage, age, puissance), mais également à des éléments intangibles (marque, style). On est donc obligé de se baser uniquement sur des données relatives au modèle de véhicule indiqué par l'utilisateur. Cela veut dire qu'on est obligé de récupérer de la nouvelle donnée à chaque requete, et donc de réentrainer le modèle 'à l'aveugle' (input limité sur le choix du modèle / finetuning). Cette situation ne permet pas d'obtenir un résultat optimal.
  - une solution pourrait être de récupérer d'un coup, à intervalle de temps déterminé, des données concernant plein de modèles de véhicules. Un modèle unique serait alors entrainé/finetuné sur ces données et permettrait de proposer des estimations pour différents modèle de véhicule. Il faudrait mettre en place un monitoring pour déterminer l'intervalle de temps adequat pour la mise à jour du modèle.
