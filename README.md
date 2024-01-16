# pricer_auto_occasion

**Objectif** : implémenter un pricer qui retourne la valeur de marché d'un véhicule d'occasion

L'idée est de créer un dashboard sur lequel l'utilisateur pourra sélectionner le modèle de véhicule qui l'interesse (qu'il soit acheteur ou vendeur).
Via un algorithme de webscrapping, des informations relatives à ce véhicule seront récupérées sur internet et des analyses visuelles seront présentées à l'utilisateur.
Sur la base de ces données, une valeur de marché sera également proposée (calculer via un algorithme de ML).

Etapes du projet
1. Réalisation d'un POC sur jupyter notebook : [scrapping_eda](/POC/Webscrapping_&_Viz.ipynb) et [model_pricer](/POC/Model_training_&_pricer.ipynb)
3. Programmation du dashboard (sur la base des travaux développés dans les notebooks) : [script fastapi](/app/backend/main.py) et [script streamlit](/app/frontend/frontend.py), le tout dockerisé (avec docker compose)
4. Déploiement du dashboard ---> à faire

Le dashboard se compose de 4 onglets:
- l'onglet générale sur lequel l'utilisateur a accès à toutes les informations collectées dans un tableau
- l'onglet "pays d'origine" sur lequel l'utilisateur peut visualiser sur une carte le pays d'origine des véhicules disponibles
- l'onglet "analyses graphiques" qui propose un certain nombres d'analyses graphiques à partir des données collectées
- l'onglet "filtres personnalisés" qui permet à l'utilisateur de filtrer les données selon ses besoins
- l'onglet "pricer" qui renvoie une valeur de marché du véhicule à partir des inputs de l'utilisateurs (kilométrage, puissance...)

<img width="1422" alt="Vision_gene" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/9cc72df7-b105-4b01-af59-945ea74672d8">

<img width="1397" alt="Spec_geo" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/44ef26cb-1d64-4b46-894b-b5c2f9fb251b">

<img width="1006" alt="pricer" src="https://github.com/estellec18/pricer_auto_occasion/assets/126951321/ea87c0c2-8021-4a5f-b7eb-b2f76f0655ad">
