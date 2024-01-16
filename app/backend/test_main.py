from fastapi.testclient import TestClient
from main import app

client = TestClient(app=app)

def test_can_call_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == "la connection à l'API fonctionne"

def test_car_pic():
    car_info = {"marque": "bmw", "modele": "320"}
    response = client.post("/car_pic", json=car_info)
    assert response.status_code == 200
    assert type(response.json()["pic_url"]) == str
    assert len(response.json()["pic_url"]) > 0

def test_car_info():
    car_info = {"marque": "bmw", "modele": "320"}
    response = client.post("/car_info", json=car_info)
    assert response.status_code == 200
    assert int(response.json()["nb_vec"]) >= 0
    assert len(response.json()["df_info"]) == 10
    assert len(response.json()["df_info"]['kilometrage']) == int(response.json()["nb_vec"])


