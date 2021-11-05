from fastapi.testclient import TestClient

from income_predictor import __version__
from income_predictor.main import app


def test_get_index():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": f"Welcome to Census Data Income Predictor v{__version__}"
    }


def test_post_gt():
    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 46781,
        "education": "Masters",
        "education-num": 4,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 13084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=data)
        print(response.text)
    assert response.status_code == 200
    assert response.json() == {"Income": ">50K"}


def test_post_lte():
    data = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 160187,
        "education": "9th",
        "education-num": 5,
        "marital-status": "Married-spouse-absent",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 16,
        "native-country": "Jamaica",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"Income": "<=50K"}
