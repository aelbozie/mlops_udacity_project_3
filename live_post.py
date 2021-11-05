import json

import requests

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


if __name__ == "__main__":
    response = requests.post("https://cdip.herokuapp.com", data=json.dumps(data))
    print(f"Status Code: {response.status_code}")
    print(f"Response:{response.json()['Income']}")
