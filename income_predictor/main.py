import os
import pickle

import pandas as pd
from fastapi import FastAPI

from income_predictor import __version__
from income_predictor.ml.data_utils import process_data
from income_predictor.ml.model_utils import inference
from income_predictor.pydantic_models import CensusData, Income

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI(
    title="Census Data Income Predictor",
    description="Income classification from census data.",
    version=__version__,
)
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


@app.on_event("startup")
def startup_event():
    """
    Additionally load model and encoder on startup for faster predictions
    """

    with open("model/encoder.pkl", "rb") as f:
        global ENCODER
        ENCODER = pickle.load(f)
    with open("model/model.pkl", "rb") as f:
        global MODEL
        MODEL = pickle.load(f)


@app.get("/")
def welcome() -> str:
    return {"message": f"Welcome to Census Data Income Predictor v{__version__}"}


@app.post("/predict", response_model=Income)
def predict(payload: CensusData):
    df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, training=False, encoder=ENCODER
    )
    pred = inference(MODEL, X)

    if pred == 1:
        pred = ">50K"
    elif pred == 0:
        pred = "<=50K"
    return {"Income": pred}
