import pandas as pd
from sklearn.model_selection import train_test_split

from income_predictor.ml.data_utils import process_data
from income_predictor.ml.model_utils import train_model

MODELS_PATH = "model"
DATA = pd.read_csv("data/census_clean.csv")
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


if __name__ == "__main__":
    train, test = train_test_split(DATA, test_size=0.20)

    X_train, y_train, encoder, label_binarizer = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X_train, y_train, MODELS_PATH)

    for obj, name in zip(
        [model, encoder, label_binarizer], ["model", "encoder", "label_binarizer"]
    ):
        pd.to_pickle(obj, f"{MODELS_PATH}/{name}.pkl")
