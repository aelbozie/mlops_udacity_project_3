import pandas as pd
from sklearn.model_selection import train_test_split

from income_predictor.ml.data_utils import process_data
from income_predictor.ml.model_utils import (compute_model_metrics, inference,
                                             performance_by_feature_slice,
                                             train_model)

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
LABEL = "salary"

if __name__ == "__main__":
    # split
    train, test = train_test_split(DATA, test_size=0.20)

    # train
    X_train, y_train, encoder, label_binarizer = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )

    model = train_model(X_train, y_train, MODELS_PATH)

    # test performance
    X_test, y_test, _, _ = process_data(
        test,
        CAT_FEATURES,
        LABEL,
        training=False,
        encoder=encoder,
        label_binarizer=label_binarizer,
    )

    predictions = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    with open(f"{MODELS_PATH}/model_info.txt", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write(f"Metrics:\nfbeta: {fbeta}\nprecision: {precision}\nrecall: {recall}")

    # slice performance
    features_to_slice = ["education", "marital-status"]
    for feature in features_to_slice:
        performance_by_feature_slice(
            test,
            feature,
            model,
            process_data_func=process_data,
            categorical_features=CAT_FEATURES,
            label=LABEL,
            training=False,
            encoder=encoder,
            label_binarizer=label_binarizer,
        )

    # save model, encoder and binarizer
    for obj, name in zip(
        [model, encoder, label_binarizer], ["model", "encoder", "label_binarizer"]
    ):
        pd.to_pickle(obj, f"{MODELS_PATH}/{name}.pkl")
