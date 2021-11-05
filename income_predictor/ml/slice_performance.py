"""
Output: model/{feature}_slice_output.txt
"""

import pandas as pd
from data_utils import process_data
from model_utils import compute_model_metrics, inference
from train_model import CAT_FEATURES


def performance_by_feature_slice(
    data: pd.DataFrame,
    feature: str,
    model,
    encoder,
    label_binarizer,
) -> None:
    """
    Performance when a categorical feature is fixed to specific value (slicing)

    Inputs
    ------
    data: pd.DataFrame
        Pandas DataFrame to be processed as X features and y output variable
    feature: str
        feature to slice
    model:
        Model to test
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        OneHotEncoder  passed in
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        LabelBinarizer passed in

    Returns
    -------
    None
    """

    with open(f"model/{feature}_slice_output.txt", "w") as f:
        f.write(f"Performance  for {feature} feature")
        for feature_slice in data[feature].unique():
            df = data[data[feature] == feature_slice]
            X_test, y_test, _, _ = process_data(
                df,
                CAT_FEATURES,
                LABEL,
                training=False,
                encoder=encoder,
                label_binarizer=label_binarizer,
            )

            predictions = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)

            f.write("\n")
            f.write(f"{feature_slice}:")
            f.write("\n")
            f.write(f"fbeta:      {fbeta}")
            f.write("\n")
            f.write(f"precision:  {precision}")
            f.write("\n")
            f.write(f"recall:     {recall}")
            f.write("\n")


if __name__ == "__main__":
    DATA = pd.read_csv(r"data/census_clean.csv")
    MODEL = pd.read_pickle("model/model.pkl")
    ENCODER = pd.read_pickle("model/encoder.pkl")
    LABEL_BINARIZER = pd.read_pickle("model/label_binarizer.pkl")
    LABEL = "salary"

    FEATURES = ["education", "marital-status"]
    for feature in FEATURES:
        performance_by_feature_slice(DATA, feature, MODEL, ENCODER, LABEL_BINARIZER)
