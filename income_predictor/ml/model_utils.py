import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def train_model(X_train: np.array, y_train: np.array, models_path: str):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    models_path: str
        Path to save model info
    unit_test: bool
        Flag to simplify unit test
    Returns
    -------
    model
        Trained machine learning model.
    """

    gb = GradientBoostingClassifier()

    gb_params = {
        "n_estimators": (5, 10, 100),
        "learning_rate": (0.1, 0.01, 0.001),
        "max_depth": [5, 15],
    }

    clf = GridSearchCV(gb, gb_params, n_jobs=-1)
    clf.fit(X_train, y_train)

    model_score, best_params, best_score = (
        clf.best_estimator_,
        clf.best_params_,
        clf.best_score_,
    )
    with open(f"{models_path}/model_info.txt", "w") as f:
        f.write(f"Model {model_score} score: {best_score}\n  params: {best_params}")

    return clf.best_estimator_


def compute_model_metrics(y, predictions):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    predictions : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, predictions, beta=1, zero_division=1)
    precision = precision_score(y, predictions, zero_division=1)
    recall = recall_score(y, predictions, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    predictions : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def performance_by_feature_slice(
    test: pd.DataFrame, feature: str, model, process_data_func, **kwargs
) -> None:
    """
    Performance when a categorical feature is fixed to specific value (slicing)

    Inputs
    ------
    test: pd.DataFrame
        X features and y output variable
    feature: str
        feature to slice
    model:
        Model to test
    process_data_func:
        function to process the data
    Returns
    -------
    None
    """

    with open(f"model/{feature}_slice_output.txt", "w") as f:
        f.write(f"Performance  for {feature} feature")

        for feature_slice in test[feature].unique():
            df = test[test[feature] == feature_slice]
            X_test, y_test, _, _ = process_data_func(df=df, **kwargs)
            predictions = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)

            f.write("\n")
            f.write("\n")
            f.write(f"{feature_slice}:")
            f.write("\n")
            f.write(f"fbeta:      {fbeta}")
            f.write("\n")
            f.write(f"precision:  {precision}")
            f.write("\n")
            f.write(f"recall:     {recall}")
            f.write("\n")
