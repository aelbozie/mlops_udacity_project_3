import numpy as np
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
