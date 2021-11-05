import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from income_predictor.ml.data_utils import process_data
from income_predictor.ml.model_utils import compute_model_metrics, inference
from income_predictor.ml.train_model import CAT_FEATURES, DATA


@pytest.fixture(scope="module")
def X():
    train, _ = train_test_split(DATA, test_size=0.20)
    X, _, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X


@pytest.fixture(scope="module")
def y():
    train, _ = train_test_split(DATA, test_size=0.20)
    _, y, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return y


@pytest.fixture(scope="module")
def model(X, y):
    dummy = DummyClassifier()
    dummy.fit(X, y)
    return dummy


@pytest.fixture(scope="module")
def preds(model, X):
    preds = inference(model, X)
    return preds


def test_compute_model_metrics_count(preds, y):
    metrics = compute_model_metrics(y, preds)
    assert len(metrics) == 3


def test_compute_model_metrics_range(y, preds):
    metrics = compute_model_metrics(y, preds)
    result = map((lambda m: (m >= 0) and (m <= 1)), metrics)
    assert all(result)


def test_inference_shape(model, X):
    preds = inference(model, X)
    assert len(preds) == len(X)


def test_inference_values(model, X):
    preds = inference(model, X)
    assert np.all((preds == 0) | (preds == 1))
