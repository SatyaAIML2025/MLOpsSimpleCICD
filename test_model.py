import pytest
from model import load_data, train_model, evaluate_model

def test_load_data():
    # Arrange
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_train_model():
    # Arrange
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")
    #assert hasattr(model, "score")
    #assert hasattr(model, "get_params")

def test_evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.5 # Expecting a reasonably good accuracy
