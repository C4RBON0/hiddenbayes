# tests/test_hnb.py
import numpy as np
from hiddenbayes.hnb import HiddenNaiveBayes

def test_hnb_initialization():
    model = HiddenNaiveBayes(num_hidden_states=3)
    assert model.num_hidden_states == 3

def test_hnb_fit():
    model = HiddenNaiveBayes()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    model.fit(X)  # Debería ejecutarse sin errores
    assert model.n_samples_ == len(X)

def test_hnb_predict():
    model = HiddenNaiveBayes()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y_pred = model.predict(X)
    assert len(y_pred) == len(X)
    assert all(y == 0 for y in y_pred)  # Debería predecir solo ceros
