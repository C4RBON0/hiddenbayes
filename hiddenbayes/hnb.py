# hiddenbayes/hnb.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class HiddenNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, num_hidden_states=2):
        """
        Inicializa el modelo con un número de estados ocultos.
        """
        self.num_hidden_states = num_hidden_states

    def fit(self, X, y=None):
        """
        Simula el entrenamiento del modelo.
        Por ahora, solo almacena la cantidad de datos.
        """
        self.n_samples_ = len(X)
        return self  # Importante para compatibilidad con scikit-learn

    def predict(self, X):
        """
        Devuelve una lista de ceros como predicción de prueba.
        """
        return np.zeros(len(X), dtype=int)
